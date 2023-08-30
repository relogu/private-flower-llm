import pickle
import time
from collections import defaultdict
from logging import DEBUG, INFO
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from socket import getfqdn

import cloudpickle
from flwr.client import NumPyClient
from flwr.common.logger import log

pickle.Pickler = cloudpickle.Pickler
from typing import Callable, Dict, List, Tuple

import flwr as fl
import hydra
import multiprocess as mp
import nvsmi
import psutil
from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from nvsmi import GPU
from omegaconf import DictConfig

from resources_manager import Node, ResourcesMonitor, get_cpu_prop, get_cuda_prop
from utils import get_parameters, partially_aggregate
from virtual_client import VirtualClient

mp.set_start_method("spawn", force=True)
import numpy as np
import torch

POLLEN_CONFIG_SHM = "pollen_config_shm"
POLLEN_PARAMETERS_SHM = "pollen_parameters_shm"
POLLEN_WORKER_SHM = "pollen_worker_"


def allocate_shm(
    parameters: NDArrays,
    create: bool = False,
    name: str = POLLEN_PARAMETERS_SHM,
) -> Tuple[NDArrays, np.ndarray, SharedMemory]:
    # Allocate memory for parameters and num_samples
    nbytes_params = [val.nbytes for val in parameters]
    nbytes_int = np.dtype(np.int64).itemsize
    array_bounds = [
        (sum(nbytes_params[:i]), sum(nbytes_params[: i + 1]))
        for i in range(len(nbytes_params))
    ]
    if create:
        total_num_bytes = sum(nbytes_params) + nbytes_int
        shm = SharedMemory(create=True, size=total_num_bytes, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh = [
        np.ndarray(shape=x.shape, dtype=x.dtype, buffer=shm.buf[y[0] : y[1]])
        for x, y in zip(parameters, array_bounds)
    ]
    # Create shared memory for num_samples
    num_samples_sh = np.ndarray((1,), dtype=np.int64, buffer=shm.buf[-nbytes_int:])
    return params_sh, num_samples_sh, shm


def copy_params_to_shm(
    parameters: NDArrays,
    num_samples: int,
    shm_name: str,
) -> None:
    # Allocate memory for parameters and num_samples
    nbytes_int = np.dtype(np.int64).itemsize
    shm = SharedMemory(name=shm_name, create=False)
    shm.buf[:-nbytes_int] = b"".join([a.tobytes() for a in parameters])
    tmp_np = num_samples * np.ones((1,), dtype=np.int64)
    shm.buf[-nbytes_int:] = tmp_np.tobytes()


class Worker(mp.Process):
    def __init__(
        self,
        client_fn: Callable[[int], NumPyClient],
        device: str,
        worker_id: str,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        run_uuid: str,
    ):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue: mp.Queue = task_queue
        self.result_queue: mp.Queue = result_queue
        self.run_uuid = run_uuid
        self.current_round: int = 0
        self.config_shm = SharedMemory(name=self.run_uuid + POLLEN_CONFIG_SHM)
        tmp_client = client_fn(client_id=0)

        # Allocate shared memory for fit parameters
        self.round_params, self.round_num_samples, self.round_shm = allocate_shm(
            parameters=tmp_client.get_parameters({}),
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )

    def process_task(self, client_id: int):
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        config = pickle.loads(self.config_shm.buf)  # Loads a dict
        config["device"] = self.device

        # Load model from shared memory
        tmp_client = self.client_fn(client_id=client_id)

        # Load client and call fit on shared parameters
        fit_trained_weights, fit_num_samples, _ = tmp_client.fit(
            self.round_params, config
        )

        # if new round, then copy result to shared memory directly
        if config["server_round"] > self.current_round:
            self.current_round = config["server_round"]
            copy_params_to_shm(fit_trained_weights, fit_num_samples, self.worker_id)

        else:  # partially aggregate #FIX
            # Existing
            params_shm, num_samples_shm, _ = allocate_shm(
                fit_trained_weights, create=False, name=self.worker_id
            )
            tmp_part_agg_params, tmp_part_agg_num_samples = partially_aggregate(
                (params_shm, num_samples_shm[0]),
                (fit_trained_weights, fit_num_samples),
            )
            copy_params_to_shm(
                tmp_part_agg_params,
                tmp_part_agg_num_samples,
                self.worker_id,
            )
        # Take the timestamp after the task is done
        end_time = time.time_ns()
        self.result_queue.put([int(client_id), start_time, end_time])

    def run(self):
        for task in iter(self.task_queue.get, None):
            self.process_task(task)
        self.result_queue.put([-1, 0, 0])
        # Un-register shared memories
        # NOTE: Bug https://bugs.python.org/issue39959#msg364351
        resource_tracker.unregister(self.config_shm._name, "shared_memory")
        resource_tracker.unregister(self.round_shm._name, "shared_memory")
        resource_tracker.unregister(
            SharedMemory(name=self.worker_id)._name, "shared_memory"
        )


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(
        self,
        client_fn: Callable[[int], NumPyClient],
        warm_up_config: Dict[str, Scalar],
        run_uuid: str,
    ) -> None:
        super().__init__()
        self.name: str = getfqdn()
        self.warm_up_config: Dict[str, Scalar] = warm_up_config
        self.properties = None
        self.all_gpus: List[GPU] = nvsmi.get_gpus()
        self.monitor = None
        self.run_uuid = run_uuid

        # One task_queue per GPU make this ctypes array
        self.task_queues = {f"cuda:{gpu.id}": mp.Queue() for gpu in self.all_gpus}
        self.result_queue = mp.Queue()  # One result_queue for all GPUs

        # Round config is sent to shared memory
        self.config_shm: SharedMemory = SharedMemory(
            name=self.run_uuid + POLLEN_CONFIG_SHM, create=True, size=10000
        )
        # Allocate shared memory for round parameters
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        tmp_client: VirtualClient = client_fn(client_id=0)
        self.round_parameters, self.round_num_samples, self.round_shm = allocate_shm(
            tmp_client.get_parameters({}),
            create=True,
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )
        # Get node properties about hardware accelerators
        self.properties = self.get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        max_proc_device = [(k, v.concurrency) for k, v in self.node.device_info.items()]
        # max_proc_device = [("cuda:0", 1)]
        log(DEBUG, f"Node {self.name} has max_proc_device {max_proc_device}")

        # Allocate shared memory for partial aggregation
        # and create workers
        worker_cnt = 0
        self.workers: Dict[str, List[Worker]] = defaultdict(list)
        self.shared_local_agg: Dict[str, List[NDArrays, int, SharedMemory]] = {}
        for device, num_proc in max_proc_device:
            for _ in range(num_proc):
                worker_id = self.run_uuid + POLLEN_WORKER_SHM + f"{worker_cnt}"
                params, num_samples, shm = allocate_shm(
                    tmp_client.get_parameters({}),
                    create=True,
                    name=worker_id,
                )
                num_samples[0] = 0
                self.shared_local_agg[worker_id] = [params, num_samples, shm]
                self.workers[device].append(
                    Worker(
                        client_fn=client_fn,
                        device=device,
                        worker_id=worker_id,
                        task_queue=self.task_queues[device],
                        result_queue=self.result_queue,
                        run_uuid=self.run_uuid,
                    )
                )
                worker_cnt += 1
        # Start all the workers
        self.start_workers({})

    def get_node_properties(self) -> Dict[str, Scalar]:
        device_info = {}
        # Get hardware accelerator properties
        tmp_client = self.client_fn(client_id=0)
        tmp_params = tmp_client.get_parameters(config={})
        if torch.cuda.is_available():
            # log(INFO, f"Node {getfqdn()}, CUDA acceleration available.")
            device_info = dict(
                get_cuda_prop(tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            # log(INFO, f"Node {getfqdn()}, MPS acceleration available.")
            device_info = dict(
                get_cpu_prop("mps", tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        if not device_info:
            log(
                INFO,
                f"Node {self.name}, No hardware accelerator available. Assessing CPU execution.",
            )
            device_info = get_cpu_prop("cpu")
        try:
            cpus = len(psutil.Process().cpu_affinity())
        except AttributeError:
            cpus = psutil.cpu_count()
        # Get general node properties
        self.node = Node(
            name=getfqdn(),
            cpu_num=cpus,
            cpu_ram_total=psutil.virtual_memory().total,
            cpu_ram_available=psutil.virtual_memory().total
            - psutil.virtual_memory().used,
            device_info=device_info,
        )
        log(DEBUG, f"Node {getfqdn()} has complete properties {self.node}")

        return {"node": str(self.node)}

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return self.properties

    def get_parameters(self, config):
        tmp_client = self.client_fn(client_id=0)
        return get_parameters(tmp_client.net)

    def start_workers(self, config):
        for device, worker_list in self.workers.items():
            for worker in worker_list:
                if not worker.is_alive():
                    worker.start()
        # Launch monitor
        # TODO: Read the CUDA id from the devices
        if self.monitor is None:
            self.monitor = ResourcesMonitor(gpu_id=0)
            self.monitor.start()

    def fit(self, parameters, config):
        # Send config and parameters to shared memory
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        self.config_shm.buf[: len(config_bytes)] = config_bytes
        copy_params_to_shm(parameters, 0, self.run_uuid + POLLEN_PARAMETERS_SHM)

        # Send parameters to shared memory
        num_total_virtual_clients = 0
        self.monitor.gpu_stats = []
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            num_total_virtual_clients += len(list_ids_for_this_gpu)

            # Close useless workers
            while len(list_ids_for_this_gpu) < len(self.workers[device]):
                # Put a None for a worker to terminate it
                self.task_queues[device].put(None)
                #  Wait for the results of the termination task
                self.result_queue.get()
                # Handle which worker has died
                flag = True
                while flag:
                    for i, worker in enumerate(self.workers[device]):
                        if not worker.is_alive():
                            # Close and unlink the shared memory
                            self.shared_local_agg[worker.worker_id][2].close()
                            self.shared_local_agg[worker.worker_id][2].unlink()
                            # Remove the shared memory from the dict
                            del self.shared_local_agg[worker.worker_id]
                            # Remove the worker from the list
                            self.workers[device].pop(i)
                            flag = False
                            break
            # Put the client ids in the queue
            for cid in list_ids_for_this_gpu:
                self.task_queues[device].put(cid)

        # Start workers
        if config["server_round"] == 1:
            self.start_workers(config)

        # Check if all clients have been processed
        num_processed_virtual_clients = 0
        stats = []
        while num_processed_virtual_clients < num_total_virtual_clients:
            # This call is blocking and it will return the statistics
            # about client's training put in the queue by the Worker
            current_stats = self.result_queue.get()
            # NOTE: Added to be compatible with the termination task's
            # return value ([-1, 0, 0])
            if current_stats[0] > -1:
                stats.append(current_stats)
            num_processed_virtual_clients += 1
        # Collect statistics
        gpu_stats = self.monitor.gpu_stats
        self.monitor.gpu_stats = []

        # Aggregate partially aggregated results from workers (if possible)
        # Aggregate only valid num_samples>0 or risk div 0
        node_part_agg = (None, 0)
        for w_params, w_num_samples_np, w_shm in self.shared_local_agg.values():
            if w_num_samples_np[0] > 0:
                node_part_agg = partially_aggregate(
                    node_part_agg, (w_params, w_num_samples_np[0])
                )
            w_shm.buf[:] = b"\0" * w_shm.size
        return (
            node_part_agg[0],
            int(node_part_agg[1]),
            {"accuracy": 0.0, "stats": str(stats), "gpu_stats": str(gpu_stats)},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        log(DEBUG, f"Closing stuff")
        # Close monitor
        while self.monitor.is_alive():
            self.monitor.do_run = False
            time.sleep(0.1)
        del self.monitor
        log(DEBUG, f"Monitor closed")
        if self.workers is not None:
            for device, list_of_workers in self.workers.items():
                [
                    self.task_queues[device].put(None)
                    for _ in range(len(list_of_workers))
                ]
        # Wait for workers to finish
        a = 0
        while a < len(self.workers[device]):
            self.result_queue.get()
            a += 1
        log(DEBUG, f"Workers closed")
        # Free shared memories
        self.config_shm.close()
        self.round_shm.close()
        self.config_shm.unlink()
        self.round_shm.unlink()
        for i, v in enumerate(self.shared_local_agg.values()):
            v[2].close()
            v[2].unlink()
        log(DEBUG, f"Shared memories closed")


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    # Start NodeManager
    warm_up_config = call(cfg.gen_on_fit_config_fn)(0)
    node_manager = NodeManager(
        client_fn=call(cfg.gen_client_fn),
        warm_up_config=warm_up_config,
        run_uuid=cfg.run_uuid,
    )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=cfg.flwr_address,
        client=node_manager,
    )


if __name__ == "__main__":
    main()
