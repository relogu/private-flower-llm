from logging import DEBUG
import pickle
from collections import OrderedDict
from multiprocessing.shared_memory import SharedMemory

import cloudpickle
from flwr.client import NumPyClient
from flwr.common.logger import log

pickle.Pickler = cloudpickle.Pickler
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import multiprocess as mp
import nvsmi
import psutil
from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from nvsmi import GPU
from omegaconf import DictConfig

from resources_manager import get_node_manager_properties
from utils import get_parameters, partially_aggregate

mp.set_start_method("spawn", force=True)
import numpy as np

# POLLEN_CONFIG_SHM = "ls985_pollen_config_shm"
# POLLEN_PARAMETERS_SHM = "ls985_pollen_parameters_shm"
# POLLEN_WORKER_SHM = "ls985_pollen_worker_"

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
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
    ):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue: mp.Queue = task_queue
        self.result_queue: mp.Queue = result_queue
        self.current_round: int = 0
        self.config_shm = SharedMemory(name=POLLEN_CONFIG_SHM)
        tmp_client = client_fn(client_id=0)

        # Allocate shared memory for fit parameters
        self.round_params, self.round_num_samples, self.round_shm = allocate_shm(
            parameters=get_parameters(tmp_client.net),
            name=POLLEN_PARAMETERS_SHM,
        )

    def process_task(self, client_id: int):
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

        self.result_queue.put(1)

    def run(self):
        for task in iter(self.task_queue.get, None):
            self.process_task(task)

        # Free shared memory
        self.config_shm.close()
        self.round_shm.close()


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self, client_fn) -> None:
        super().__init__()
        self.properties = None
        self.all_gpus: List[GPU] = nvsmi.get_gpus()

        # One task_queue per GPU make this ctypes array
        self.task_queues = {f"cuda:{gpu.id}": mp.Queue() for gpu in self.all_gpus}
        self.result_queue = mp.Queue()  # One result_queue for all GPUs

        # Round config is sent to shared memory
        self.config_shm: SharedMemory = SharedMemory(
            name=POLLEN_CONFIG_SHM, create=True, size=1000
        )
        # Allocate shared memory for round parameters
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        temp_client = client_fn(client_id=0)
        self.round_parameters, self.round_num_samples, self.round_shm = allocate_shm(
            get_parameters(temp_client.net), create=True
        )
        # Find out how many processes can be run on each GPU
        max_proc_device = [("cuda:0", 10)]

        # Allocate shared memory for partial aggregation
        # and create workers
        worker_cnt = 0
        self.workers: Dict[str, List[Worker]] = {}
        self.shared_local_agg: Dict[str, List[NDArrays, int, SharedMemory]] = {}
        for device, num_proc in max_proc_device:
            self.workers[device] = []
            for i in range(num_proc):
                # worker_id = f"pollen_worker_{worker_cnt}"
                worker_id = POLLEN_WORKER_SHM + f"{worker_cnt}"
                params, num_samples, shm = allocate_shm(
                    get_parameters(temp_client.net),
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
                    )
                )
                worker_cnt += 1

        # # Get properties
        # self.properties = get_node_manager_properties()

    def get_cpu_prop(self) -> Dict[str, float]:
        node_prop = {
            "cpu_num": mp.cpu_count(),
            "cpu_ram_total": psutil.virtual_memory().total,
            "cpu_ram_available": psutil.virtual_memory().available,
        }
        return node_prop

    def get_gpus_prop(self) -> Dict[str, float]:
        gpus_prop = {}
        for gpu in nvsmi.get_gpus():
            if gpu["memory.free"] > 0:
                gpus_prop[f"cuda:{gpu.id}_ram_total"] = gpu.mem_total
        return gpus_prop

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # cpu_prop = self.get_cpu_prop()
        # gpus_prop = self.get_gpus_prop()
        # return dict(cpu_prop, **gpus_prop)
        return (
            self.properties
            if self.properties is not None
            else get_node_manager_properties()
        )

    def get_parameters(self, config):
        temp_client = self.client_fn(client_id=0)
        return get_parameters(temp_client.net)

    def start_workers(self, config):
        for worker_list in self.workers.values():
            for worker in worker_list:
                worker.start()

    def fit(self, parameters, config):
        # Send config and parameters to shared memory
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        self.config_shm.buf[: len(config_bytes)] = config_bytes
        copy_params_to_shm(parameters, 0, POLLEN_PARAMETERS_SHM)

        # Send parameters to shared memory
        num_total_virtual_clients = 0
        num_total_workers = 0
        for device in self.workers.keys():
            num_total_workers += len(self.workers[device])
            list_ids_for_this_gpu = config[device].split(",")
            num_total_virtual_clients += len(list_ids_for_this_gpu)

            for cid in list_ids_for_this_gpu:
                self.task_queues[device].put(cid)

            # Start workers in this GPU move this outside
            if config["server_round"] == 1:
                self.start_workers(config)

        # Check if all clients have been processed
        num_processed_virtual_clients = 0
        while num_processed_virtual_clients < num_total_virtual_clients:
            num_processed_virtual_clients += self.result_queue.get()

        # Aggregate partially aggregated results from workers (if possible)
        # Aggregate only valid num_samples>0 or risk div 0
        node_part_agg = (None, 0)
        for w_params, w_num_samples_np, w_shm in self.shared_local_agg.values():
            node_part_agg = partially_aggregate(
                node_part_agg, (w_params, w_num_samples_np[0])
            )
            w_shm.buf[:] = b"\0" * w_shm.size
        return node_part_agg[0], int(node_part_agg[1]), {"accuracy": 0.0}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.workers is not None:
            for device, list_of_workers in self.workers.items():
                [
                    self.task_queues[device].put(None)
                    for _ in range(len(list_of_workers))
                ]
        # Free shared memory
        self.config_shm.close()
        self.round_shm.close()
        for v in self.shared_local_agg.values():
            v[2].close()

@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Start NodeManager
    node_manager = NodeManager(client_fn=call(cfg.gen_client_fn))

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=cfg.flwr_address,
        client=node_manager,
    )


if __name__ == "__main__":
    main()
