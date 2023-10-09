import gc
import os
import pickle
import time
from collections import defaultdict
from logging import DEBUG, ERROR
from multiprocessing import resource_tracker
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from socket import getfqdn
from typing import Callable, Dict, List, Tuple

import cloudpickle
import flwr as fl
import hydra
import multiprocess as mp
import numpy as np
import nvsmi
import psutil
import pyarrow as pa
import pynvml
import torch
import transformers
from flwr.client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from hydra.utils import call
from multiprocess import Queue, set_start_method
from nvsmi import GPU
from omegaconf import DictConfig

from pollen_utils import get_pyarrow_buffer_from_table
from resources_manager import DaemonResourcesMonitor, Node, get_cpu_prop, get_cuda_prop
from utils import get_parameters, partially_aggregate_with_metrics
from virtual_client import VirtualClient

pickle.Pickler = cloudpickle.Pickler
transformers.logging.set_verbosity_error()
set_start_method("spawn", force=True)

POLLEN_CONFIG_SHM = "pollen_config_shm"
POLLEN_PARAMETERS_SHM = "pollen_parameters_shm"
POLLEN_WORKER_SHM = "pollen_worker_"


def allocate_shm(
    parameters: NDArrays,
    create: bool = False,
    name: str = POLLEN_PARAMETERS_SHM,
) -> Tuple[NDArrays, np.ndarray, np.ndarray, np.ndarray, SharedMemory]:
    # Allocate memory for parameters and num_samples
    nbytes_params = [val.nbytes for val in parameters]
    nbytes_int = np.dtype(np.int64).itemsize
    nbytes_float = np.dtype(np.float64).itemsize
    array_bounds = [
        (sum(nbytes_params[:i]), sum(nbytes_params[: i + 1]))
        for i in range(len(nbytes_params))
    ]
    if create:
        total_num_bytes = sum(nbytes_params) + nbytes_int + 2 * nbytes_float
        shm = SharedMemory(create=True, size=total_num_bytes, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh = [
        np.ndarray(shape=x.shape, dtype=x.dtype, buffer=shm.buf[y[0] : y[1]])
        for x, y in zip(parameters, array_bounds)
    ]
    # Create shared memory for num_samples, train loss, and train accuracy
    num_samples_sh = np.ndarray(
        (1,),
        dtype=np.int64,
        buffer=shm.buf[-int(nbytes_int + 2 * nbytes_float) : -int(2 * nbytes_float)],
    )
    train_loss_sh = np.ndarray(
        (1,), dtype=np.float64, buffer=shm.buf[-int(2 * nbytes_float) : -nbytes_float]
    )
    train_accuracy_sh = np.ndarray(
        (1,), dtype=np.float64, buffer=shm.buf[-nbytes_float:]
    )
    return params_sh, num_samples_sh, train_loss_sh, train_accuracy_sh, shm


def write_to_fit_result_shm(
    buffer_backed_ndarrays: NDArrays,
    buffer_backed_num_samples: np.ndarray,
    buffer_backed_train_loss: np.ndarray,
    buffer_backed_train_accuracy: np.ndarray,
    new_ndarrays: NDArrays,
    new_num_samples: int,
    new_train_loss: float,
    new_train_accuracy: float,
) -> None:
    for i in range(len(new_ndarrays)):
        if len(new_ndarrays[i].shape) == 0:
            buffer_backed_ndarrays[i] = new_ndarrays[i]
        else:
            buffer_backed_ndarrays[i][:] = new_ndarrays[i][:]
    buffer_backed_num_samples[0] = new_num_samples
    buffer_backed_train_loss[0] = new_train_loss
    buffer_backed_train_accuracy[0] = new_train_accuracy


class Worker(mp.Process):
    def __init__(
        self,
        client_fn: Callable[[int], NumPyClient],
        device: str,
        worker_id: str,
        task_queue: QueueType,
        result_queue: QueueType,
        run_uuid: str,
        concurrency: int,
    ):
        super(Worker, self).__init__()
        self.worker_id = worker_id
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.run_uuid = run_uuid
        self.current_round: int = 0
        self.concurrency = concurrency

        # Instatiate shared memories variables
        self.config_shm = None
        (
            self.round_params,
            self.round_num_samples,
            self.round_shm,
            self.round_train_loss,
            self.round_train_acc,
        ) = (None, None, None, None, None)
        (
            self.worker_params,
            self.worker_num_samples,
            self.worker_shm,
            self.worker_train_loss,
            self.worker_train_acc,
        ) = (None, None, None, None, None)
        self.test_params = None

    def process_task(self, client_id: int):
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        # Loads a dict from the shared memory buffer
        config = pickle.loads(self.config_shm.buf)
        config["device"] = self.device

        # Load client
        tmp_client = self.client_fn(client_id=client_id)

        done = False
        while not done:
            try:
                # Call fit on shared parameters
                fit_trained_weights, fit_num_samples, train_metrics = tmp_client.fit(
                    self.round_params, config
                )
                done = True
            except Exception as e:
                log(
                    ERROR,
                    "Worker %s failed in training client %s with exception %s."
                    " Retrying...",
                    self.worker_id,
                    client_id,
                    e,
                )

        # If new round, then copy result to shared memory directly
        if config["server_round"] > self.current_round:
            self.current_round = config["server_round"]
            write_to_fit_result_shm(
                self.worker_params,
                self.worker_num_samples,
                self.worker_train_loss,
                self.worker_train_acc,
                fit_trained_weights,
                fit_num_samples,
                train_metrics["train_loss"],
                train_metrics["accuracy"],
            )
        # Partially aggregating fit results
        else:
            (
                tmp_part_agg_params,
                tmp_part_agg_num_samples,
                tmp_part_agg_loss,
                tmp_part_agg_acc,
            ) = partially_aggregate_with_metrics(
                (
                    self.worker_params,
                    self.worker_num_samples[0],
                    self.worker_train_loss[0],
                    self.worker_train_acc[0],
                ),
                (
                    fit_trained_weights,
                    fit_num_samples,
                    train_metrics["train_loss"],
                    train_metrics["accuracy"],
                ),
            )
            write_to_fit_result_shm(
                self.worker_params,
                self.worker_num_samples,
                self.worker_train_loss,
                self.worker_train_acc,
                tmp_part_agg_params,
                tmp_part_agg_num_samples,
                tmp_part_agg_loss,
                tmp_part_agg_acc,
            )
        # Take the timestamp after the task is done
        end_time = time.time_ns()
        self.result_queue.put([int(client_id), start_time, end_time])
        # NOTE: PyTorch's memory management works bad with multiprocessing. In our case,
        # it might happen that each process eagerly allocates more MBs of memory on the
        # same VRAM at the same w/o cleaning the cache because each of them thinks that
        # it is the only one using the GPU. We need to clean the cache manually to
        # prevent this, i.e. call `torch.cuda.empty_cache()`. When to call it is a
        # trade-off between performance and memory usage because cleaning the cache
        # is time expensive (for Reddit it costs ~15 seconds, quick took ~333s slow
        # took ~348s in 10 rounds, 100 clients/round, 1 A40). We call it after each
        # client, but it might be better to call it after each round.
        for dev_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if os.getpid() == proc.pid:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if proc.usedGpuMemory > mem.total / self.concurrency:
                        torch.cuda.empty_cache()
                        gc.collect()

    def run(self):
        # Allocate shared memories.
        # NOTE: This goes here because it needs to be done in the child process!
        # This is the first piece of code of the worker that live in the child
        # process, the `__init__()` function does not.
        # NOTE: This is the NodeManager's shared memory for the fit config
        # dictionary. Workers should only read this. NodeManager should only
        # write this.
        self.config_shm = SharedMemory(name=self.run_uuid + POLLEN_CONFIG_SHM)
        # NOTE: This is the NodeManager's shared memory for the fit results.
        # Workers should only read this. NodeManager should only write this.
        (
            self.round_params,
            self.round_num_samples,
            self.round_train_loss,
            self.round_train_acc,
            self.round_shm,
        ) = allocate_shm(
            parameters=self.client_fn(0).get_parameters({}),
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )
        # NOTE: This is the Worker's shared memory for the fit results.
        # NodeManager should only read this. Worker should only write this.
        (
            self.worker_params,
            self.worker_num_samples,
            self.worker_train_loss,
            self.worker_train_acc,
            self.worker_shm,
        ) = allocate_shm(
            parameters=self.client_fn(0).get_parameters({}),
            name=self.worker_id,
        )
        # NOTE: This is for controlling the GPU memory allocation
        pynvml.nvmlInit()
        # Task loop
        for task in iter(self.task_queue.get, None):
            self.process_task(task)
        # Put the closing task's results in the result queue
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
        self.task_queues: Dict[str, QueueType] = {
            f"cuda:{gpu.id}": Queue() for gpu in self.all_gpus
        }
        self.result_queue: QueueType = Queue()  # One result_queue for all GPUs

        # Round config is sent to shared memory
        self.config_shm: SharedMemory = SharedMemory(
            name=self.run_uuid + POLLEN_CONFIG_SHM, create=True, size=10000
        )
        # Allocate shared memory for round parameters
        self.client_fn = client_fn
        tmp_client: VirtualClient = client_fn(client_id=0)
        (
            self.round_parameters,
            self.round_num_samples,
            self.round_train_loss,
            self.round_train_acc,
            self.round_shm,
        ) = allocate_shm(
            tmp_client.get_parameters({}),
            create=True,
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )
        # Get node properties about hardware accelerators
        self.properties = self.get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        max_proc_device = [(k, v.concurrency) for k, v in self.node.device_info.items()]
        # max_proc_device = [('cuda:0', 1)]
        # log(DEBUG, f"Node {self.name} has max_proc_device {max_proc_device}")

        # Allocate shared memory for partial aggregation
        # and create workers
        worker_cnt = 0
        self.workers: Dict[str, List[Worker]] = defaultdict(list)
        self.shared_local_agg: Dict[str, List[NDArrays, int, SharedMemory]] = {}
        for device, num_proc in max_proc_device:
            for _ in range(num_proc):
                worker_id = self.run_uuid + POLLEN_WORKER_SHM + f"{worker_cnt}"
                params, num_samples, train_loss, train_acc, shm = allocate_shm(
                    tmp_client.get_parameters({}),
                    create=True,
                    name=worker_id,
                )
                num_samples[0] = 0
                self.shared_local_agg[worker_id] = [
                    params,
                    num_samples,
                    train_loss,
                    train_acc,
                    shm,
                ]
                self.workers[device].append(
                    Worker(
                        client_fn=client_fn,
                        device=device,
                        worker_id=worker_id,
                        task_queue=self.task_queues[device],
                        result_queue=self.result_queue,
                        run_uuid=self.run_uuid,
                        concurrency=num_proc,
                    )
                )
                worker_cnt += 1
        # Start all the workers
        self.start_workers({})

    def get_node_properties(self) -> Dict[str, Scalar]:
        device_info = {}
        # Get hardware accelerator properties
        tmp_client: VirtualClient = self.client_fn(client_id=0)
        tmp_params = tmp_client.get_parameters(config={})
        if torch.cuda.is_available():
            # log(INFO, f"Node {getfqdn()}, CUDA acceleration available.")
            device_info = dict(
                get_cuda_prop(tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        if torch._C._is_mps_available() and torch._C.has_mps():
            # log(INFO, f"Node {getfqdn()}, MPS acceleration available.")
            device_info = dict(
                get_cpu_prop("mps", tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        if not device_info:
            # log(
            #     INFO,
            #     "Node %s, No hardware accelerator available."
            #     " Assessing CPU execution.",
            #     self.name
            # )
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
        # log(DEBUG, f"Node {getfqdn()} has complete properties {self.node}")

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
        if self.monitor is None:
            gpu_ids = [gpu.id for _, gpu in self.node.device_info.items()]
            self.monitor = DaemonResourcesMonitor(gpu_ids=gpu_ids)
            self.monitor.start()

    def fit(self, parameters, config):
        # Update shared memories objects
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        self.config_shm.buf[: len(config_bytes)] = config_bytes
        write_to_fit_result_shm(
            self.round_parameters,
            self.round_num_samples,
            self.round_train_loss,
            self.round_train_acc,
            parameters,
            0,
            0.0,
            0.0,
        )

        # Send parameters to shared memory
        num_total_virtual_clients = 0
        self.monitor.gpu_stats = []
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            num_total_virtual_clients += len(list_ids_for_this_gpu)

            # Close useless workers, one by one
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
                            self.shared_local_agg[worker.worker_id][4].close()
                            self.shared_local_agg[worker.worker_id][4].unlink()
                            # Remove the shared memory from the dict
                            del self.shared_local_agg[worker.worker_id]
                            # Remove the worker from the list
                            self.workers[device].pop(i)
                            flag = False
                            break
            # Put the client ids in the queue
            for cid in list_ids_for_this_gpu:
                self.task_queues[device].put(cid)

        # Create cid->GPU mapping
        cid_gpu_mapping = {}
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            cid_gpu_mapping.update({cid: device for cid in list_ids_for_this_gpu})

        # Check if all clients have been processed
        num_processed_virtual_clients = 0
        stats = defaultdict(list)
        while num_processed_virtual_clients < num_total_virtual_clients:
            # This call is blocking and it will return the statistics
            # about client's training put in the queue by the Worker
            current_stats = self.result_queue.get()
            # NOTE: Added to be compatible with the termination task's
            # return value, i.e. `[-1, 0, 0]`
            if current_stats[0] > -1:
                stats["cid"].append(current_stats[0])
                stats["start_time"].append(current_stats[1])
                stats["end_time"].append(current_stats[2])
            num_processed_virtual_clients += 1
        # Collect statistics to pyarrow.Table
        gpu_stats = pa.concat_tables(self.monitor.gpu_stats)
        clients_training_stats = pa.Table.from_pydict(stats)
        # Add info to `clients_training_stats`
        clients_training_stats = clients_training_stats.add_column(
            0,
            "gpu",
            pa.array(
                [cid_gpu_mapping[str(cid)] for cid in clients_training_stats["cid"]]
            ),
        )
        clients_training_stats = clients_training_stats.add_column(
            0, "node", pa.array([self.name] * len(clients_training_stats["cid"]))
        )
        # Prepare statistics to be sent to the server
        gpu_buf = get_pyarrow_buffer_from_table(gpu_stats)
        clients_training_buf = get_pyarrow_buffer_from_table(clients_training_stats)
        ## Node aggregation
        node_trained_params = aggregate(
            [(val[0], val[1][0]) for val in self.shared_local_agg.values()]
        )
        node_n_samples = sum([val[1][0] for val in self.shared_local_agg.values()])
        node_train_loss = weighted_loss_avg(
            [(val[1][0], val[2][0]) for val in self.shared_local_agg.values()]
        )
        node_accuracy = weighted_loss_avg(
            [(val[1][0], val[3][0]) for val in self.shared_local_agg.values()]
        )
        # Reset shared memories
        for val in self.shared_local_agg.values():
            val[4].buf[:] = b"\0" * val[4].size
        return (
            node_trained_params,
            int(node_n_samples),
            {
                "train_loss": node_train_loss,
                "accuracy": node_accuracy,
                "stats": clients_training_buf.to_pybytes(),
                "gpu_stats": gpu_buf.to_pybytes(),
            },
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        log(DEBUG, "Closing stuff")
        # Close monitor
        while self.monitor.is_alive():
            self.monitor.do_run = False
            time.sleep(0.1)
        del self.monitor
        log(DEBUG, "Monitor closed")
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
        log(DEBUG, "Workers closed")
        # Free shared memories
        self.config_shm.close()
        self.round_shm.close()
        self.config_shm.unlink()
        self.round_shm.unlink()
        for i, v in enumerate(self.shared_local_agg.values()):
            v[4].close()
            v[4].unlink()
        log(DEBUG, "Shared memories closed")


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
