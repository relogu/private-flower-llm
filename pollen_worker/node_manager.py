"""A highly efficient node-manager for Pollen.

The role of the node manager is to manage multiple workers on a node.
The workers are distributed over the available hardware devices
in an N:M mapping with N>=M.
The number of workers depends on:
- how many resources each client needs
- the resources available for a given device
- the parallelism supported by the system.

In order to minimize data movement and unnecessary allocations+copies
the node-manager uses a statically-assigned shared memory
to host the memory of the workers and clients.

In a single node setting, the node-manager
is the only process that runs on the node and
is only conceptually separate from the server.
In a multi node setting, each node hosts
a node-manager which communicates
to the simulation server.
"""

import ast
import multiprocessing
import pickle
from queue import Empty
import time
from collections import defaultdict
from collections.abc import Callable
from logging import DEBUG
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from socket import getfqdn
from typing import Any, cast
import uuid

import cloudpickle
import flwr as fl
import hydra
import nvsmi
import psutil
import pyarrow as pa
import torch
import transformers
from flwr.client import NumPyClient
from flwr.common import Config, NDArrays, Scalar
from flwr.common.logger import log
from hydra.utils import call
from multiprocess import set_start_method, Queue  # type: ignore[reportAttributeAccessIssue]
from nvsmi import GPU
from omegaconf import DictConfig

from pollen_worker.horovod_utils import get_free_tcp_port
from pollen_worker.pollen_utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_PARAMETERS_SHM,
    WorkerResult,
    allocate_shm,
    get_pyarrow_buffer_from_table,
    remove_shm_from_resource_tracker,
    write_to_fit_result_shm,
)
from pollen_worker.resources_manager import Device, Node, get_cpu_prop, get_cuda_prop
from pollen_worker.virtual_client import VirtualClient
from pollen_worker.worker import Worker

pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]
transformers.logging.set_verbosity_error()
set_start_method("spawn", force=True)


class NodeManager(fl.client.NumPyClient):
    """NodeManager of Pollen."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualClient],
        warm_up_config: dict[str, Scalar],
        run_uuid: str,
        placement_policy: str,
        cap_num_workers_per_gpu: int | None,
        concurrency_estimator: bool,
    ) -> None:
        super().__init__()
        self.name: str = getfqdn()
        self.warm_up_config: dict[str, Scalar] = warm_up_config
        self.properties = None
        self.all_gpus: list[GPU] = list(nvsmi.get_gpus())
        self.run_uuid = run_uuid
        self.cap_num_workers_per_gpu = cap_num_workers_per_gpu
        self.concurrency_estimator = concurrency_estimator
        self.node_manager_uuid = str(uuid.uuid4())
        # Set the auth key for the multiprocessing. Necessary for accessing the queues.
        new_auth_key = bytes(str(uuid.uuid4()), encoding="utf-8")
        multiprocessing.current_process().authkey = new_auth_key
        # Monkey-patch resource tracker to avoid tracking shared memory
        remove_shm_from_resource_tracker()
        # Set the master address and port for PyTorch distributed
        self.master_addr = "localhost"
        # NOTE: The must must be set when it is certain that it is not gonna be occupied
        # by anything else
        self.master_port: int | None = None
        # Initialize auxiliary variables
        self.assignment_config: dict[str, str] | None = None
        self.clients_training_stats: dict[str, list] | None = None
        self.num_clients_to_process: int | None = None
        self.num_clients_processed: int | None = None

        # Create multiprocessing Manager
        # One task queue per GPU/device
        self.task_queues: dict[str, QueueType] = {
            f"cuda:{gpu.id}": Queue() for gpu in self.all_gpus
        }
        # One result queue for all GPUs/devices
        self.result_queue: QueueType = Queue()
        # Workers' system metrics queue
        self.workers_system_metrics_queue: QueueType = Queue()

        # Round config is sent to shared memory
        self.config_shm: SharedMemory = SharedMemory(
            name=self.run_uuid + POLLEN_CONFIG_SHM, create=True, size=10000
        )
        # Allocate shared memory for round parameters
        self.client_fn = client_fn
        tmp_client: VirtualClient = client_fn(0)
        (
            self.round_parameters,
            self.round_num_samples,
            self.round_train_loss,
            self.round_train_acc,
            self.round_shm,
        ) = allocate_shm(
            tmp_client.get_parameters({}),  # type: ignore[reportArgumentType]
            create=True,
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )

        self.dataset_name = tmp_client.name
        # Get node properties about hardware accelerators
        self.properties = self._get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        # NOTE: We assume that the node has GPUs with the same type that can host the
        # same number of processes
        self.n_workers = 0
        self.max_proc_device: list[tuple[str, int]] = []
        if placement_policy == "llb":
            # NOTE: Parrot uses one process per GPU
            max_proc_device = [(k, 1) for k, v in self.node.device_info.items()]
            self.n_workers = len(max_proc_device)
        else:
            max_proc_device = [
                (k, v.concurrency) for k, v in self.node.device_info.items()
            ]
            self.n_workers = sum([v for _, v in max_proc_device])
        log(DEBUG, "Max processes per device: %s", max_proc_device)

        # Create worker object
        self.workers: list[Worker] = []
        for _local_rank in range(self.n_workers):
            _worker_id = str(uuid.uuid4())
            self.workers.append(
                Worker(
                    client_fn=client_fn,
                    run_uuid=self.run_uuid,
                    concurrency=self.n_workers // len(self.all_gpus),
                    dataset_name=self.dataset_name,
                    task_queues=self.task_queues,
                    result_queue=self.result_queue,
                    auth_key=new_auth_key,
                    local_rank=_local_rank,
                    world_size=self.n_workers,
                    workers_system_metrics_queue=(
                        self.workers_system_metrics_queue if _local_rank == 0 else None
                    ),
                    worker_id=_worker_id,
                )
            )
        # Start workers
        self.start_workers()
        self.workers_shms: list[SharedMemory] | None = None

    def _get_node_properties(self) -> dict[str, Scalar]:
        device_info: dict[str, Device] = {}
        # Get hardware accelerator properties
        tmp_client: VirtualClient = self.client_fn(0)
        tmp_params: NDArrays = tmp_client.get_parameters(config={})  # type: ignore[reportArgumentType]
        if torch.cuda.is_available():
            device_info = dict(
                get_cuda_prop(
                    tmp_client,
                    tmp_params,
                    config=self.warm_up_config,
                    cap_num_workers_per_gpu=self.cap_num_workers_per_gpu,
                    concurrency_estimator=self.concurrency_estimator,
                ),
                **device_info,
            )
        if torch._C._mps_is_available() and torch._C.has_mps:  # type: ignore[reportPrivateImportUsage]
            device_info = dict(
                get_cpu_prop("mps", tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        if not device_info:
            device_info = dict(
                get_cpu_prop("cpu", tmp_client, tmp_params, config=self.warm_up_config),
                **device_info,
            )
        try:
            cpus = 1
            cpus_affinity = psutil.Process().cpu_affinity()
            if cpus_affinity:
                cpus = len(cpus_affinity)
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

        return {"node": str(self.node)}

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return self.properties if self.properties else {}

    def get_parameters(self, config: Config) -> NDArrays:
        """Implement how to get parameters."""
        tmp_client: NumPyClient = self.client_fn(0)
        return tmp_client.get_parameters(config=config)  # type: ignore[reportArgumentType]

    def start_workers(
        self,
    ) -> None:
        """Launch the workers."""
        if self.master_port is None:
            self.master_port = get_free_tcp_port()
        for _worker in self.workers:
            _worker.master_address = self.master_addr
            _worker.master_port = self.master_port
            _worker.start()

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """Implement the fit step."""
        # TODO: Make this dropouts-ready
        # Extract assignments from config so that we don't need to write it to the
        # shared memory
        assignment_config: dict[str, str] = {}
        for device in self.node.device_info:
            assignment_config[device] = str(config.pop(device))
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
        for device in self.node.device_info:
            # Convert assignment to list of lists
            assignments: list[list[int]] = ast.literal_eval(assignment_config[device])
            # Count the number of clients
            num_total_virtual_clients += sum(len(_l) for _l in assignments)
            # Put each list of clients in the queue
            for list_of_clients in assignments:
                self.task_queues[device].put(list_of_clients)

        # Check if all clients have been processed
        num_processed_virtual_clients = 0
        stats = defaultdict(list)
        while num_processed_virtual_clients < num_total_virtual_clients:
            # This call is blocking and it will return the statistics
            # about client's training put in the queue by the Worker
            worker_result: WorkerResult = self.result_queue.get()
            # NOTE: Added to be compatible with the termination task's
            # return value, i.e. `[-1, 0, 0]`
            if worker_result.client_id > -1:
                stats["cid"].append(worker_result.client_id)
                stats["start_time"].append(worker_result.start_time)
                stats["end_time"].append(worker_result.end_time)
                stats["gpu"].append(worker_result.device)  # type: ignore[arg-type]
            num_processed_virtual_clients += 1
        start_time = time.time()
        self.workers_shms = []
        nm_p, nm_s_array, n_tl, n_ta, w_shm = allocate_shm(
            parameters=self.client_fn(0).get_parameters({}),  # type: ignore[reportArgumentType]
            name=self.workers[0].worker_id,
        )
        self.workers_shms.append(w_shm)
        nm_s = nm_s_array[0]
        nm_m = {"train_loss": n_tl[0], "accuracy": n_ta[0]}
        # Get workers' system metrics
        workers_system_metrics: dict[str, float] = {}
        try:
            while True:
                _metric: tuple[str, float] = self.workers_system_metrics_queue.get(
                    timeout=0.01
                )
                workers_system_metrics[_metric[0]] = _metric[1]
        except (TimeoutError, Empty):
            pass
        nm_m = nm_m | workers_system_metrics
        # Collect statistics to pyarrow.Table
        clients_training_stats = pa.Table.from_pydict(stats)
        # Add info to `clients_training_stats`
        clients_training_stats = clients_training_stats.add_column(
            0,
            "node",
            cast(pa.Array, pa.array([self.name] * len(clients_training_stats["cid"]))),
        )
        clients_training_stats = clients_training_stats.add_column(
            0,
            "server_round",
            cast(
                pa.Array,
                pa.array(
                    [int(config["server_round"])] * len(clients_training_stats["cid"])
                ),
            ),
        )
        # Prepare statistics to be sent to the server
        clients_training_buf = get_pyarrow_buffer_from_table(clients_training_stats)
        nm_m = nm_m | {"stats": clients_training_buf.to_pybytes()}
        log(
            DEBUG,
            "NodeManager %s: elaborate %s clients in %s seconds",
            self.name,
            len(clients_training_stats["cid"]),
            time.time() - start_time,
        )
        return (
            nm_p,
            int(nm_s),
            nm_m,
        )

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        return 0.0, 1, {}

    def __del__(self) -> None:
        """Implement the deletion of the NodeManager."""
        log(DEBUG, "Closing NodeManager...")
        if self.workers is not None:
            for _worker in self.workers:
                _worker.terminate()
                _worker.join()
        log(DEBUG, "Workers closed")
        # Free shared memories
        self.config_shm.close()
        self.round_shm.close()
        self.config_shm.unlink()
        self.round_shm.unlink()
        log(DEBUG, "Shared memories closed")


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Start a node manager directly with hydra."""
    warm_up_config = call(cfg.gen_on_fit_config_fn)(0)
    node_manager = NodeManager(
        client_fn=call(cfg.gen_client_fn),
        warm_up_config=warm_up_config,
        run_uuid=cfg.run_uuid,
        placement_policy=cfg.placement_policy,
        cap_num_workers_per_gpu=cfg.cap_num_workers_per_gpu,
        concurrency_estimator=cfg.concurrency_estimator,
    )

    # Start Flower client
    fl.client.start_client(
        server_address=cfg.flwr_address,
        client=node_manager.to_client(),
    )


if __name__ == "__main__":
    main()
