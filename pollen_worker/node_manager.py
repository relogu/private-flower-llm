"""A highly efficient node-manager for Pollen.

The role of the node manager is to manage multiple workers on a node.
The workers are distributed over the available hardware devices
in an N:M mapping with N>=M.
The number of workers depends on:
- how many resources each client needs
- the resources availabe for a given device
- the parallelism supported by the system.

In order to minimize data movement and unnecessary allocations+copies
the node-manager uses a statically-assigned shared memory
to host the memory of the workers and clients.

In a singlenode setting, the node-manager
is the only process that runs on the node and
is only conceptually separate from the server.
In a multinode setting, each node hosts
a node-manager which communicates
to the simulation server.
"""

import ast
import pickle
import time
from collections import defaultdict
from collections.abc import Callable
from logging import DEBUG
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from socket import getfqdn
from typing import Any, cast

import cloudpickle
import flwr as fl
import hydra
import numpy as np
import nvsmi
import psutil
import pyarrow as pa
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

from pollen_worker.pollen_utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_PARAMETERS_SHM,
    POLLEN_WORKER_SHM,
    allocate_shm,
    get_client_ds_fn,
    get_pyarrow_buffer_from_table,
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
    ) -> None:
        super().__init__()
        self.name: str = getfqdn()
        self.warm_up_config: dict[str, Scalar] = warm_up_config
        self.properties = None
        self.all_gpus: list[GPU] = list(nvsmi.get_gpus())
        self.run_uuid = run_uuid

        # One task_queue per GPU make this ctypes array
        self.task_queues: dict[str, QueueType] = {
            f"cuda:{gpu.id}": Queue() for gpu in self.all_gpus
        }
        self.result_queue: QueueType = Queue()  # One result_queue for all GPUs

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
            tmp_client.get_parameters({}),
            create=True,
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )

        self.dataset_name = tmp_client.name
        # Get node properties about hardware accelerators
        self.properties = self._get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        if placement_policy == "llb":
            # NOTE: Parrot uses one process per GPU
            max_proc_device = [(k, 1) for k, v in self.node.device_info.items()]
        else:
            max_proc_device = [
                (k, v.concurrency) for k, v in self.node.device_info.items()
            ]
        log(DEBUG, "Max processes per device: %s", max_proc_device)

        # Allocate shared memory for partial aggregation
        # and create workers
        worker_cnt = 0
        self.workers: dict[str, list[Worker]] = defaultdict(list)
        self.shared_local_agg: dict[
            str,
            tuple[
                NDArrays,
                np.ndarray[Any, np.dtype[Any]],
                np.ndarray[Any, np.dtype[Any]],
                np.ndarray[Any, np.dtype[Any]],
                SharedMemory,
            ],
        ] = {}
        for device, num_proc in max_proc_device:
            for _ in range(num_proc):
                worker_id = self.run_uuid + POLLEN_WORKER_SHM + f"{worker_cnt}"
                params, num_samples, train_loss, train_acc, shm = allocate_shm(
                    tmp_client.get_parameters({}),
                    create=True,
                    name=worker_id,
                )
                num_samples[0] = 0
                self.shared_local_agg[worker_id] = (
                    params,
                    num_samples,
                    train_loss,
                    train_acc,
                    shm,
                )
                self.workers[device].append(
                    Worker(
                        client_fn=client_fn,
                        dataset_generator=lambda cid: get_client_ds_fn(
                            name=self.dataset_name
                        )(cid)[0],
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
        self._start_workers({})

    def _get_node_properties(self) -> dict[str, Scalar]:
        device_info: dict[str, Device] = {}
        # Get hardware accelerator properties
        tmp_client: VirtualClient = self.client_fn(0)
        tmp_params = tmp_client.get_parameters(config={})
        if torch.cuda.is_available():
            device_info = dict(
                get_cuda_prop(
                    tmp_client, tmp_params, config=self.warm_up_config, cap_workers=1
                ),
                **device_info,
            )
        if torch._C._mps_is_available() and torch._C.has_mps:
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

        return {"node": str(self.node)}

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return self.properties if self.properties else {}

    def get_parameters(self, config: Config) -> NDArrays:
        """Implement how to get parameters."""
        tmp_client: NumPyClient = self.client_fn(0)
        return tmp_client.get_parameters(config=config)

    def _start_workers(self, config: Config) -> None:
        for worker_list in self.workers.values():
            for worker in worker_list:
                if not worker.is_alive():
                    worker.start()

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """Implement the fit step."""
        # TODO: Make this dropouts-ready
        # Extract assignments from config
        assignment_config: dict[str, str] = {}
        for device in self.workers:
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
        for device, workers in self.workers.items():
            # list_ids_for_this_gpu = cast(str, assignment_config[device]).split(",")
            list_ids_for_this_gpu: list[list[int]] = ast.literal_eval(
                assignment_config[device]
            )
            num_total_virtual_clients += sum(len(_l) for _l in list_ids_for_this_gpu)

            # Close useless workers, one by one
            while len(list_ids_for_this_gpu) < len(workers):
                # Put a None for a worker to terminate it
                self.task_queues[device].put(None)
                #  Wait for the results of the termination task
                self.result_queue.get()
                # Handle which worker has died
                flag = True
                while flag:
                    for i, worker in enumerate(workers):
                        if not worker.is_alive():
                            # Close and unlink the shared memory
                            self.shared_local_agg[worker.worker_id][4].close()
                            self.shared_local_agg[worker.worker_id][4].unlink()
                            # Remove the shared memory from the dict
                            del self.shared_local_agg[worker.worker_id]
                            # Remove the worker from the list
                            workers.pop(i)
                            flag = False
                            log(DEBUG, "Worker %s closed", worker.worker_id)
                            break
            # Put the client ids in the queue
            for cid in list_ids_for_this_gpu:
                self.task_queues[device].put(cid)

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
                stats["gpu"].append(current_stats[3])
            num_processed_virtual_clients += 1
        start_time = time.time()
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
        # Node aggregation
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
        log(
            DEBUG,
            "NodeManager %s: elaborate %s clients in %s seconds",
            self.name,
            len(clients_training_stats["cid"]),
            time.time() - start_time,
        )
        return (
            node_trained_params,
            int(node_n_samples),
            {
                "train_loss": node_train_loss,
                "accuracy": node_accuracy,
                "stats": clients_training_buf.to_pybytes(),
            },
        )

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        return 0.0, 1, {}

    def __del__(self) -> None:
        """Implement the closing on the NodeManager."""
        log(DEBUG, "Closing NodeManager...")
        device: str = ""
        if self.workers is not None:
            for device, list_of_workers in self.workers.items():
                for _ in range(len(list_of_workers)):
                    # Put a None for a worker to terminate it
                    self.task_queues[device].put(None)
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
        for _i, v in enumerate(self.shared_local_agg.values()):
            v[4].close()
            v[4].unlink()
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
    )

    # Start Flower client
    fl.client.start_client(
        server_address=cfg.flwr_address,
        client=node_manager.to_client(),
    )


if __name__ == "__main__":
    main()
