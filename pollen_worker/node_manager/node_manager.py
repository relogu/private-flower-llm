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
import copy
import gc
import pickle
import time
import uuid
from logging import DEBUG, INFO
from multiprocessing.queues import Queue as QueueType
from socket import getfqdn
from typing import Any, Callable, Dict, List, Tuple, cast

import cloudpickle
import flwr as fl
import hydra
import nvsmi
import psutil
import torch
import transformers
from composer.utils.misc import get_free_tcp_port
from flwr.common import Config, NDArrays, Scalar
from flwr.common.logger import log
from flwr.server.strategy.aggregate import weighted_loss_avg
from multiprocess import Queue, set_start_method  # type: ignore
from nvsmi import GPU
from omegaconf import DictConfig, OmegaConf

from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from pollen_worker.clients.virtual_llm_client import VirtualLLMClient, gen_client_fn
from pollen_worker.node_manager.utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_EVAL_LOSS_SHM,
    POLLEN_METRICS_SHM,
    POLLEN_N_SAMPLES_SHM,
    POLLEN_PARAMETERS_SHM,
    close_all_shms,
    get_config_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    get_parameters_shm,
    set_parameters_shm,
)
from pollen_worker.node_manager.worker import Worker, create_new_worker, start_worker
from pollen_worker.resources_manager import Device, Node, get_gpu_prop
from pollen_worker.utils import (
    get_n_cuda_devices,
    partially_aggregate,
    weighted_average,
)

transformers.logging.set_verbosity_error()
set_start_method("spawn", force=True)
pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]


class NodeManager(fl.client.NumPyClient):
    """NodeManager of Pollen."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualLLMClient],
        run_uuid: str,
        parameters: NDArrays,
    ) -> None:
        super().__init__()
        ## NodeManager general attributes
        self.name: str = getfqdn()
        self.properties: Dict[str, Scalar] = {}
        self.all_gpus: List[GPU] = list(nvsmi.get_gpus())
        self.run_uuid = run_uuid
        self.node_manager_uuid = run_uuid + str(uuid.uuid4())
        self.client_fn = client_fn
        ## Set up Queues
        self.task_queue: QueueType = Queue()
        # One result_queue for all GPUs
        self.result_queue: QueueType = Queue()
        # Get node properties about hardware accelerators
        self.properties = self._get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        max_proc_device = [(k, v.concurrency) for k, v in self.node.device_info.items()]
        log(DEBUG, "Max processes per device: %s", max_proc_device)
        ## Set up round parameters SharedMemory
        # Shared memory for round parameters
        self.round_parameters, self.round_parameters_sh = get_parameters_shm(
            parameters=parameters,
            create=True,
            name=self.node_manager_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
        )

    def _get_node_properties(self) -> Dict[str, Scalar]:
        device_info: Dict[str, Device] = {}
        # Get hardware accelerator properties
        if torch.cuda.is_available():
            device_info = dict(
                get_gpu_prop(merge=True),
                **device_info,
            )
        try:
            cpus = len(psutil.Process().cpu_affinity())  # type: ignore
        except AttributeError:
            cpus = psutil.cpu_count()
        log(DEBUG, "NodeManager %s: device_info are %s", self.name, device_info)
        # Get general node properties
        self.node = Node(
            name=getfqdn(),
            cpu_num=cpus,
            cpu_ram_total=psutil.virtual_memory().total,
            cpu_ram_available=psutil.virtual_memory().total
            - psutil.virtual_memory().used,
            device_info=device_info,
        )
        log(DEBUG, "NodeManager %s: node properties are %s", self.name, self.node)

        return {"node": str(self.node)}

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Implement how to get properties."""
        return self.properties

    def get_parameters(self, config: Config) -> NDArrays:
        """Implement how to get parameters."""
        return self.round_parameters

    def _launch_worker_task(self, current_cid: int, action: str) -> Dict[int, Worker]:
        """Launch tasks and return workers dict."""
        # Set the task
        current_task = (current_cid, action)
        # Get the port
        current_port = str(get_free_tcp_port())
        # Create a new worker
        workers_dict: Dict[int, Worker] = {}
        for i in range(get_n_cuda_devices()):
            worker = create_new_worker(
                client_fn=self.client_fn,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                node_manager_uuid=self.node_manager_uuid,
                parameters=self.round_parameters,
                n_workers=get_n_cuda_devices(),
                worker_rank=i,
                port=current_port,
            )
            workers_dict[i] = worker
        log(
            DEBUG,
            "NodeManager %s: the worker dict has been build %s.",
            self.name,
            workers_dict,
        )
        # Start the workers
        for _, worker in workers_dict.items():
            start_worker(worker)
        # Send the task to the workers
        for _ in range(get_n_cuda_devices()):
            self.task_queue.put(current_task)
        log(
            DEBUG,
            "NodeManager %s: set task %s to workers.",
            self.name,
            current_task,
        )
        return workers_dict

    def _close_workers(self, workers_dict: Dict[int, Worker]) -> None:
        """Delete workers and close shared memories."""
        # Close and unlink the shared memory
        close_all_shms(workers_dict[0].worker_uuid)
        log(
            DEBUG,
            "NodeManager %s: worker %s shared memories have been closed.",
            self.name,
            workers_dict[0].worker_uuid,
        )
        # for _, worker in workers_dict.items():
        #     worker.terminate()
        # log(
        #     DEBUG,
        #     "NodeManager %s: sending termination command to workers.",
        #     self.name,
        # )
        # Wait until the worker is dead
        for _, worker in workers_dict.items():
            while worker.is_alive():
                time.sleep(0.1)
        log(
            DEBUG,
            "NodeManager %s: workers are dead.",
            self.name,
        )
        del workers_dict
        gc.collect()
        torch.cuda.empty_cache()

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, Dict[str, Scalar]]:
        """Implement the fit step."""
        log(DEBUG, "NodeManager %s: fit with config %s", self.name, config)
        start_time = time.time()
        # Extract assignments from config
        assignments = config.pop("merged", "0,1")
        list_of_cids_to_train = cast(str, assignments).split(",")
        # list_of_cids_to_train = [0, 1, 2, 3, 4, 5]
        # Update shared memories objects
        self.fl_instructions_config, self.fl_instructions_config_sh = get_config_shm(
            config=config,
            create=True,
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        set_parameters_shm(self.round_parameters, parameters)
        # Loop over virtual clients' results
        num_processed_virtual_clients = 0
        partially_aggregated_params: Tuple[NDArrays, int] = ([], 0)
        clients_train_metrics: List[Tuple[int, Dict]] = []
        while len(list_of_cids_to_train) > 0:
            # Get the current cid
            current_cid = int(list_of_cids_to_train.pop(0))
            workers_dict = self._launch_worker_task(current_cid, "fit")
            # Get the result
            current_stats = self.result_queue.get()
            log(
                DEBUG,
                "NodeManager %s: worker %s finished and returned %s.",
                self.name,
                workers_dict[0].worker_uuid,
                current_stats,
            )
            # Check if the training was successful
            if current_stats[0] > -1:
                # TODO: Collect stats
                # Get stuff from shared memories
                # NOTE: Keep the `*_shm` variables to prevent Seg Fault
                w_parameters, w_parameters_shm = get_parameters_shm(
                    parameters=self.round_parameters,
                    name=workers_dict[0].worker_uuid
                    + POLLEN_PARAMETERS_SHM,  # noqa: F821
                )
                w_num_samples, w_num_samples_shm = get_num_samples_shm(
                    name=workers_dict[0].worker_uuid
                    + POLLEN_N_SAMPLES_SHM,  # noqa: F821
                )
                w_metrics, w_metrics_shm = get_config_shm(
                    name=workers_dict[0].worker_uuid + POLLEN_METRICS_SHM
                )
                # Node's partial aggregation for parameters
                partially_aggregated_params = partially_aggregate(
                    partially_aggregated_params,
                    (copy.deepcopy(w_parameters), copy.deepcopy(w_num_samples[0])),
                )
                # Append train metrics to aggregate later
                clients_train_metrics.append((w_num_samples[0], w_metrics))
                num_processed_virtual_clients += 1
                log(
                    DEBUG,
                    "NodeManager %s: processed results.",
                    self.name,
                )
            else:
                list_of_cids_to_train.append(str(current_cid))
            self._close_workers(workers_dict)
        # Aggregation of train metrics
        node_train_metrics = weighted_average(clients_train_metrics)
        log(
            DEBUG,
            "NodeManager %s: time spent before collecting results is %s seconds.",
            self.name,
            time.time() - start_time,
        )
        log(
            DEBUG,
            "NodeManager %s: Results (%s, %s, %s).",
            self.name,
            len(partially_aggregated_params[0]),
            partially_aggregated_params[1],
            node_train_metrics,
        )
        # Close the config shared memory
        self.fl_instructions_config_sh.close()
        self.fl_instructions_config_sh.unlink()
        # Return results
        return (
            partially_aggregated_params[0],
            int(partially_aggregated_params[1]),
            node_train_metrics,
        )

    def evaluate(self, parameters, config) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        start_time = time.time()
        # Extract assignments from config
        assignments = config.pop("merged", "0,1")
        list_of_cids_to_eval = cast(str, assignments).split(",")
        # list_of_cids_to_eval = [0, 1, 2, 3, 4, 5]
        # Update shared memories objects
        self.fl_instructions_config, self.fl_instructions_config_sh = get_config_shm(
            config=config,
            create=True,
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        set_parameters_shm(self.round_parameters, parameters)
        # Loop over virtual clients' results
        num_processed_virtual_clients = 0
        clients_eval_losses: List[Tuple[int, float]] = []
        clients_eval_metrics: List[Tuple[int, Dict]] = []
        clients_eval_samples: List[int] = []
        while len(list_of_cids_to_eval) > 0:
            # Get the current cid
            current_cid = int(list_of_cids_to_eval.pop(0))
            workers_dict = self._launch_worker_task(current_cid, "evaluate")
            # Get the result
            current_stats = self.result_queue.get()
            log(
                DEBUG,
                "NodeManager %s: worker %s finished and returned %s.",
                self.name,
                workers_dict[0].worker_uuid,
                current_stats,
            )
            # Check if the evaluation was successful
            if current_stats[0] > -1:
                # TODO: Collect stats
                # Get stuff from shared memories
                # NOTE: Keep the `*_shm` variables to prevent Seg Fault
                w_eval_loss, w_eval_loss_shm = get_eval_loss_shm(
                    name=workers_dict[0].worker_uuid
                    + POLLEN_EVAL_LOSS_SHM,  # noqa: F821
                )
                w_num_samples, w_num_samples_shm = get_num_samples_shm(
                    name=workers_dict[0].worker_uuid
                    + POLLEN_N_SAMPLES_SHM,  # noqa: F821
                )
                w_metrics, w_metrics_shm = get_config_shm(
                    name=workers_dict[0].worker_uuid + POLLEN_METRICS_SHM
                )
                # Append eval losses to aggregate later
                clients_eval_losses.append((w_num_samples[0], w_eval_loss[0]))
                # Append eval metrics to aggregate later
                clients_eval_metrics.append((w_num_samples[0], w_metrics))
                # Append eval samples to aggregate later
                clients_eval_samples.append(w_num_samples[0])
                num_processed_virtual_clients += 1
                log(
                    DEBUG,
                    "NodeManager %s: processed results.",
                    self.name,
                )
            else:
                list_of_cids_to_eval.append(str(current_cid))
            self._close_workers(workers_dict)
        # Aggregation of eval losses
        node_eval_loss = weighted_loss_avg(clients_eval_losses)
        # Aggregation of eval metrics
        node_eval_metrics = weighted_average(clients_eval_metrics)
        # Aggregation of eval samples
        node_eval_samples = sum(clients_eval_samples)
        log(
            DEBUG,
            "NodeManager %s: time spent before collecting results is %s seconds.",
            self.name,
            time.time() - start_time,
        )
        # Close the config shared memory
        self.fl_instructions_config_sh.close()
        self.fl_instructions_config_sh.unlink()
        # Return results
        return (
            node_eval_loss,
            int(node_eval_samples),
            node_eval_metrics,
        )

    def __del__(self) -> None:
        """Implement the closing on the NodeManager."""
        log(DEBUG, "Closing NodeManager...")
        # TODO: Closing workers?
        # Free shared memories
        close_all_shms(self.node_manager_uuid)
        log(DEBUG, "Shared memories closed")


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Start a node manager directly with hydra."""
    start_time = time.time()
    log(
        INFO,
        "NodeManager received the following config:\n%s",
        OmegaConf.to_yaml(cfg, resolve=True),
    )
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    log(
        INFO,
        "NodeManager received the llm_config:\n%s",
        OmegaConf.to_yaml(_llm_config, resolve=True),
    )
    assert isinstance(_llm_config, DictConfig)
    # Get the client generator function
    client_fn = gen_client_fn(
        cfg=copy.deepcopy(_llm_config),
    )
    # Get initial model parameters
    parameters = get_raw_model_parameters(copy.deepcopy(_llm_config))
    # Create the NodeManager object
    node_manager = NodeManager(
        client_fn=client_fn,
        run_uuid=cfg.run_uuid,
        parameters=parameters,
    )
    # Choose the type of execution
    if cfg.is_test:
        log(INFO, "NodeManager::test")
        fl_instructions_config: Config = {"server_round": 1, "merged": "0"}
        # loss, n_samples, train_metrics = node_manager.evaluate(
        #     parameters, fl_instructions_config
        # )
        # log(
        #     INFO,
        #     "NodeManager::test::evaluate : loss=%s",
        #     loss,
        # )
        # log(
        #     INFO,
        #     "NodeManager::test::evaluate : n_samples=%s",
        #     n_samples,
        # )
        # log(
        #     INFO,
        #     "NodeManager::test::evaluate : train_metrics=%s",
        #     train_metrics,
        # )
        parameters, n_samples, train_metrics = node_manager.fit(
            parameters, fl_instructions_config
        )
        log(
            INFO,
            "NodeManager::test::fit : len(parameters)=%s",
            len(parameters),
        )
        log(
            INFO,
            "NodeManager::test::fit : n_samples=%s",
            n_samples,
        )
        log(
            INFO,
            "NodeManager::test::fit : train_metrics=%s",
            train_metrics,
        )
        properties = node_manager.get_properties(fl_instructions_config)
        log(
            INFO,
            "NodeManager::test::get_properties : properties=%s",
            properties,
        )
        parameters = node_manager.get_parameters(fl_instructions_config)
        log(
            INFO,
            "NodeManager::test::get_parameters : len(parameters)=%s",
            len(parameters),
        )
    else:
        # Start NodeManager as a Flower client
        fl.client.start_numpy_client(
            server_address=cfg.pollen.server_address,
            client=node_manager,
            grpc_max_message_length=int(1_000_000_000),
        )
    log(
        INFO,
        "NodeManager::Total time spent is %s seconds.",
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
