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
from logging import DEBUG, ERROR, INFO
from multiprocessing.queues import Queue as QueueType
from socket import getfqdn
from typing import Any, Callable, cast

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
    aggregate_training_results,
    close_all_shms,
    get_config_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    get_parameters_shm,
    partially_aggregate_training_results,
    set_config_shm,
    set_num_samples_shm,
    set_parameters_shm,
)
from pollen_worker.node_manager.worker import (
    Worker,
    create_new_worker,
    get_training_results_from_worker,
    get_training_results_from_workers_dict,
    start_worker,
)
from pollen_worker.resources_manager import Device, Node, get_gpu_prop
from pollen_worker.utils import (
    POLLEN_LLM_MAX_MESSAGE_LENGTH,
    get_n_cuda_devices,
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
        refresh_period: int,
    ) -> None:
        super().__init__()
        ## NodeManager general attributes
        self.name: str = getfqdn()
        self.properties: dict[str, Scalar] = {}
        self.all_gpus: list[GPU] = list(nvsmi.get_gpus())
        self.run_uuid = run_uuid
        self.node_manager_uuid = run_uuid + "-" + str(uuid.uuid4())
        self.client_fn = client_fn
        self.refresh_period = refresh_period
        ## Set up Queues
        self.task_queue: QueueType = Queue()
        # One result_queue for all GPUs
        self.result_queue: QueueType = Queue()
        # Get node properties about hardware accelerators
        self.properties = self._get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        [(k, v.concurrency) for k, v in self.node.device_info.items()]
        # log(DEBUG, "Max processes per device: %s", max_proc_device)
        ## Set up round parameters SharedMemory
        # Shared memory for round parameters
        self.round_parameters, self.round_parameters_sh = get_parameters_shm(
            parameters=parameters,
            create=True,
            name=self.node_manager_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
        )
        # Create workers
        self.workers_dict: dict[int, Worker] = {}
        self._create_and_start_workers()

    def _get_node_properties(self) -> dict[str, Scalar]:
        device_info: dict[str, Device] = {}
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
        # log(DEBUG, "NodeManager %s: device_info are %s", self.name, device_info)
        # Get general node properties
        self.node = Node(
            name=getfqdn(),
            cpu_num=cpus,
            cpu_ram_total=psutil.virtual_memory().total,
            cpu_ram_available=psutil.virtual_memory().total
            - psutil.virtual_memory().used,
            device_info=device_info,
        )
        # log(DEBUG, "NodeManager %s: node properties are %s", self.name, self.node)
        return {"node": str(self.node)}

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return self.properties

    def get_parameters(self, config: Config) -> NDArrays:
        """Implement how to get parameters."""
        return self.round_parameters

    def _check_workers_health(self) -> None:
        """Check if workers are alive and restart them if not."""
        for rank, worker in self.workers_dict.items():
            if not worker.is_alive():
                log(
                    DEBUG,
                    "NodeManager %s: worker %s is dead. Restarting it...",
                    self.name,
                    rank,
                )
                close_all_shms(worker.worker_uuid)
                self.workers_dict[rank] = create_new_worker(
                    client_fn=self.client_fn,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    node_manager_uuid=self.node_manager_uuid,
                    run_uuid=self.run_uuid,
                    parameters=self.round_parameters,
                    worker_rank=rank,
                )
                start_worker(self.workers_dict[rank])

    def _create_and_start_workers(self) -> None:
        """Create and start workers."""
        for i in range(get_n_cuda_devices()):
            worker = create_new_worker(
                client_fn=self.client_fn,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                node_manager_uuid=self.node_manager_uuid,
                run_uuid=self.run_uuid,
                parameters=self.round_parameters,
                worker_rank=i,
            )
            self.workers_dict[i] = worker
            log(DEBUG, f"Created worker with rank {i}")
        # log(
        #     DEBUG,
        #     "NodeManager %s: the worker dict has been build %s.",
        #     self.name,
        #     self.workers_dict,
        # )
        # Start the workers
        for _, worker in self.workers_dict.items():
            start_worker(worker)
        log(DEBUG, "NodeManager %s: all workers started.", self.name)

    def _close_workers(self) -> None:
        """Delete workers and close shared memories."""
        # Close and unlink all the shared memories
        for _, worker in self.workers_dict.items():
            close_all_shms(worker.worker_uuid)
        # Wait until the worker is dead
        for _, worker in self.workers_dict.items():
            worker.soft_shutdown()
            while worker.is_alive():
                time.sleep(0.1)
                worker.terminate()
        log(
            DEBUG,
            "NodeManager %s: workers are dead.",
            self.name,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _independent_fit(
        self, config: Config, list_of_cids_to_train: list[str], parameters: NDArrays
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Append NodeManager's config
        config["MASTER_PORT"] = ""
        config["run_uuid"] = ""
        # Update shared memories objects
        fl_instructions_config, fl_instructions_config_sh = get_config_shm(
            config=config,
            create=True,
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        set_config_shm(config, fl_instructions_config_sh)
        set_parameters_shm(self.round_parameters, parameters)
        # Here, workers are forced to train independently
        # Send the independent tasks to the workers
        for cid in list_of_cids_to_train:
            self.task_queue.put((cid, "fit"))
        # Get the results
        successes = 0
        while successes < len(list_of_cids_to_train):
            self._check_workers_health()
            # TODO: Handle the case where they all fail
            current_stats = self.result_queue.get()
            log(
                DEBUG,
                "NodeManager %s: worker %s finished and returned cid %s.",
                self.name,
                current_stats[3],
                current_stats[0],
            )
            # Check if the training was successful
            if current_stats[0] > -1:
                # TODO: Collect stats
                successes += 1
        # Get stuff from shared memories of the workers
        # NOTE: Keep a reference to the `*_shm` variables to prevent Seg Fault
        w_p_s, w_s_m, w_s, w_shms = get_training_results_from_workers_dict(
            self.workers_dict
        )
        # Partially aggregate training results
        (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        ) = aggregate_training_results(
            w_p_s,
            [s[0] for s in w_s],
            w_s_m,
        )
        # Zero out the n_samples shared memories
        for ww_ss in w_s:
            set_num_samples_shm(ww_ss, 0)
        # Close the config shared memory
        fl_instructions_config_sh.close()
        fl_instructions_config_sh.unlink()
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def _collaborative_fit(
        self, config: Config, list_of_cids_to_train: list[str], parameters: NDArrays
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Update paramters shared memory
        set_parameters_shm(self.round_parameters, parameters)
        # Initialise partial aggregation variables
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
        node_train_metrics: dict = {}
        # Here, workers are forced to collaborate with each other,
        # as such, we evaluate one client at a time
        while len(list_of_cids_to_train) > 0:
            self._check_workers_health()
            # Get the current cid
            current_cid = int(list_of_cids_to_train.pop(0))
            # Append NodeManager's config
            config["MASTER_PORT"] = str(get_free_tcp_port())
            # NOTE: Putting the node_manager_uuid in the config fails
            config["run_uuid"] = self.run_uuid
            # Update instruction config shared memory
            fl_instructions_config, fl_instructions_config_sh = get_config_shm(
                config=config,
                create=True,
                name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
            )
            set_config_shm(config, fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "fit"))
            # Wait for the result
            current_stats = None
            while current_stats is None:
                try:
                    current_stats = self.result_queue.get(timeout=10)
                except Exception as e:
                    # log(
                    #     ERROR,
                    #     "NodeManager %s: no results received in time.",
                    #     self.name,
                    #     exc_info=e,
                    #     stack_info=True,
                    # )
                    for _, worker in self.workers_dict.items():
                        if not worker.is_alive():
                            current_stats = [-1, 0, 0, -1]
            log(
                DEBUG,
                "NodeManager %s: worker %s finished and returned cid %s.",
                self.name,
                current_stats[3],
                current_stats[0],
            )
            # Check if the training was successful
            if current_stats[0] > -1:
                # TODO: Collect stats
                # Get stuff from shared memories of the workers
                # NOTE: Keep a reference to the `*_shm` variables to prevent Seg Fault
                results = get_training_results_from_worker(self.workers_dict[0])
                if results is not None:
                    w_p_s, w_s_m, w_s, w_shms = results
                    # Partial aggregation of training results
                    (
                        aggregated_params,
                        sum_of_samples,
                        node_train_metrics,
                    ) = partially_aggregate_training_results(
                        (aggregated_params, sum_of_samples, node_train_metrics),
                        (w_p_s[0], w_s[0], w_s_m[1]),
                    )
                    # Zero out the n_samples shared memories
                    set_num_samples_shm(w_s, 0)
                else:
                    log(ERROR, "Results received are invalid!")
                    list_of_cids_to_train.append(str(current_cid))
            else:
                # If the training was not successful, put the cid back in the list
                list_of_cids_to_train.append(str(current_cid))
            # Close the config shared memory
            fl_instructions_config_sh.close()
            fl_instructions_config_sh.unlink()
            # Empty the tasks list
            while not self.task_queue.empty():
                self.task_queue.get()
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Implement the fit step."""
        # log(DEBUG, "NodeManager %s: fit with config %s", self.name, config)
        start_time = time.time()
        # Restart all the worker every `self.refresh_period` rounds
        if config["server_round"] % self.refresh_period == 0:
            # Close and remove the workers
            self._close_workers()
            # Re-create and start the workers
            self._create_and_start_workers()
        # Extract assignments from config
        assignments = config.pop("merged", "0,1")
        list_of_cids_to_train = cast(str, assignments).split(",")
        try:
            # Choose the type of execution
            if config["collaborative"]:
                (
                    aggregated_params,
                    sum_of_samples,
                    node_train_metrics,
                ) = self._collaborative_fit(config, list_of_cids_to_train, parameters)
            else:
                (
                    aggregated_params,
                    sum_of_samples,
                    node_train_metrics,
                ) = self._independent_fit(config, list_of_cids_to_train, parameters)
        except Exception as e:
            log(ERROR, "NodeManager %s", self.name, exc_info=e, stack_info=True)
        # Adding node training time in the metrics
        node_train_metrics.update(
            {"node_training_time_s": float(time.time() - start_time)}
        )
        log(
            DEBUG,
            "NodeManager %s: resuls have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.name,
            time.time() - start_time,
        )
        log(
            DEBUG,
            "NodeManager %s: Results (%s, %s, %s).",
            self.name,
            len(aggregated_params),
            sum_of_samples,
            node_train_metrics,
        )
        # Return results
        return (
            aggregated_params,
            int(sum_of_samples),
            node_train_metrics,
        )

    def evaluate(self, parameters, config) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        start_time = time.time()
        # Extract assignments from config
        assignments = config.pop("merged", "0,1")
        list_of_cids_to_eval = cast(str, assignments).split(",")
        # Append NodeManager's config
        config["run_uuid"] = (
            self.run_uuid if config["collaborative"] else self.node_manager_uuid
        )
        set_parameters_shm(self.round_parameters, parameters)
        # Loop over virtual clients' results
        num_processed_virtual_clients = 0
        clients_eval_losses: list[tuple[int, float]] = []
        clients_eval_metrics: list[tuple[int, dict[str, Scalar]]] = []
        clients_eval_samples: list[int] = []
        # Here, workers are forced to collaborate with each other,
        # as such, we evaluate one client at a time
        while len(list_of_cids_to_eval) > 0:
            self._check_workers_health()
            # Get the current cid
            current_cid = int(list_of_cids_to_eval.pop(0))
            config["MASTER_PORT"] = str(get_free_tcp_port())
            # Update shared memories objects
            (
                self.fl_instructions_config,
                self.fl_instructions_config_sh,
            ) = get_config_shm(
                config=config,
                create=True,
                name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
            )
            set_config_shm(config, self.fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "evaluate"))
            # Wait for the result
            current_stats = None
            while current_stats is None:
                try:
                    current_stats = self.result_queue.get(timeout=10)
                except Exception as e:
                    # log(
                    #     ERROR,
                    #     "NodeManager %s: no results received in time.",
                    #     self.name,
                    #     exc_info=e,
                    #     stack_info=True,
                    # )
                    for _, worker in self.workers_dict.items():
                        if not worker.is_alive():
                            current_stats = [-1, 0, 0, -1]
            log(
                DEBUG,
                "NodeManager %s: worker %s finished and returned cid %s.",
                self.name,
                current_stats[3],
                current_stats[0],
            )
            # Check if the evaluation was successful
            if current_stats[0] > -1:
                # TODO: Collect stats
                # Get stuff from shared memories of the rank 0 worker
                # NOTE: Keep the `*_shm` variables to prevent Seg Fault
                w_eval_loss, w_eval_loss_shm = get_eval_loss_shm(
                    name=self.workers_dict[0].worker_uuid
                    + POLLEN_EVAL_LOSS_SHM,  # noqa: F821
                )
                w_num_samples, w_num_samples_shm = get_num_samples_shm(
                    name=self.workers_dict[0].worker_uuid
                    + POLLEN_N_SAMPLES_SHM,  # noqa: F821
                )
                w_metrics, w_metrics_shm = get_config_shm(
                    config={},
                    name=self.workers_dict[0].worker_uuid + POLLEN_METRICS_SHM,
                )
                # Append eval losses to aggregate later
                clients_eval_losses.append((int(w_num_samples[0]), w_eval_loss[0]))
                # Append eval metrics to aggregate later
                clients_eval_metrics.append((int(w_num_samples[0]), w_metrics))
                # Append eval samples to aggregate later
                clients_eval_samples.append(int(w_num_samples[0]))
                num_processed_virtual_clients += 1
                # Zero out the n_samples shared memory
                set_num_samples_shm(w_num_samples, 0)
            else:
                list_of_cids_to_eval.append(str(current_cid))
                # Kill all the workers and restart
                self._close_workers()
            # Close the config shared memory
            self.fl_instructions_config_sh.close()
            self.fl_instructions_config_sh.unlink()
            # Empty the tasks list
            while not self.task_queue.empty():
                self.task_queue.get()
        # Aggregation of eval losses
        node_eval_loss = weighted_loss_avg(clients_eval_losses)
        # Aggregation of eval metrics
        node_eval_metrics = weighted_average(clients_eval_metrics)
        node_eval_metrics.update({"node_eval_time_s": float(time.time() - start_time)})
        # Aggregation of eval samples
        node_eval_samples = sum(clients_eval_samples)
        log(
            DEBUG,
            "NodeManager %s: resuls have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.name,
            time.time() - start_time,
        )
        log(
            DEBUG,
            "NodeManager %s: Results (%s, %s, %s).",
            self.name,
            node_eval_loss,
            int(node_eval_samples),
            node_eval_metrics,
        )
        # Return results
        return (
            node_eval_loss,
            int(node_eval_samples),
            node_eval_metrics,
        )

    def __del__(self) -> None:
        """Implement the closing on the NodeManager."""
        log(DEBUG, "Closing NodeManager...")
        # Closing workers
        self._close_workers()
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
        refresh_period=int(cfg.pollen.refresh_period),
    )
    # Choose the type of execution
    if cfg.is_test:
        log(INFO, "NodeManager::test")
        fl_instructions_config: Config = {"server_round": 1, "merged": "0,1,2"}
        loss, n_samples, train_metrics = node_manager.evaluate(
            parameters, fl_instructions_config
        )
        log(
            INFO,
            "NodeManager::test::evaluate : loss=%s",
            loss,
        )
        log(
            INFO,
            "NodeManager::test::evaluate : n_samples=%s",
            n_samples,
        )
        log(
            INFO,
            "NodeManager::test::evaluate : train_metrics=%s",
            train_metrics,
        )
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
            client=node_manager.to_client(),
            grpc_max_message_length=POLLEN_LLM_MAX_MESSAGE_LENGTH,
        )
    log(
        INFO,
        "NodeManager::Total time spent is %s seconds.",
        time.time() - start_time,
    )


if __name__ == "__main__":
    main()
