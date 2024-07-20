"""Flower NodeManagerApp."""

import ast
from collections import defaultdict
import copy
import gc
from logging import DEBUG, ERROR
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
from tempfile import TemporaryDirectory
import time
from typing import Any, cast
import uuid
import cloudpickle
from multiprocessing.queues import Queue as QueueType
from flwr.client import ClientApp
from flwr.client.typing import ClientFnExt, Mod
from flwr.common.typing import NDArrays, ConfigsRecordValues, Scalar
from flwr.common.recordset_compat import ConfigsRecord
from flwr.common.record.typeddict import TypedDict
from flwr.common.logger import log
from multiprocess import Queue, set_start_method  # type: ignore[reportAttributeAccessIssue]
import numpy as np
from omegaconf import DictConfig, OmegaConf
from composer.loggers import RemoteUploaderDownloader
from composer.utils.misc import get_free_tcp_port
import torch

from flower_llm.clients.llm_client_functions import get_raw_model_parameters
from flower_llm.conf.base_schema import BaseConfig
from flower_llm.node_manager.utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_EVAL_LOSS_SHM,
    POLLEN_METRICS_SHM,
    POLLEN_N_SAMPLES_SHM,
    ModelParametersMetadata,
    WorkerResult,
    aggregate_training_results,
    close_all_shms,
    get_config_shm,
    get_dict_configsrecord_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    partially_aggregate_training_results,
    remove_shm_from_resource_tracker,
    set_dict_configsrecord_shm,
    set_num_samples_shm,
)
from flower_llm.node_manager.worker import (
    Worker,
    create_new_worker,
    get_training_results_from_worker,
    get_training_results_from_workers_dict,
    start_worker,
)
from flower_llm.resources_manager import get_node_properties
from flower_llm.utils import get_n_cuda_devices, sum_of_squares
from flwr.server.strategy.aggregate import weighted_loss_avg
from flower_llm.strategy.aggregation import weighted_average

set_start_method("spawn", force=True)
pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]


class NodeManagerApp(ClientApp):
    """Flower NodeManagerApp."""

    def __init__(
        self,
        client_fn: ClientFnExt | None = None,  # Only for backward compatibility
        mods: list[Mod] | None = None,
    ) -> None:
        super().__init__(client_fn=client_fn, mods=mods)
        # Get the environmental variable for the dump folder
        save_path = os.environ.get("POLLEN_SAVE_PATH", "")
        # Raise an error if the environmental variable is not set
        if not save_path:
            raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
        # Load the configuration from the config file
        self.cfg = cast(BaseConfig, OmegaConf.load(save_path + "/config.yaml"))
        # Resolve the config and set it to be editable in place
        OmegaConf.resolve(self.cfg)
        OmegaConf.set_struct(self.cfg, False)
        # Set up Queues
        self.task_queue: QueueType = Queue()
        # One result_queue for all GPUs
        self.result_queue: QueueType = Queue()
        # Get node properties about hardware accelerators
        self.node = get_node_properties(
            self.cfg.pollen.cpu_only, self.cfg.pollen.cpu_concurrency
        )
        self.properties: dict[str, ConfigsRecordValues] = {"node": str(self.node)}
        self.node_manager_uuid = self.cfg.run_uuid + "-" + str(uuid.uuid4())
        assert self.node.device_info is not None
        self.refresh_period = self.cfg.pollen.refresh_period
        self.node_manager_temp_dir = TemporaryDirectory()
        self.remote_up_down: RemoteUploaderDownloader | None = None
        self._create_remote_up_down()
        # Call the monkey-patch for the resource-register
        remove_shm_from_resource_tracker()
        # Extract the LLM part of the config
        _llm_config = self.cfg.llm_config
        assert isinstance(_llm_config, DictConfig)
        # Get initial model parameters
        parameters = cast(
            NDArrays, get_raw_model_parameters(copy.deepcopy(_llm_config))
        )
        parameters_metadata = ModelParametersMetadata.from_ndarrays(parameters)
        del parameters
        # Parameter metadata
        self.parameters_metadata = parameters_metadata
        # Shared memory for round parameters
        self.round_parameters: NDArrays | None = None
        self.round_parameters_sh: SharedMemory | None = None
        # Create workers
        self.workers_dict: dict[int, Worker] = {}
        self._create_and_start_workers()

    def _create_and_start_workers(self) -> None:
        """Create and start workers."""
        for i in range(
            get_n_cuda_devices()
            if not self.cfg.pollen.cpu_only
            else self.cfg.pollen.cpu_concurrency
        ):
            worker = create_new_worker(
                config=self.cfg,
                task_queue=self.task_queue,
                result_queue=self.result_queue,
                node_manager_uuid=self.node_manager_uuid,
                run_uuid=self.cfg.run_uuid,
                parameters_metadata=self.parameters_metadata,
                worker_rank=i,
                cpu_only=self.cfg.pollen.cpu_only,
                cpu_concurrency=self.cfg.pollen.cpu_concurrency,
            )
            self.workers_dict[i] = worker
            log(
                DEBUG,
                "Created %s worker with rank %s",
                "cpu" if self.cfg.pollen.cpu_only else "gpu",
                i,
            )
        # log(
        #     DEBUG,
        #     "NodeManagerApp %s: the worker dict has been build %s.",
        #     self.name,
        #     self.workers_dict,
        # )
        # Start the workers
        for worker in self.workers_dict.values():
            start_worker(worker)
        log(DEBUG, "NodeManagerApp %s: all workers started.", self.node.name)

    def _check_workers_health(self) -> None:
        """Check if workers are alive and restart them if not."""
        for rank, worker in self.workers_dict.items():
            if not worker.is_alive():
                log(
                    DEBUG,
                    "NodeManagerApp %s: worker %s is dead. Restarting it...",
                    self.node.name,
                    rank,
                )
                close_all_shms(worker.worker_uuid)
                self.workers_dict[rank] = create_new_worker(
                    config=self.cfg,
                    task_queue=self.task_queue,
                    result_queue=self.result_queue,
                    node_manager_uuid=self.node_manager_uuid,
                    run_uuid=self.cfg.run_uuid,
                    parameters_metadata=self.parameters_metadata,
                    worker_rank=rank,
                    cpu_only=self.cfg.pollen.cpu_only,
                    cpu_concurrency=self.cfg.pollen.cpu_concurrency,
                )
                start_worker(self.workers_dict[rank])

    def _close_workers(self) -> None:
        """Delete workers and close shared memories."""
        # Wait until the worker is dead
        for worker in self.workers_dict.values():
            worker.soft_shutdown()
            while worker.is_alive():
                time.sleep(0.1)
                worker.terminate()
        log(
            DEBUG,
            "NodeManagerApp %s: workers are dead.",
            self.node.name,
        )
        gc.collect()
        torch.cuda.empty_cache()

    def _create_remote_up_down(self) -> None:
        """Create the remote uploader/downloader."""
        if self.cfg.use_s3_comm:
            bucket_uri = f"s3://{self.cfg.s3_comm_config.bucket_name}"
            self.remote_up_down = RemoteUploaderDownloader(
                bucket_uri=bucket_uri,
                backend_kwargs={
                    "bucket": self.cfg.s3_comm_config.bucket_name,
                    "prefix": f"{self.cfg.run_uuid}/server",  # Don't touch
                    "region_name": None,  # Not necessary
                    "endpoint_url": None,  # Will be read from env var
                    "aws_access_key_id": None,  # Will be read from config file
                    "aws_secret_access_key": None,  # Will be read from config file
                    "aws_session_token": None,  # Will be automatically generated
                    "client_config": OmegaConf.to_container(
                        self.cfg.s3_comm_config.backend_kwargs.client_config
                    ),  # And using defaults
                    "transfer_config": None,  # Using defaults
                },
                file_path_format_string="{remote_file_name}",  # Don't touch
                num_concurrent_uploads=1,
                upload_staging_folder=None,  # Don't touch, it's /tmp by default
                use_procs=True,  # Don't touch
                num_attempts=self.cfg.s3_comm_config.num_attempts,
            )
            self.remote_up_down.init(run_name=self.cfg.run_uuid)

    def __del__(self) -> None:
        """Implement the closing on the NodeManagerApp."""
        log(DEBUG, "Closing NodeManagerApp...")
        # Closing workers
        self._close_workers()
        # Free shared memories
        close_all_shms(self.node_manager_uuid)
        log(DEBUG, "Shared memories closed")

    def fit(
        self,
        configs: TypedDict[str, ConfigsRecord],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        start_time = time.time()
        # # TODO: This is old based on Pollen that assigns clients to specific hardware
        # # resources. We will need to re-think how to do this.
        # # Extract assignments from config
        # assignments: str | None = None
        # assert self.node.device_info is not None
        # for key in self.node.device_info:
        #     assignments = cast(str, config.pop(key, str([[0, 1]])))
        # if not assignments:
        #     raise ValueError("No assignments found in the config.")
        # list_of_cids_to_train: list[str] = ast.literal_eval(assignments)[0]

        fit_ins_config = configs.pop("fitins.config")
        assignments = fit_ins_config.pop("client_ids")
        assert assignments is not None, "No assignments found in the config."
        list_of_cids_to_train: list[str] = ast.literal_eval(str(assignments))
        is_fit_collaborative = self.cfg.pollen.fit_collaborative
        # Force collaborative if number of clients < number of workers
        if len(list_of_cids_to_train) < len(self.workers_dict):
            log(
                DEBUG,
                "Forcing collaborative training since %s clients for %s workers.",
                len(list_of_cids_to_train),
                len(self.workers_dict),
            )
            is_fit_collaborative = True
        # Update each client config in `configs` with the shared parameters in
        # `fit_ins_config`
        log(
            DEBUG,
            "NodeManager %s: training %s clients using %s configs.",
            self.node_manager_uuid,
            list_of_cids_to_train,
            configs,
        )
        for cid in list_of_cids_to_train:
            configs[str(cid)].update(fit_ins_config)
            configs[str(cid)].update({"collaborative": is_fit_collaborative})

        node_train_metrics: dict[str, Scalar] = {}
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
        try:
            # Pass the collaborative flag to the config so that Workers know what
            # policy to adopt
            # Choose the type of execution
            if is_fit_collaborative:
                (
                    aggregated_params,
                    sum_of_samples,
                    node_train_metrics,
                ) = self._collaborative_fit(configs, list_of_cids_to_train)
            else:
                (
                    aggregated_params,
                    sum_of_samples,
                    node_train_metrics,
                ) = self._independent_fit(configs, list_of_cids_to_train)
        except Exception as e:
            log(
                ERROR,
                "NodeManager %s",
                self.node_manager_uuid,
                exc_info=e,
                stack_info=True,
            )
        # Adding node training time in the metrics
        node_train_metrics.update(
            {
                "node_training_time_s": float(time.time() - start_time),
            }
        )
        log(
            DEBUG,
            "NodeManager %s: results have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.node_manager_uuid,
            time.time() - start_time,
        )
        # Return results
        return (
            aggregated_params,
            int(sum_of_samples),
            node_train_metrics,
        )

    def _independent_fit(
        self, configs: TypedDict[str, ConfigsRecord], list_of_cids_to_train: list[str]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Append NodeManager's config
        for config in configs.values():
            config["MASTER_PORT"] = ""
            config["run_uuid"] = ""
        # Update shared memories objects
        _fl_instructions_config, fl_instructions_config_sh = get_dict_configsrecord_shm(
            config=configs,
            create=True,
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
        )
        set_dict_configsrecord_shm(configs, fl_instructions_config_sh)
        # Here, workers are forced to train independently
        # Send the independent tasks to the workers
        for cid in list_of_cids_to_train:
            self.task_queue.put((cid, "fit"))
        # Get the results
        successes = 0
        stats = defaultdict(list)
        while successes < len(list_of_cids_to_train):
            self._check_workers_health()
            # TODO: Handle the case where they all fail
            worker_result: WorkerResult = self.result_queue.get()
            # Check if the training was successful
            if worker_result.n_samples > -1:
                # NOTE: The order here is important for compatibility with the server
                stats["device"].append(worker_result.device)
                stats["n_samples"].append(worker_result.n_samples)  # type: ignore[arg-type]
                stats["delta"].append(worker_result.delta)  # type: ignore[arg-type]
                successes += 1
        # Get stuff from shared memories of the workers
        # NOTE: Keep a reference to the `*_shm` variables to prevent Seg Fault
        w_p_s, w_s_m, w_s, _w_shms = get_training_results_from_workers_dict(
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
        assert aggregated_params is not None
        # Close the parameters shared memories of the workers
        for shared_memories in _w_shms:
            parameters_shared_memory = shared_memories[0]
            parameters_shared_memory.close()
            parameters_shared_memory.unlink()
        # Zero out the n_samples shared memories
        for ww_ss in w_s:
            set_num_samples_shm(ww_ss, 0)
        # Close the config shared memory
        fl_instructions_config_sh.close()
        fl_instructions_config_sh.unlink()
        # Return the results
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def _collaborative_fit(
        self,
        configs: TypedDict[str, ConfigsRecord],
        list_of_cids_to_train: list[str],
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Initialise partial aggregation variables
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
        node_train_metrics: dict = {}
        stats = defaultdict(list)
        # Here, workers are forced to collaborate with each other,
        # as such, we evaluate one client at a time
        while len(list_of_cids_to_train) > 0:
            self._check_workers_health()
            # Get the current cid
            current_cid = int(list_of_cids_to_train.pop(0))
            # Extract config
            config = cast(TypedDict, {str(current_cid): configs[str(current_cid)]})
            # Append NodeManager's config
            config[str(current_cid)]["MASTER_PORT"] = str(get_free_tcp_port())
            # NOTE: Putting the node_manager_uuid in the config fails
            config[str(current_cid)]["run_uuid"] = self.cfg.run_uuid
            # Update instruction config shared memory
            _fl_instructions_config, fl_instructions_config_sh = (
                get_dict_configsrecord_shm(
                    config=config,
                    create=True,
                    name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
                )
            )
            set_dict_configsrecord_shm(config, fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "fit"))
            # Wait for the result
            worker_result: WorkerResult | None = None
            while worker_result is None:
                try:
                    worker_result = self.result_queue.get(timeout=10)
                except Exception:
                    # log(
                    #     ERROR,
                    #     "NodeManager %s: no results received in time.",
                    #     self.name,
                    #     exc_info=e,
                    #     stack_info=True,
                    # )
                    for worker in self.workers_dict.values():
                        if not worker.is_alive():
                            worker_result = WorkerResult(
                                -1,
                                0.0,
                                "",
                            )
            # Check if the training was successful
            if worker_result.n_samples > -1:
                # NOTE: The order here is important for compatibility with the server
                stats["device"].append(worker_result.device)
                stats["n_samples"].append(worker_result.n_samples)  # type: ignore[arg-type]
                stats["delta"].append(worker_result.delta)  # type: ignore[arg-type]
                # Get stuff from shared memories of the workers
                # NOTE: Keep a reference to the `*_shm` variables to prevent Seg Fault
                results = get_training_results_from_worker(self.workers_dict[0])
                if results is not None:
                    w_p_s, w_s_m, w_s, _w_shms = results
                    # Partial aggregation of training results
                    (
                        aggregated_params,
                        sum_of_samples,
                        node_train_metrics,
                    ) = partially_aggregate_training_results(
                        (aggregated_params, sum_of_samples, node_train_metrics),
                        (w_p_s[0], w_s[0], w_s_m[1]),
                    )
                    if sum_of_samples > 0:
                        node_clients_pairwise_delta = sum(
                            sum_of_squares([x - y])
                            for x, y in zip(aggregated_params, w_p_s[0], strict=False)
                        )
                        node_train_metrics |= {
                            "node_clients_pairwise_delta": float(
                                np.sqrt(node_clients_pairwise_delta)
                            )
                        }
                    # Close the parameters shared memories of the workers
                    parameters_shared_memory = _w_shms[0]
                    parameters_shared_memory.close()
                    parameters_shared_memory.unlink()
                    # Zero out the n_samples shared memories
                    set_num_samples_shm(w_s, 0)
                else:
                    log(ERROR, "Results received are invalid!")
                    list_of_cids_to_train.append(str(current_cid))
            else:
                # If the training was not successful, put the cid back in the list
                list_of_cids_to_train.append(str(current_cid))
                # Close all workers to refresh the state
                self._close_workers()
            # Close the config shared memory
            fl_instructions_config_sh.close()
            fl_instructions_config_sh.unlink()
            # Empty the tasks list
            while not self.task_queue.empty():
                self.task_queue.get()
        # Return the results
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def eval(
        self,
        configs: TypedDict[str, ConfigsRecord],
    ) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        start_time = time.time()

        # # Extract assignments from config
        # assignments: str | None = None
        # assert self.node.device_info is not None
        # for key in self.node.device_info:
        #     assignments = cast(str, config.pop(key, str([[0, 1]])))
        # if not assignments:
        #     raise ValueError("No assignments found in the config.")
        # list_of_cids_to_eval: list[str] = ast.literal_eval(assignments)[0]

        evaluate_ins_config = configs.pop("evaluateins.config")
        assignments = evaluate_ins_config.pop("client_ids")
        assert assignments is not None, "No assignments found in the config."
        list_of_cids_to_eval: list[str] = ast.literal_eval(str(assignments))
        is_evaluate_collaborative = self.cfg.pollen.fit_collaborative
        # Force collaborative if number of clients < number of workers
        if len(list_of_cids_to_eval) < len(self.workers_dict):
            log(
                DEBUG,
                "Forcing collaborative training since %s clients for %s workers.",
                len(list_of_cids_to_eval),
                len(self.workers_dict),
            )
            is_evaluate_collaborative = True
        # Update each client config in `configs` with the shared parameters in
        # `fit_ins_config`
        log(
            DEBUG,
            "NodeManager %s: evaluating %s clients using %s configs.",
            self.node_manager_uuid,
            list_of_cids_to_eval,
            configs,
        )
        for cid in list_of_cids_to_eval:
            configs[str(cid)].update(evaluate_ins_config)
            configs[str(cid)].update({"collaborative": is_evaluate_collaborative})
            configs[str(cid)].update(
                {
                    "run_uuid": (
                        self.cfg.run_uuid
                        if is_evaluate_collaborative
                        else self.node_manager_uuid
                    )
                }
            )
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
            # Extract config
            config = cast(TypedDict, {str(current_cid): configs[str(current_cid)]})
            # Append NodeManager's config
            config[str(current_cid)]["MASTER_PORT"] = str(get_free_tcp_port())
            # Update shared memories objects
            (
                self.fl_instructions_config,
                self.fl_instructions_config_sh,
            ) = get_dict_configsrecord_shm(
                config=config,
                create=True,
                name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
            )
            set_dict_configsrecord_shm(config, self.fl_instructions_config_sh)
            # Send the collaborative task to the workers
            for _ in range(len(self.workers_dict)):
                self.task_queue.put((current_cid, "evaluate"))
            # Wait for the result
            worker_results: WorkerResult | None = None
            while worker_results is None:
                try:
                    worker_results = self.result_queue.get(timeout=10)
                except Exception:
                    # log(
                    #     ERROR,
                    #     "NodeManager %s: no results received in time.",
                    #     self.name,
                    #     exc_info=e,
                    #     stack_info=True,
                    # )
                    for worker in self.workers_dict.values():
                        if not worker.is_alive():
                            worker_results = WorkerResult(
                                -1,
                                0.0,
                                "",
                            )
            # Check if the evaluation was successful
            if worker_results.n_samples > -1:
                # TODO: Collect stats
                # Get stuff from shared memories of the rank 0 worker
                # NOTE: Keep the `*_shm` variables to prevent Seg Fault
                w_eval_loss, _w_eval_loss_shm = get_eval_loss_shm(
                    name=self.workers_dict[0].worker_uuid + POLLEN_EVAL_LOSS_SHM,
                )
                w_num_samples, _w_num_samples_shm = get_num_samples_shm(
                    name=self.workers_dict[0].worker_uuid + POLLEN_N_SAMPLES_SHM,
                )
                w_metrics, _w_metrics_shm = get_config_shm(
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
            "NodeManager %s: results have been processed. "
            "The time spent before collecting results was %s seconds.",
            self.node_manager_uuid,
            time.time() - start_time,
        )
        # log(
        #     DEBUG,
        #     "NodeManager %s: Results (%s, %s, %s).",
        #     self.name,
        #     node_eval_loss,
        #     int(node_eval_samples),
        #     node_eval_metrics,
        # )
        # Return results
        return (
            node_eval_loss,
            int(node_eval_samples),
            node_eval_metrics,
        )
