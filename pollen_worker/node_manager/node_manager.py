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

from collections import defaultdict
import copy
import gc
from pathlib import Path
import pickle
import time
import uuid
from collections.abc import Callable
from logging import DEBUG, ERROR, INFO
from multiprocessing.queues import Queue as QueueType
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
from composer.utils.misc import get_free_tcp_port
from flwr.common import (
    Config,
    NDArrays,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.strategy.aggregate import weighted_loss_avg
from multiprocess import Queue, set_start_method  # type: ignore[reportAttributeAccessIssue]
from nvsmi import GPU
from omegaconf import DictConfig, OmegaConf
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path

from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from pollen_worker.clients.virtual_llm_client import VirtualLLMClient, gen_client_fn
from pollen_worker.node_manager.utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_EVAL_LOSS_SHM,
    POLLEN_METRICS_SHM,
    POLLEN_N_SAMPLES_SHM,
    POLLEN_PARAMETERS_SHM,
    WorkerResult,
    aggregate_training_results,
    close_all_shms,
    get_config_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    get_parameters_shm,
    partially_aggregate_training_results,
    remove_shm_from_resource_tracker,
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
from pollen_worker.placements import add_constant_column_to_clients_stats_table
from pollen_worker.resources_manager import Device, Node, get_gpu_prop
from pollen_worker.utils import (
    POLLEN_LLM_MAX_MESSAGE_LENGTH,
    download_file_from_s3,
    dump_model_parameters_to_file,
    get_n_cuda_devices,
    get_pyarrow_buffer_from_table,
    load_model_parameters_from_file,
    sum_of_squares,
    upload_file_to_s3,
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
        use_s3_comm: bool = False,
        s3_comm_config: DictConfig | None = None,
    ) -> None:
        super().__init__()
        # NodeManager general attributes
        self.name: str = getfqdn()
        self.properties: dict[str, Scalar] = {}
        self.all_gpus: list[GPU] = list(nvsmi.get_gpus())
        self.run_uuid = run_uuid

        self.use_s3_comm = use_s3_comm
        self.s3_comm_config = s3_comm_config
        self.node_manager_uuid = run_uuid + "-" + str(uuid.uuid4())

        self.client_fn = client_fn
        self.refresh_period = refresh_period

        self._create_remote_up_down()

        # Set up Queues
        self.task_queue: QueueType = Queue()
        # One result_queue for all GPUs
        self.result_queue: QueueType = Queue()
        # Get node properties about hardware accelerators
        self.properties = self._get_node_properties()
        # Set how many processes can be run on each GPU given the properties
        [(k, v.concurrency) for k, v in self.node.device_info.items()]
        # log(DEBUG, "Max processes per device: %s", max_proc_device)
        # Set up round parameters SharedMemory
        # Call the monkey-patch for the resource-register
        remove_shm_from_resource_tracker()
        # Shared memory for round parameters
        self.round_parameters, self.round_parameters_sh = get_parameters_shm(
            parameters=parameters,
            create=True,
            name=self.node_manager_uuid + POLLEN_PARAMETERS_SHM,
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
            cpus = len(psutil.Process().cpu_affinity())  # type: ignore[reportArgumentType]
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

    def _create_remote_up_down(self) -> None:
        """Create the remote uploader/downloader."""
        if self.use_s3_comm:
            bucket_uri = f"s3://{self.s3_comm_config.bucket_name}"  # type: ignore[union-attr]
            self.remote_up_down = RemoteUploaderDownloader(
                bucket_uri=bucket_uri,
                backend_kwargs={
                    "bucket": self.s3_comm_config.bucket_name,  # type: ignore[union-attr]
                    "prefix": f"{self.run_uuid}/server",  # Don't touch
                    "region_name": None,  # Not necessary
                    "endpoint_url": None,  # Will be read from env var
                    "aws_access_key_id": None,  # Will be read from config file
                    "aws_secret_access_key": None,  # Will be read from config file
                    "aws_session_token": None,  # Will be automatically generated
                    "client_config": OmegaConf.to_container(
                        self.s3_comm_config.backend_kwargs.client_config  # type: ignore[union-attr]
                    ),  # And using defaults
                    "transfer_config": None,  # Using defaults
                },
                file_path_format_string="{remote_file_name}",  # Don't touch
                num_concurrent_uploads=1,
                upload_staging_folder=None,  # Don't touch, it's /tmp by default
                use_procs=True,  # Don't touch
                num_attempts=self.s3_comm_config.num_attempts,  # type: ignore[union-attr]
            )
            self.remote_up_down.init(run_name=self.run_uuid)

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
        for worker in self.workers_dict.values():
            start_worker(worker)
        log(DEBUG, "NodeManager %s: all workers started.", self.name)

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
        _fl_instructions_config, fl_instructions_config_sh = get_config_shm(
            config=config,
            create=True,
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
        )
        set_config_shm(config, fl_instructions_config_sh)
        set_parameters_shm(self.round_parameters, parameters)
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
        # Zero out the n_samples shared memories
        for ww_ss in w_s:
            set_num_samples_shm(ww_ss, 0)
        # Close the config shared memory
        fl_instructions_config_sh.close()
        fl_instructions_config_sh.unlink()
        # Collect statistics to pyarrow.Table
        clients_training_stats = pa.Table.from_pydict(stats)
        # Add info to `clients_training_stats`
        clients_training_stats = add_constant_column_to_clients_stats_table(
            clients_training_stats, "node", self.name
        )
        clients_training_stats = add_constant_column_to_clients_stats_table(
            clients_training_stats, "server_round", int(config["server_round"])
        )
        # Prepare statistics to be sent to the server
        clients_training_buf = get_pyarrow_buffer_from_table(clients_training_stats)
        # Append the statistics to the node_train_metrics
        node_train_metrics = node_train_metrics | {
            "stats": clients_training_buf.to_pybytes(),
        }
        # Return the results
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def _collaborative_fit(
        self, config: Config, list_of_cids_to_train: list[str], parameters: NDArrays
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Update parameters shared memory
        set_parameters_shm(self.round_parameters, parameters)
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
            # Append NodeManager's config
            config["MASTER_PORT"] = str(get_free_tcp_port())
            # NOTE: Putting the node_manager_uuid in the config fails
            config["run_uuid"] = self.run_uuid
            # Update instruction config shared memory
            _fl_instructions_config, fl_instructions_config_sh = get_config_shm(
                config=config,
                create=True,
                name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
            )
            set_config_shm(config, fl_instructions_config_sh)
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
        # Collect statistics to pyarrow.Table
        clients_training_stats = pa.Table.from_pydict(stats)
        # Add info to `clients_training_stats`
        clients_training_stats = add_constant_column_to_clients_stats_table(
            clients_training_stats, "node", self.name
        )
        clients_training_stats = add_constant_column_to_clients_stats_table(
            clients_training_stats, "server_round", int(config["server_round"])
        )
        # Prepare statistics to be sent to the server
        clients_training_buf = get_pyarrow_buffer_from_table(clients_training_stats)
        # Append the statistics to the node_train_metrics
        node_train_metrics = node_train_metrics | {
            "stats": clients_training_buf.to_pybytes(),
        }
        # Return the results
        return (
            aggregated_params,
            sum_of_samples,
            node_train_metrics,
        )

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Implement the fit step."""
        # Get the server round
        server_round = int(config["server_round"])
        # If applicable, override the parameters with values from S3 Object Store
        if self.use_s3_comm:
            log(
                DEBUG,
                "NodeManager %s: pulling parameters from S3 Object Store",
                self.name,
            )
            # Check whether the server has uploaded the parameters
            file_found = False
            remote_file_name_no_ext = (
                f"s3://{self.s3_comm_config.bucket_name}/"  # type: ignore[union-attr]
                f"{self.run_uuid}/server/"
                f"{int(server_round) - 1}/current_server_parameters"
            )
            while not file_found:
                file_found = validate_given_remote_path(
                    remote_file_name_no_ext + ".bin"
                ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
                time.sleep(0.5)
            # Set the file names depending on the extension found
            remote_file_name = (
                f"{int(server_round) - 1}/current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else f"{int(server_round) - 1}/current_server_parameters.npz"
            )
            local_file_name = (
                Path.cwd() / f"{self.node_manager_uuid}_current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else Path.cwd()
                / f"{self.node_manager_uuid}_current_server_parameters.npz"
            )
            # Download the parameters
            file_downloaded = False
            while not file_downloaded:
                try:
                    download_file_from_s3(
                        self.remote_up_down, remote_file_name, local_file_name
                    )
                    file_downloaded = True
                except FileNotFoundError:
                    time.sleep(0.5)
                except RuntimeError as e:
                    log(
                        ERROR,
                        "NodeManager %s: error while pulling parameters from S3 Object"
                        " Store. Refreshing the connection.",
                        self.name,
                        exc_info=e,
                        stack_info=True,
                    )
                    self.remote_up_down.post_close()
                    self._create_remote_up_down()
            log(INFO, "Read server parameters from disk")
            parameters = load_model_parameters_from_file(local_file_name)
            log(INFO, "Server parameters have been read from disk")

        # log(DEBUG, "NodeManager %s: fit with config %s", self.name, config)
        start_time = time.time()
        # Restart all the worker every `self.refresh_period` rounds
        if server_round % self.refresh_period == 0:
            # Close and remove the workers
            self._close_workers()
            # Re-create and start the workers
            self._create_and_start_workers()
        # Extract assignments from config
        assignments = config.pop("merged", "0,1")
        list_of_cids_to_train = cast(str, assignments).split(",")
        node_train_metrics: dict[str, Scalar] = {}
        aggregated_params: NDArrays = []
        sum_of_samples: int = 0
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
        node_train_metrics.update({
            "node_training_time_s": float(time.time() - start_time),
        })
        log(
            DEBUG,
            "NodeManager %s: results have been processed. "
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

        # If applicable, push the aggregated parameters to S3 Object Store
        if self.use_s3_comm:
            # Set the file names
            remote_file_name = f"{server_round}/{self.node_manager_uuid}/parameters.npz"
            local_file_name = Path.cwd() / f"{self.node_manager_uuid}_parameters.npz"
            log(INFO, "Dump node parameters to disk")
            dump_model_parameters_to_file(local_file_name, aggregated_params)
            log(
                DEBUG,
                "NodeManager %s: pushing parameters to S3 Object Store",
                self.name,
            )
            # Upload the parameters to S3 Object Store
            try:
                upload_file_to_s3(
                    self.remote_up_down, remote_file_name, local_file_name
                )
            except RuntimeError as e:
                log(
                    ERROR,
                    "NodeManager %s: error while pushing parameters from S3 Object"
                    " Store. Refreshing the connection.",
                    self.name,
                    exc_info=e,
                    stack_info=True,
                )
                self.remote_up_down.post_close()
                self._create_remote_up_down()
            log(INFO, "Node parameters have been pushed to S3 Object Store")
            node_train_metrics.update({
                "endpoint_id": self.node_manager_uuid,
            })

            # Return results
            return (
                [np.array([[0.0], [0.0]])],
                int(sum_of_samples),
                node_train_metrics,
            )
        else:
            # Return results
            return (
                aggregated_params,
                int(sum_of_samples),
                node_train_metrics,
            )

    def evaluate(
        self, parameters: NDArrays, config: dict
    ) -> tuple[float, int, dict[Any, Any]]:
        """Implement the evaluation step."""
        # Get the server round
        server_round = config["server_round"]
        # If applicable, override the parameters with values from S3 Object Store
        if self.use_s3_comm:
            log(
                DEBUG,
                "NodeManager %s: pulling parameters from S3 Object Store",
                self.name,
            )
            # Check whether the server has uploaded the parameters
            file_found = False
            remote_file_name_no_ext = (
                f"s3://{self.s3_comm_config.bucket_name}/"  # type: ignore[union-attr]
                f"{self.run_uuid}/server/"
                f"{int(server_round)}/current_server_parameters"
            )
            while not file_found:
                file_found = validate_given_remote_path(
                    remote_file_name_no_ext + ".bin"
                ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
                time.sleep(0.5)
            # Set the file names depending on the extension found
            remote_file_name = (
                f"{int(server_round)}/current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else f"{int(server_round)}/current_server_parameters.npz"
            )
            local_file_name = (
                Path.cwd() / f"{self.node_manager_uuid}_current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else Path.cwd()
                / f"{self.node_manager_uuid}_current_server_parameters.npz"
            )
            # Download the parameters
            file_downloaded = False
            while not file_downloaded:
                try:
                    download_file_from_s3(
                        self.remote_up_down, remote_file_name, local_file_name
                    )
                    file_downloaded = True
                except FileNotFoundError:
                    time.sleep(0.5)
                except RuntimeError as e:
                    log(
                        ERROR,
                        "NodeManager %s: error while pulling parameters from S3 Object"
                        " Store. Refreshing the connection.",
                        self.name,
                        exc_info=e,
                        stack_info=True,
                    )
                    self.remote_up_down.post_close()
                    self._create_remote_up_down()
            log(INFO, "Read server parameters from disk")
            parameters = load_model_parameters_from_file(local_file_name)
            log(INFO, "Server parameters have been read from disk")

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
                name=self.node_manager_uuid + POLLEN_CONFIG_SHM,
            )
            set_config_shm(config, self.fl_instructions_config_sh)
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
        use_s3_comm=cfg.use_s3_comm,
        s3_comm_config=cfg.s3_comm_config,
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
        fl.client.start_client(
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
