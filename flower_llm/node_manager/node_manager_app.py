"""Flower NodeManagerApp."""

import copy
import gc
from logging import DEBUG
from multiprocessing.shared_memory import SharedMemory
import os
import pickle
from tempfile import TemporaryDirectory
import time
from typing import cast
import uuid
import cloudpickle
from multiprocessing.queues import Queue as QueueType
from flwr.client import ClientApp
from flwr.client.typing import ClientFnExt, Mod
from flwr.common import NDArrays
from flwr.common.logger import log
from multiprocess import Queue, set_start_method  # type: ignore[reportAttributeAccessIssue]
from omegaconf import DictConfig, OmegaConf
from composer.loggers import RemoteUploaderDownloader
import torch

from flower_llm.clients.llm_client_functions import get_raw_model_parameters
from flower_llm.conf.base_schema import BaseConfig
from flower_llm.node_manager.utils import (
    ModelParametersMetadata,
    close_all_shms,
    remove_shm_from_resource_tracker,
)
from flower_llm.node_manager.worker import Worker, create_new_worker, start_worker
from flower_llm.resources_manager import get_node_properties
from flower_llm.utils import get_n_cuda_devices

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
        self.properties = {"node": str(self.node)}
        self.node_manager_uuid = self.cfg.run_uuid + "-" + str(uuid.uuid4())
        assert self.node.device_info is not None
        self.refresh_period = self.cfg.pollen.refresh_period
        self.node_manager_temp_dir = TemporaryDirectory()
        self.remote_up_down: RemoteUploaderDownloader
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
