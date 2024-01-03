"""TODO: Add description here."""
import copy
import gc
import time
import uuid
from logging import DEBUG, ERROR, INFO
from multiprocessing import resource_tracker  # type: ignore[attr-defined]
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional, Tuple

import multiprocess as mp
import torch
from composer.cli.launcher import _patch_env
from flwr.common import Config, NDArrays
from flwr.common.logger import log

from pollen_worker.clients.llm_client_functions import (
    set_all_data_paths,
    set_n_workers_dataloaders,
)
from pollen_worker.clients.virtual_llm_client import VirtualLLMClient
from pollen_worker.node_manager.utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_EVAL_LOSS_SHM,
    POLLEN_METRICS_SHM,
    POLLEN_N_SAMPLES_SHM,
    POLLEN_PARAMETERS_SHM,
    get_config_shm,
    get_eval_loss_shm,
    get_num_samples_shm,
    get_parameters_shm,
    set_eval_loss_shm,
    set_num_samples_shm,
    set_parameters_shm,
)


class Worker(mp.Process):  # type: ignore
    """Worker Process child of the NodeManager."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualLLMClient],
        worker_uuid: str,
        task_queue: QueueType,
        result_queue: QueueType,
        node_manager_uuid: str,
        parameters: NDArrays,
        n_workers: int,
        worker_rank: int,
        port: str,
    ) -> None:
        super(Worker, self).__init__()
        self.worker_uuid = worker_uuid
        self.client_fn: Callable[[int], VirtualLLMClient] = client_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.node_manager_uuid = node_manager_uuid
        self.auto_terminate = False
        self.parameters = parameters
        self.n_workers = n_workers
        self.worker_rank = worker_rank
        self.port = port
        self.worker_metrics_sh: SharedMemory | None = None

    def _fit_action(
        self, client: VirtualLLMClient, fl_instructions_config: Config
    ) -> None:
        """Fit action."""
        # Call fit on shared parameters
        fit_trained_weights, fit_num_samples, train_metrics = client.fit(
            self.round_parameters, fl_instructions_config
        )
        log(
            INFO,
            "Worker %s with rank %s successfully obtained the training results from"
            " client %s: (%s, %s, %s).",
            self.worker_uuid,
            self.worker_rank,
            client.cid,
            len(fit_trained_weights),
            fit_num_samples,
            len(train_metrics),
        )
        if self.worker_rank == 0:
            set_num_samples_shm(self.worker_num_samples, fit_num_samples)
            set_parameters_shm(self.worker_parameters, fit_trained_weights)
            # NOTE: Now we know the structure and we can create the train metrics
            # shared memory
            (
                self.worker_metrics,
                self.worker_metrics_sh,
            ) = get_config_shm(
                config=train_metrics,
                create=self.worker_metrics_sh is None,
                name=self.worker_uuid + POLLEN_METRICS_SHM,  # noqa: F821
            )
        log(
            INFO,
            "Worker %s with rank %s successfully trained client %s.",
            self.worker_uuid,
            self.worker_rank,
            client.cid,
        )

    def _evaluate_action(
        self, client: VirtualLLMClient, fl_instructions_config: Config
    ) -> None:
        """Evaluate action."""
        # Call evaluate on shared parameters
        eval_loss, eval_num_samples, eval_metrics = client.evaluate(
            self.round_parameters, fl_instructions_config
        )
        log(
            INFO,
            "Worker %s with rank %s successfully obtained the training results from"
            " client %s: (%s, %s, %s).",
            self.worker_uuid,
            self.worker_rank,
            client.cid,
            eval_loss,
            eval_num_samples,
            eval_metrics,
        )
        if self.worker_rank == 0:
            set_num_samples_shm(self.worker_num_samples, eval_num_samples)
            set_eval_loss_shm(self.worker_eval_loss, eval_loss)
            # NOTE: Now we know the structure and we can create the train metrics
            # shared memory
            (
                self.worker_metrics,
                self.worker_metrics_sh,
            ) = get_config_shm(
                config=eval_metrics,
                create=self.worker_metrics_sh is None,
                name=self.worker_uuid + POLLEN_METRICS_SHM,  # noqa: F821
            )
        log(
            INFO,
            "Worker %s with rank %s successfully evaluated client %s.",
            self.worker_uuid,
            self.worker_rank,
            client.cid,
        )

    def process_task(self, client_id: int, action: str = "fit") -> None:
        """Process the received task."""
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        # Loads a dict from the shared memory buffer
        # FL config shared memory
        fl_instructions_config, fl_instructions_config_sh = get_config_shm(
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        # Load client
        tmp_client = self.client_fn(client_id)
        # NOTE: We MUST change the save folder for the checkpoints,
        # it won't train otherwise
        if tmp_client.cfg.save_folder is not None:  # type: ignore[union-attr]
            tmp_client.cfg.save_folder = (  # type: ignore[union-attr]
                tmp_client.cfg.save_folder  # type: ignore[union-attr]
                + "_c"
                + str(tmp_client.cid)
            )
        # NOTE: Prevent slave workers to log to the console
        if self.worker_rank > 0:
            tmp_client.cfg.log_to_console = False  # type: ignore[union-attr]
        # Automatically setting the `n_workers` parameter based on CPU available
        tmp_client.cfg = set_n_workers_dataloaders(
            tmp_client.cfg  # type: ignore[union-attr]
        )
        # NOTE: When using remote data, we need one tmp folder per worker
        if tmp_client.cfg.data_remote is not None:  # type: ignore[union-attr]
            # Set the appropriate path given the `client_id`
            new_remote_path = (
                str(tmp_client.cfg.data_remote)  # type: ignore[union-attr]
                + f"/client_{client_id}"
            )
            tmp_client.cfg = set_all_data_paths(
                tmp_client.cfg, new_remote_path, False
            )
        # TODO: This must be the same for all the NodeManagers in a node
        # (if any), linked to run_uuid
        new_local_path = (
            str(tmp_client.cfg.data_local)  # type: ignore[union-attr]
            + f"/{self.node_manager_uuid}_client_{client_id}"
        )
        tmp_client.cfg = set_all_data_paths(tmp_client.cfg, new_local_path)
        # Set `max_duration` as the number of steps times the number of rounds
        max_duration = int(fl_instructions_config["server_round"]) * int(
            tmp_client.cfg.local_steps  # type: ignore[union-attr]
        )
        tmp_client.cfg.max_duration = f"{max_duration}ba"  # type: ignore[union-attr]
        # Forcing not to load the model from a checkpoint
        # From: https://github.com/mosaicml/composer/blob/2aa50e7741a077ff21f5743934fbcf4b755d441e/composer/trainer/trainer.py#L639
        tmp_client.cfg.load_ignore_keys = ["state/model/*"]  # type: ignore[union-attr]
        # Try to execute the task of the client
        try:
            if action == "fit":
                self._fit_action(tmp_client, fl_instructions_config)
            elif action == "evaluate":
                self._evaluate_action(tmp_client, fl_instructions_config)
            # Take the timestamp after the task is done
            end_time = time.time_ns()
            if self.worker_rank == 0:
                self.result_queue.put(
                    [int(tmp_client.cid), start_time, end_time, self.worker_uuid]
                )
        except Exception as e:
            log(
                ERROR,
                "Worker %s failed executing %s for client %s\n\t\t\texception %s.",
                self.worker_uuid,
                action,
                client_id,
                e,
            )
            if self.worker_rank == 0:
                self.task_queue.put((client_id, action))
                self.result_queue.put([-1, 0, 0, self.worker_uuid])
            # Take the timestamp after the task is done
            end_time = time.time_ns()
        # # Removing the tmp folder used for the dataset
        # if self.worker_rank == 0:
        #     shutil.rmtree(Path(new_local_path), ignore_errors=True)

    def _unregister_shms(self) -> None:
        """Unregister shared memories."""
        _, fl_instructions_config_sh = get_config_shm(
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        resource_tracker.unregister(
            name=fl_instructions_config_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        resource_tracker.unregister(
            name=self.round_parameters_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        if hasattr(self, "worker_parameters_sh"):
            resource_tracker.unregister(
                name=self.worker_parameters_sh._name,  # type: ignore[attr-defined]
                rtype="shared_memory",
            )
        if hasattr(self, "worker_num_samples_sh"):
            resource_tracker.unregister(
                name=self.worker_num_samples_sh._name,  # type: ignore[attr-defined]
                rtype="shared_memory",
            )
        if self.worker_metrics_sh is not None:
            resource_tracker.unregister(
                name=self.worker_metrics_sh._name,  # type: ignore[attr-defined]
                rtype="shared_memory",
            )
        if hasattr(self, "worker_eval_loss_sh"):
            resource_tracker.unregister(
                name=self.worker_eval_loss_sh._name,  # type: ignore[attr-defined]
                rtype="shared_memory",
            )

    def _link_shms(
        self,
    ) -> None:
        """Create the Worker's shared memories."""
        # NOTE: This is the NodeManager's shared memories. Workers should only
        # read this. NodeManager should only write this.
        # Shared memory for round parameters
        self.round_parameters, self.round_parameters_sh = get_parameters_shm(
            parameters=self.parameters,
            name=self.node_manager_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
        )
        if self.worker_rank == 0:
            # NOTE: This is the Worker's shared memory for the fit results.
            # NodeManager should only read this. Worker should only write this.
            # Shared memory for worker's parameters
            self.worker_parameters, self.worker_parameters_sh = get_parameters_shm(
                create=True,
                parameters=self.parameters,
                name=self.worker_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
            )
            # Number of samples shared memory
            self.worker_num_samples, self.worker_num_samples_sh = get_num_samples_shm(
                create=True,
                name=self.worker_uuid + POLLEN_N_SAMPLES_SHM,  # noqa: F821
            )
            # Evaluation loss shared memory
            self.worker_eval_loss, self.worker_eval_loss_sh = get_eval_loss_shm(
                create=True,
                name=self.worker_uuid + POLLEN_EVAL_LOSS_SHM,  # noqa: F821
            )

    def run(self) -> None:
        """Start the process."""
        with _patch_env(
            RANK=str(self.worker_rank),
            WORLD_SIZE=str(self.n_workers),
            LOCAL_RANK=str(self.worker_rank),
            LOCAL_WORLD_SIZE=str(self.n_workers),
            NODE_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=self.port,
            PYTHONUNBUFFERED="1",
            NCCL_ASYNC_ERROR_HANDLING="1",
        ):
            ## Create shared memories
            # NOTE: This goes here because it needs to be done in the child process!
            # This is the first piece of code of the worker that live in the child
            # process, the `__init__()` function does not.
            self._link_shms()
            ## Task loop
            task: Optional[Tuple[int, str]] = None
            for task in iter(self.task_queue.get, None):
                cid, action = task  # type: ignore[misc]
                self.process_task(cid, action)  # type: ignore[has-type]
                if self.auto_terminate:
                    break
            torch.cuda.empty_cache()
            gc.collect()
            ## Un-register shared memories
            # NOTE: Bug https://bugs.python.org/issue39959#msg364351
            self._unregister_shms()


def create_new_worker(
    client_fn: Callable[[int], VirtualLLMClient],
    task_queue: QueueType,
    result_queue: QueueType,
    node_manager_uuid: str,
    parameters: NDArrays,
    n_workers: int,
    worker_rank: int,
    port: str,
) -> Worker:
    """Create a new Worker."""
    # Generate the Worker's UUID
    worker_uuid = node_manager_uuid + str(uuid.uuid4())
    # Create the Worker object
    worker = Worker(
        client_fn=client_fn,
        worker_uuid=worker_uuid,
        task_queue=task_queue,
        result_queue=result_queue,
        node_manager_uuid=node_manager_uuid,
        parameters=parameters,
        n_workers=n_workers,
        worker_rank=worker_rank,
        port=port,
    )
    # Return the Worker object, its UUID, and the shared objects
    return worker


def start_worker(worker: Worker):
    """Start a worker."""
    worker.start()
    log(
        DEBUG,
        "Worker %s with rank %s started.",
        worker.worker_uuid,
        worker.worker_rank,
    )
