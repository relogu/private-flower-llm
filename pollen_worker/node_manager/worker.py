"""TODO: Add description here."""
import copy
import gc
import sys
import time
import traceback
import uuid
from logging import DEBUG, ERROR
from multiprocessing import resource_tracker  # type: ignore[attr-defined]
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Optional, Tuple

import multiprocess as mp
import torch
import torch.distributed as dist
from composer.cli.launcher import _patch_env
from composer.utils.misc import get_free_tcp_port
from flwr.common import Config, NDArrays
from flwr.common.logger import log

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
    set_config_shm,
    set_eval_loss_shm,
    set_num_samples_shm,
    set_parameters_shm,
)
from pollen_worker.utils import partially_aggregate, partially_aggregate_metrics


class Worker(mp.Process):  # type: ignore
    """Worker Process child of the NodeManager."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualLLMClient],
        worker_uuid: str,
        task_queue: QueueType,
        result_queue: QueueType,
        node_manager_uuid: str,
        run_uuid: str,
        parameters: NDArrays,
        worker_rank: int,
    ) -> None:
        super(Worker, self).__init__()
        self.worker_uuid = worker_uuid
        self.client_fn: Callable[[int], VirtualLLMClient] = client_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.node_manager_uuid = node_manager_uuid
        self.run_uuid = run_uuid
        self.parameters = parameters
        self.worker_rank = worker_rank
        self.worker_metrics_sh: SharedMemory | None = None
        self.worker_metrics: Config | None = None
        self.auto_terminate = False

    def _fit_action(
        self, client: VirtualLLMClient, fl_instructions_config: Config
    ) -> None:
        """Fit action."""
        # Call fit on shared parameters
        fit_trained_weights, fit_num_samples, train_metrics = client.fit(
            self.round_parameters, fl_instructions_config
        )
        # log(
        #     INFO,
        #     "Worker %s with rank %s successfully obtained the training results from"
        #     " client %s: (%s, %s, %s).",
        #     self.worker_uuid,
        #     self.worker_rank,
        #     client.cid,
        #     len(fit_trained_weights),
        #     fit_num_samples,
        #     len(train_metrics),
        # )
        # Worker's partial aggregation for parameters and n_samples
        (p_agg_params, p_agg_samples) = partially_aggregate(
            (self.worker_parameters, self.worker_num_samples[0]),
            (copy.deepcopy(fit_trained_weights), copy.deepcopy(fit_num_samples)),
        )
        # Worker's partial aggregation for metrics
        (p_agg_samples, p_agg_metrics) = partially_aggregate_metrics(
            (int(self.worker_num_samples[0]), self.worker_metrics),
            (copy.deepcopy(fit_num_samples), copy.deepcopy(train_metrics)),
        )
        # Update shared memories
        set_num_samples_shm(self.worker_num_samples, p_agg_samples)
        set_parameters_shm(self.worker_parameters, p_agg_params)
        # Destroy the shared memory for the metrics
        if self.worker_metrics_sh is not None:
            self.worker_metrics_sh.close()
            self.worker_metrics_sh.unlink()
        # NOTE: Now we know the structure and we can create the train metrics
        # shared memory. Since the structure might changed, we cannot assume
        # a fixed size for the shared memory.
        (
            self.worker_metrics,
            self.worker_metrics_sh,
        ) = get_config_shm(
            config=p_agg_metrics,
            create=True,
            name=self.worker_uuid + POLLEN_METRICS_SHM,  # noqa: F821
        )
        set_config_shm(p_agg_metrics, self.worker_metrics_sh)
        # log(
        #     INFO,
        #     "Worker %s with rank %s successfully trained client %s.",
        #     self.worker_uuid,
        #     self.worker_rank,
        #     client.cid,
        # )

    def _evaluate_action(
        self, client: VirtualLLMClient, fl_instructions_config: Config
    ) -> None:
        """Evaluate action."""
        # Call evaluate on shared parameters
        eval_loss, eval_num_samples, eval_metrics = client.evaluate(
            self.round_parameters, fl_instructions_config
        )
        # log(
        #     INFO,
        #     "Worker %s with rank %s successfully obtained the validation results from"
        #     " client %s: (%s, %s, %s).",
        #     self.worker_uuid,
        #     self.worker_rank,
        #     client.cid,
        #     eval_loss,
        #     eval_num_samples,
        #     eval_metrics,
        # )
        if self.worker_rank == 0:
            # TODO: Worker's partial aggregation for eval_loss
            # Worker's partial aggregation for metrics
            (agg_n_samples, p_agg_metrics) = partially_aggregate_metrics(
                (int(self.worker_num_samples[0]), self.worker_metrics),
                (copy.deepcopy(eval_num_samples), copy.deepcopy(eval_metrics)),
            )
            set_num_samples_shm(self.worker_num_samples, agg_n_samples)
            set_eval_loss_shm(self.worker_eval_loss, eval_loss)
            # Destroy the shared memory for the metrics
            if self.worker_metrics_sh is not None:
                self.worker_metrics_sh.close()
                self.worker_metrics_sh.unlink()
            # NOTE: Now we know the structure and we can create the train metrics
            # shared memory. Since the structure might changed, we cannot assume
            # a fixed size for the shared memory.
            (
                self.worker_metrics,
                self.worker_metrics_sh,
            ) = get_config_shm(
                config=p_agg_metrics,
                create=True,
                name=self.worker_uuid + POLLEN_METRICS_SHM,  # noqa: F821
            )
            set_config_shm(p_agg_metrics, self.worker_metrics_sh)
        # log(
        #     INFO,
        #     "Worker %s with rank %s successfully evaluated client %s.",
        #     self.worker_uuid,
        #     self.worker_rank,
        #     client.cid,
        # )

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
        # Try to execute the task of the client
        try:
            if action == "fit":
                # Fit the client
                # NOTE: We force the workers to be independent and
                # not to collaborate while training clients
                with _patch_env(
                    RANK="0",
                    WORLD_SIZE="1",
                    LOCAL_RANK="0",
                    LOCAL_WORLD_SIZE="1",
                    NODE_RANK="0",
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(get_free_tcp_port()),
                    PYTHONUNBUFFERED="1",
                    NCCL_ASYNC_ERROR_HANDLING="1",
                    RUN_UUID=self.worker_uuid,
                    APPOINTED_CUDA_DEVICE=str(self.worker_rank),
                ):
                    # Trying to destroy the process group of PyTorch Distributed, if any
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    # Lauch the fit routine
                    self._fit_action(tmp_client, fl_instructions_config)
                    # Take the timestamp after the task is done
                    end_time = time.time_ns()
                    # Put the result in the result queue
                    self.result_queue.put(
                        [int(tmp_client.cid), start_time, end_time, self.worker_uuid]
                    )
            elif action == "evaluate":
                # Eval the client
                # Prevent slave workers to log to the console
                if self.worker_rank > 0:
                    tmp_client.cfg.log_to_console = False  # type: ignore[union-attr]
                # NOTE: We force the workers to collaborate with each other
                # in doing the evaluation
                with _patch_env(
                    RANK=str(self.worker_rank),
                    WORLD_SIZE=str(torch.cuda.device_count()),
                    LOCAL_RANK=str(self.worker_rank),
                    LOCAL_WORLD_SIZE=str(torch.cuda.device_count()),
                    NODE_RANK="0",
                    MASTER_ADDR="127.0.0.1",
                    MASTER_PORT=str(fl_instructions_config["MASTER_PORT"]),
                    PYTHONUNBUFFERED="1",
                    NCCL_ASYNC_ERROR_HANDLING="1",
                    RUN_UUID=self.node_manager_uuid,
                ):
                    # Lauch the evaluate routine
                    self._evaluate_action(tmp_client, fl_instructions_config)
                    # Trying to destroy the process group of PyTorch Distributed, if any
                    if dist.is_initialized():
                        dist.destroy_process_group()
                    # Since they are collaborating, only rank 0 returns the result
                    if self.worker_rank == 0:
                        # Take the timestamp after the task is done
                        end_time = time.time_ns()
                        # Put the result in the result queue
                        self.result_queue.put(
                            [
                                int(tmp_client.cid),
                                start_time,
                                end_time,
                                self.worker_uuid,
                            ]
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
            traceback.print_exc(*sys.exc_info())
            self.result_queue.put([-1, 0, 0, self.worker_uuid])
            if action == "fit":
                self.task_queue.put((client_id, action))
            elif self.worker_rank == 0:
                self.task_queue.put((client_id, action))
            self.auto_terminate = True

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
        set_num_samples_shm(self.worker_num_samples, 0)
        # Evaluation loss shared memory
        self.worker_eval_loss, self.worker_eval_loss_sh = get_eval_loss_shm(
            create=True,
            name=self.worker_uuid + POLLEN_EVAL_LOSS_SHM,  # noqa: F821
        )

    def run(self) -> None:
        """Start the process."""
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
    run_uuid: str,
    parameters: NDArrays,
    worker_rank: int,
) -> Worker:
    """Create a new Worker."""
    # Generate the Worker's UUID
    worker_uuid = node_manager_uuid + "-" + str(uuid.uuid4())
    # Create the Worker object
    worker = Worker(
        client_fn=client_fn,
        worker_uuid=worker_uuid,
        task_queue=task_queue,
        result_queue=result_queue,
        node_manager_uuid=node_manager_uuid,
        run_uuid=run_uuid,
        parameters=parameters,
        worker_rank=worker_rank,
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
