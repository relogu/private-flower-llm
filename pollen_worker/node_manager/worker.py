"""TODO: Add description here."""
import gc
import pickle
import time
import uuid
from logging import ERROR, INFO
from multiprocessing import resource_tracker  # type: ignore[attr-defined]
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from typing import Callable, Dict, Optional, Tuple

import multiprocess as mp
import numpy as np
import torch
from flwr.common import NDArrays, Scalar
from flwr.common.logger import log

from pollen_worker.clients.virtual_llm_client import VirtualLLMClient
from pollen_worker.node_manager.utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_N_SAMPLES_SHM,
    POLLEN_PARAMETERS_SHM,
    POLLEN_TRAIN_METRICS_SHM,
    get_config_shm,
    get_num_samples_shm,
    get_parameters_shm,
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
        train_metrics: Dict[str, Scalar],
    ) -> None:
        super(Worker, self).__init__()
        self.worker_uuid = worker_uuid
        self.client_fn: Callable[[int], VirtualLLMClient] = client_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.node_manager_uuid = node_manager_uuid
        self.auto_terminate = False
        self.parameters = parameters
        self.train_metrics = train_metrics

    def process_task(self, client_id: int) -> None:
        """Process the received task."""
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        # Loads a dict from the shared memory buffer
        fl_instructions_config = pickle.loads(self.fl_instructions_config_sh.buf)
        # Load client
        tmp_client = self.client_fn(client_id)
        # NOTE: We MUST change the save folder for the checkpoints,
        # it won't train otherwise
        tmp_client.cfg.save_folder = (  # type: ignore[union-attr]
            tmp_client.cfg.save_folder  # type: ignore[union-attr]
            + "_c"
            + str(client_id)
            + "_r"
            + str(fl_instructions_config["server_round"])
        )
        # NOTE: This is necessary to prevent erros when executing a
        # config with the same `cfg.save_folder`
        tmp_client.cfg.save_overwrite = True  # type: ignore[union-attr]
        # Try to train the client
        done = False
        fit_trained_weights: Optional[NDArrays] = None
        fit_num_samples: Optional[int] = None
        train_metrics: Optional[Dict[str, Scalar]] = None
        while not done:
            try:
                # Call fit on shared parameters
                fit_trained_weights, fit_num_samples, train_metrics = tmp_client.fit(
                    self.round_parameters, fl_instructions_config
                )
                log(
                    INFO,
                    "Worker %s successfully obtained the training results from"
                    " client %s: (%s, %s, %s).",
                    self.worker_uuid,
                    client_id,
                    len(fit_trained_weights),
                    fit_num_samples,
                    len(train_metrics),
                )
                set_num_samples_shm(self.worker_num_samples, fit_num_samples)
                set_parameters_shm(self.worker_parameters, fit_trained_weights)
                # NOTE: Now we know the structure and we can create the train metrics
                # shared memory
                (
                    self.worker_train_metrics,
                    self.worker_train_metrics_sh,
                ) = get_config_shm(
                    config=train_metrics,
                    create=True,
                    name=self.worker_uuid + POLLEN_TRAIN_METRICS_SHM,  # noqa: F821
                )
                # Take the timestamp after the task is done
                end_time = time.time_ns()
                self.result_queue.put(
                    [int(client_id), start_time, end_time, self.worker_uuid]
                )
                log(
                    INFO,
                    "Worker %s successfully trained client %s.",
                    self.worker_uuid,
                    client_id,
                )
            except Exception as e:
                log(
                    ERROR,
                    "Worker %s failed in training client %s with exception %s.",
                    self.worker_uuid,
                    client_id,
                    e,
                )
                self.task_queue.put(client_id)
                # Take the timestamp after the task is done
                end_time = time.time_ns()
                self.result_queue.put([-1, 0, 0, self.worker_uuid])
            self.auto_terminate = True
            done = True
        torch.cuda.empty_cache()
        gc.collect()

    def _unregister_shms(self) -> None:
        """Unregister shared memories."""
        resource_tracker.unregister(
            name=self.fl_instructions_config_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        resource_tracker.unregister(
            name=self.round_parameters_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        resource_tracker.unregister(
            name=self.worker_parameters_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        resource_tracker.unregister(
            name=self.worker_train_metrics_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )
        resource_tracker.unregister(
            name=self.worker_num_samples_sh._name,  # type: ignore[attr-defined]
            rtype="shared_memory",
        )

    def _link_shms(
        self,
    ) -> None:
        """Create the Worker's shared memories."""
        # NOTE: This is the NodeManager's shared memories. Workers should only
        # read this. NodeManager should only write this.
        # FL config shared memory
        self.fl_instructions_config, self.fl_instructions_config_sh = get_config_shm(
            name=self.node_manager_uuid + POLLEN_CONFIG_SHM,  # noqa: F821
        )
        # Shared memory for round parameters
        self.round_parameters, self.round_parameters_sh = get_parameters_shm(
            parameters=self.parameters,
            name=self.node_manager_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
        )
        # NOTE: This is the Worker's shared memory for the fit results.
        # NodeManager should only read this. Worker should only write this.
        # Shared memory for worker's parameters
        self.worker_parameters, self.worker_parameters_sh = get_parameters_shm(
            parameters=self.parameters,
            name=self.worker_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
        )
        # Number of samples shared memory
        self.worker_num_samples, self.worker_num_samples_sh = get_num_samples_shm(
            name=self.worker_uuid + POLLEN_N_SAMPLES_SHM,  # noqa: F821
        )

    def run(self) -> None:
        """Start the process."""
        ## Create shared memories
        # NOTE: This goes here because it needs to be done in the child process!
        # This is the first piece of code of the worker that live in the child
        # process, the `__init__()` function does not.
        self._link_shms()
        ## Task loop
        task: int
        for task in iter(self.task_queue.get, None):
            self.process_task(task)
            if self.auto_terminate:
                break
        ## Un-register shared memories
        # NOTE: Bug https://bugs.python.org/issue39959#msg364351
        self._unregister_shms()
        ## Put the closing task's results in the result queue
        if not self.auto_terminate:
            self.result_queue.put([-1, 0, 0, ""])


def create_new_worker(
    client_fn: Callable[[int], VirtualLLMClient],
    task_queue: QueueType,
    result_queue: QueueType,
    node_manager_uuid: str,
    parameters: NDArrays,
    train_metrics: Dict[str, Scalar],
) -> Tuple[Worker, str, NDArrays, SharedMemory, np.ndarray, SharedMemory,]:
    """Create a new Worker."""
    # Generate the Worker's UUID
    worker_uuid = node_manager_uuid + str(uuid.uuid4())
    # Create the Shared Memory objects
    w_parameters, w_parameters_shm = get_parameters_shm(
        create=True,
        parameters=parameters,
        name=worker_uuid + POLLEN_PARAMETERS_SHM,  # noqa: F821
    )
    # Number of samples shared memory
    w_num_samples, w_num_samples_shm = get_num_samples_shm(
        create=True,
        name=worker_uuid + POLLEN_N_SAMPLES_SHM,  # noqa: F821
    )
    # Create the Worker object
    worker = Worker(
        client_fn=client_fn,
        worker_uuid=worker_uuid,
        task_queue=task_queue,
        result_queue=result_queue,
        node_manager_uuid=node_manager_uuid,
        parameters=parameters,
        train_metrics=train_metrics,
    )
    # Return the Worker object, its UUID, and the shared objects
    return (
        worker,
        worker_uuid,
        w_parameters,
        w_parameters_shm,
        w_num_samples,
        w_num_samples_shm,
    )
