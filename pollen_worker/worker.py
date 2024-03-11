"""A highly efficient worker for Pollen."""

import gc
import os
import pickle
import time
from collections.abc import Callable
from logging import ERROR
from multiprocessing import resource_tracker
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory

import cloudpickle
import multiprocess as mp
import pynvml
import torch
import transformers
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from flwr.common.logger import log
from multiprocess import set_start_method

from pollen_worker.pollen_utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_PARAMETERS_SHM,
    allocate_shm,
    write_to_fit_result_shm,
)
from pollen_worker.utils import partially_aggregate_with_metrics

pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]
transformers.logging.set_verbosity_error()
set_start_method("spawn", force=True)


class Worker(mp.Process):
    """Worker Process child of the NodeManager."""

    def __init__(
        self,
        client_fn: Callable[[int], NumPyClient],
        device: str,
        worker_id: str,
        task_queue: QueueType,
        result_queue: QueueType,
        run_uuid: str,
        concurrency: int,
    ) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.run_uuid = run_uuid
        self.current_round: int = 0
        self.concurrency = concurrency

    def process_task(self, client_id: int) -> None:
        """Process the received task."""
        # Take the timestamp before training a single client
        start_time = time.time_ns()
        # Loads a dict from the shared memory buffer
        config = pickle.loads(self.config_shm.buf)
        config["device"] = self.device

        # Load client
        tmp_client = self.client_fn(client_id)

        done = False
        fit_trained_weights: NDArrays | None = None
        fit_num_samples: int | None = None
        train_metrics: dict[str, Scalar] | None = None
        while not done:
            try:
                # Call fit on shared parameters
                fit_trained_weights, fit_num_samples, train_metrics = tmp_client.fit(
                    self.round_params, config
                )
                done = True
            except Exception as e:
                log(
                    ERROR,
                    "Worker %s failed in training client %s with exception %s."
                    " Retrying...",
                    self.worker_id,
                    client_id,
                    e,
                )
        if (
            fit_trained_weights is None
            or fit_num_samples is None
            or train_metrics is None
        ):
            raise ValueError(
                f"Worker {self.worker_id} failed in training client {client_id}."
                " fit_trained_weights, fit_num_samples, or train_metrics is None."
            )
        # If new round, then copy result to shared memory directly
        if config["server_round"] > self.current_round:
            self.current_round = config["server_round"]
            write_to_fit_result_shm(
                self.worker_params,
                self.worker_num_samples,
                self.worker_train_loss,
                self.worker_train_acc,
                fit_trained_weights,
                fit_num_samples,
                float(train_metrics["train_loss"]),
                float(train_metrics["accuracy"]),
            )
        # Partially aggregating fit results
        else:
            (
                tmp_part_agg_params,
                tmp_part_agg_num_samples,
                tmp_part_agg_loss,
                tmp_part_agg_acc,
            ) = partially_aggregate_with_metrics(
                (
                    self.worker_params,
                    self.worker_num_samples[0],
                    self.worker_train_loss[0],
                    self.worker_train_acc[0],
                ),
                (
                    fit_trained_weights,
                    fit_num_samples,
                    float(train_metrics["train_loss"]),
                    float(train_metrics["accuracy"]),
                ),
            )
            write_to_fit_result_shm(
                self.worker_params,
                self.worker_num_samples,
                self.worker_train_loss,
                self.worker_train_acc,
                tmp_part_agg_params,
                tmp_part_agg_num_samples,
                tmp_part_agg_loss,
                tmp_part_agg_acc,
            )
        # Take the timestamp after the task is done
        end_time = time.time_ns()
        self.result_queue.put([int(client_id), start_time, end_time, self.device])
        # NOTE: PyTorch's memory management works bad with multiprocessing. In our case,
        # it might happen that each process eagerly allocates more MBs of memory on the
        # same VRAM at the same w/o cleaning the cache because each of them thinks that
        # it is the only one using the GPU. We need to clean the cache manually to
        # prevent this, i.e. call `torch.cuda.empty_cache()`. When to call it is a
        # trade-off between performance and memory usage because cleaning the cache
        # is time expensive (for Reddit it costs ~15 seconds, quick took ~333s slow
        # took ~348s in 10 rounds, 100 clients/round, 1 A40). We call it after each
        # client, but it might be better to call it after each round.
        for dev_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if os.getpid() == proc.pid:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if proc.usedGpuMemory > mem.total / self.concurrency:
                        torch.cuda.empty_cache()
                        gc.collect()

    def run(self) -> None:
        """Start the process."""
        # Allocate shared memories.
        # NOTE: This goes here because it needs to be done in the child process!
        # This is the first piece of code of the worker that live in the child
        # process, the `__init__()` function does not.
        # NOTE: This is the NodeManager's shared memory for the fit config
        # dictionary. Workers should only read this. NodeManager should only
        # write this.
        self.config_shm = SharedMemory(name=self.run_uuid + POLLEN_CONFIG_SHM)
        # NOTE: This is the NodeManager's shared memory for the fit results.
        # Workers should only read this. NodeManager should only write this.
        (
            self.round_params,
            self.round_num_samples,
            self.round_train_loss,
            self.round_train_acc,
            self.round_shm,
        ) = allocate_shm(
            parameters=self.client_fn(0).get_parameters({}),
            name=self.run_uuid + POLLEN_PARAMETERS_SHM,
        )
        # NOTE: This is the Worker's shared memory for the fit results.
        # NodeManager should only read this. Worker should only write this.
        (
            self.worker_params,
            self.worker_num_samples,
            self.worker_train_loss,
            self.worker_train_acc,
            self.worker_shm,
        ) = allocate_shm(
            parameters=self.client_fn(0).get_parameters({}),
            name=self.worker_id,
        )
        # NOTE: This is for controlling the GPU memory allocation
        pynvml.nvmlInit()
        # Task loop
        task: int
        for task in iter(self.task_queue.get, None):
            self.process_task(task)
        # Put the closing task's results in the result queue
        self.result_queue.put([-1, 0, 0, ""])
        # Un-register shared memories
        # NOTE: Bug https://bugs.python.org/issue39959#msg364351
        resource_tracker.unregister(
            self.config_shm._name,  # type: ignore[attr-defined]
            "shared_memory",
        )
        resource_tracker.unregister(
            self.round_shm._name,  # type: ignore[attr-defined]
            "shared_memory",
        )
        resource_tracker.unregister(
            SharedMemory(name=self.worker_id)._name,  # type: ignore[attr-defined]
            "shared_memory",
        )
