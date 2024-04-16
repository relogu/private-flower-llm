"""A highly efficient worker for Pollen."""

import multiprocessing
from pathlib import Path
import pickle
import time
from collections.abc import Callable
from logging import DEBUG, INFO
from multiprocessing.queues import Queue as QueueType
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.process import AuthenticationString  # type: ignore[attr-defined]
from typing import Any
import uuid
from torch import device
from torch.nn import Module
from torch.distributed import (
    init_process_group,
    dist_backend,
    barrier,
)

import cloudpickle
import torch
import transformers
from flwr.common.logger import log
from multiprocess import set_start_method, Process  # type: ignore[reportAttributeAccessIssue]

from pollen_worker.models.pfl_cnns import MultiLabelCNN
from pollen_worker.pollen_utils import (
    POLLEN_CONFIG_SHM,
    POLLEN_PARAMETERS_SHM,
    WorkerResult,
    allocate_shm,
    get_model,
    get_optimizer,
    remove_shm_from_resource_tracker,
    write_to_fit_result_shm,
)
from pollen_worker.horovod_utils import (
    ClientDataset,
    ClientDatasetOpenImageDataloader,
    ClientDatasetOpenImageNoDataloader,
    FLAIRFederatedDataset,
    ClientDatasetShakespeareDataloader,
    FederatedDataset,
    FederatedDatasetOpenImage,
    all_reduce,
    classification_training_loop,
    flair_training_loop,
    get_model_difference,
    get_ndarrays_from_model,
    get_parameters,
    get_variable_map,
    make_cifar10_iid_datasets,
    make_openimage_natural_partition,
    make_shakespeare_natural_partition,
    set_model_parameters_from_ndarrays,
    set_parameters,
    make_flair_federated_dataset,
)

from pollen_worker.virtual_client import VirtualClient

pickle.Pickler = cloudpickle.Pickler  # type: ignore[misc]
transformers.logging.set_verbosity_error()
set_start_method("spawn", force=True)


class Worker(Process):
    """Worker Process child of the NodeManager."""

    def __init__(
        self,
        client_fn: Callable[[int], VirtualClient],
        run_uuid: str,
        concurrency: int,
        dataset_name: str,
        task_queues: dict[str, QueueType],
        result_queue: QueueType,
        auth_key: bytes,
        local_rank: int,
        world_size: int,
        workers_system_metrics_queue: QueueType | None = None,
        worker_id: str | None = None,
    ) -> None:
        super().__init__()
        self.client_fn: Callable[[int], VirtualClient] = client_fn
        self.task_queues = task_queues
        self.result_queue = result_queue
        self.workers_system_metrics_queue: QueueType | None = (
            workers_system_metrics_queue if workers_system_metrics_queue else None
        )
        self.run_uuid = run_uuid
        self.concurrency = concurrency
        self.dataset_name = dataset_name
        self.auth_key = auth_key
        self.worker_id = worker_id if worker_id else str(uuid.uuid4())
        self.local_rank = local_rank
        self.world_size = world_size
        # These attributes will be set after having initialized PyTorch Distributed
        self.master_address: str
        self.master_port: int
        self.device: device
        self.federated_dataset: (
            FederatedDataset
            | FLAIRFederatedDataset
            | FederatedDataset
            | FederatedDatasetOpenImage
        )
        self.worker_global_model: Module
        self.client_model: Module | MultiLabelCNN
        self.buffer: dict[str, torch.Tensor] | None = None
        self.cache: dict[str, torch.Tensor] | None = None
        self.worker_global_model_variable_map: dict[str, torch.Tensor]
        self.worker_central_optimizer: torch.optim.Optimizer

    def __getstate__(self) -> dict[str, Any]:
        """Return the state of the object.

        It is called when pickling - this hack allows subprocesses to be spawned without
        the AuthenticationString raising an error.
        """
        state = self.__dict__.copy()
        conf = state["_config"]
        if "authkey" in conf:
            conf["authkey"] = (
                self.auth_key if self.auth_key is not None else bytes(conf["authkey"])
            )
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the object.

        It is used for unpickling. It couples with the __getstate__ method to complete
        the hack that allows subprocesses to be spawned without the AuthenticationString
        raising an error.
        """
        state["_config"]["authkey"] = AuthenticationString(state["_config"]["authkey"])
        self.__dict__.update(state)

    def process_task_pytorch_distributed(self, client_ids: list[int]) -> None:
        """Execute the training of the passed client ids using Horovod."""
        worker_system_metrics: list[tuple[str, float]] = []
        barrier()
        init_time = time.time()
        # Loads a dict from the shared memory buffer
        config = pickle.loads(self.config_shm.buf)
        # Initialise number of samples and loss to be then reduced
        aggregated_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        aggregated_accuracy = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        aggregated_num_samples = torch.tensor(
            0, dtype=torch.float32, device=self.device
        )
        # Initialise the model parameters
        # NOTE: Method 1 is quicker for CIFAR10
        # Method 1 -- All workers read the shared memory concurrently
        set_model_parameters_from_ndarrays(self.round_params, self.client_model)
        # Method 2 -- Just rank zero reads and then it broadcasts
        # if self.local_rank == 0:
        #     set_model_parameters_from_ndarrays(self.round_params, self.client_model)
        # with torch.no_grad():
        #     for _, variable in self.client_model.named_parameters():
        #         variable_copy = variable.detach()
        #         broadcast(variable_copy, 0)
        #         variable.copy_(variable_copy)
        set_parameters(
            get_variable_map(self.client_model),
            get_variable_map(self.worker_global_model),
        )
        self.worker_global_model_variable_map = get_variable_map(
            self.worker_global_model
        )
        self.worker_central_optimizer = torch.optim.SGD(
            self.worker_global_model.parameters(), lr=1.0
        )
        barrier()
        elapsed_time = time.time() - init_time
        worker_system_metrics.append(("workers/fit_init_time", elapsed_time))
        barrier()
        train_time = time.time()
        results: list[WorkerResult] = []
        if self.dataset_name == "flair":
            train_loop = flair_training_loop
            local_optimizer = get_optimizer(name="flair", model=self.client_model)
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            train_loop = classification_training_loop
            local_optimizer = get_optimizer(
                name=self.dataset_name, model=self.client_model
            )
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        for client_dataset in self.federated_dataset.get_cohort(client_ids):
            # Take the timestamp before training a single client
            start_time = time.time_ns()

            cumulative_loss, cumulative_accuracy, cumulative_num_samples = train_loop(
                self.client_model,
                local_optimizer,
                criterion,
                self.device,
                client_dataset,
                config["batch_size"],
                self.dataset_name,
                1,
            )
            aggregated_loss += cumulative_loss * cumulative_num_samples
            aggregated_accuracy += cumulative_accuracy * cumulative_num_samples
            aggregated_num_samples += cumulative_num_samples

            # Get the pseudo-gradients
            deltas = get_model_difference(
                variable_map=get_variable_map(self.client_model),
                other_variable_map=self.worker_global_model_variable_map,
                cache=self.cache,
            )

            # Update the pseudo-gradients buffer
            if self.buffer is None:
                self.buffer = get_parameters(deltas)
                # Scale the pseudo-gradients on the number of samples trained
                for tensor_value in self.buffer.values():
                    tensor_value.mul_(cumulative_num_samples)
            else:
                for tensor_name, tensor_value in get_parameters(deltas).items():
                    # Scale the pseudo-gradients on the number of samples trained
                    tensor_value.mul_(cumulative_num_samples)
                    # Add the pseudo-gradients to the buffer
                    self.buffer[tensor_name].add_(tensor_value)
            set_parameters(
                self.worker_global_model_variable_map,
                get_variable_map(self.client_model),
            )
            # Take the timestamp after the task is done
            end_time = time.time_ns()
            results.append(
                WorkerResult(
                    int(cumulative_num_samples.cpu().item()),
                    (end_time - start_time) * 1e-9,
                    str(self.device),
                )
            )
        barrier()
        elapsed_time = time.time() - train_time
        worker_system_metrics.append(("workers/fit_train_time", elapsed_time))
        # Write to shared memory if this is the last client
        if self.buffer is not None:
            barrier()
            aggregation_time = time.time()
            # log(
            #     INFO,
            #     "Local Rank: %s, Averaging %s clients.",
            #     self.local_rank,
            #     j,
            # )
            # All reduce the updates
            reduced_buffer = all_reduce(
                self.world_size, list(self.buffer.values()), average=False
            )
            reduced_metrics = all_reduce(
                self.world_size,
                [aggregated_loss, aggregated_accuracy, aggregated_num_samples],
                average=False,
            )
            # Post-process the results of the reduce operation on rank 0 only
            if self.local_rank == 0:
                # Scale the metrics on the number of samples trained on this node
                reduced_metrics[0].div_(reduced_metrics[2])
                reduced_metrics[1].div_(reduced_metrics[2])
                log(
                    INFO,
                    "Local Rank: %s, W. Avg. Loss: %s,"
                    " W. Avg. Accuracy: %s, Tot. N. Samples: %s",
                    self.local_rank,
                    float(reduced_metrics[0].cpu().item()),
                    float(reduced_metrics[1].cpu().item()),
                    int(reduced_metrics[2].cpu().item()),
                )
                # Update the global model applying the updates with the optimizer
                self.worker_central_optimizer.zero_grad()
                for variable_name, difference in dict(
                    zip(self.buffer.keys(), reduced_buffer, strict=False)
                ).items():
                    # Scale the update for the number of samples trained on this node
                    difference.div_(reduced_metrics[2])
                    if (
                        self.worker_global_model_variable_map[variable_name].grad
                        is None
                    ):
                        self.worker_global_model_variable_map[variable_name].grad = (
                            torch.zeros_like(
                                self.worker_global_model_variable_map[variable_name],
                                device=self.device,
                            )
                        )
                    # Interpret the model updates as gradients.
                    self.worker_global_model_variable_map[variable_name].grad.data.copy_(  # type: ignore[union-attr]
                        -1 * difference
                    )
                # Apply the update
                self.worker_central_optimizer.step()
                # Write buffer to shared memory
                write_to_fit_result_shm(
                    self.worker_params,
                    self.worker_num_samples,
                    self.worker_train_loss,
                    self.worker_train_acc,
                    get_ndarrays_from_model(net=self.worker_global_model),  # type: ignore[reportArgumentType]
                    int(reduced_metrics[2].cpu().item()),
                    float(reduced_metrics[0].cpu().item()),
                    float(reduced_metrics[1].cpu().item()),
                )
            barrier()
            elapsed_time = time.time() - aggregation_time
            worker_system_metrics.append(("workers/aggregation_time", elapsed_time))
            # Add systems metrics to the queue
            if self.workers_system_metrics_queue is not None:
                for metric in worker_system_metrics:
                    self.workers_system_metrics_queue.put(metric)
            # Put results in the result queue
            for result in results:
                self.result_queue.put(result)
            # log(
            #     DEBUG,
            #     "Node partial aggregation time measure at rank %s: %s",
            #     self.local_rank,
            #     elapsed_time,
            # )
            # Empty the buffer
            if self.buffer is not None:
                for tensor_name, tensor_value in self.buffer.items():
                    self.buffer[tensor_name] = torch.empty(
                        tensor_value.shape,
                        dtype=torch.float32,
                        device=tensor_value.device,
                    )
            # gc.collect()
            # torch.cuda.empty_cache()

    def run(self) -> None:
        """Start the process."""
        # Set the authentication key. Necessary for accessing the queues.
        multiprocessing.current_process().authkey = self.auth_key
        # Monkey-patch resource tracker to avoid tracking shared memory
        remove_shm_from_resource_tracker()
        # Initializes the distributed backend which will take care of synchronizing
        # nodes/GPUs
        backend = dist_backend.NCCL if torch.cuda.is_available() else dist_backend.GLOO
        # NOTE: NCCL can only use one process per GPU, so we use GLOO if there are more
        backend = (
            dist_backend.GLOO
            if self.world_size > torch.cuda.device_count()
            else backend
        )
        # We assume that the master is always in the first position.
        init_method = f"tcp://{self.master_address}:{self.master_port}"
        init_process_group(
            backend=backend,
            init_method=init_method,
            rank=self.local_rank,
            world_size=self.world_size,
        )
        aggregation_time = time.time()
        # Set the device and the task queue
        gpu_id = self.local_rank % torch.cuda.device_count()
        self.device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        log(
            INFO,
            "Worker %s uses GPU %i, device %s",
            self.worker_id,
            gpu_id,
            self.device,
        )
        # Set the task queue
        task_queue = self.task_queues[str(self.device)]
        log(
            INFO,
            "Worker %s selected %s from %s.",
            self.worker_id,
            task_queue,
            self.task_queues,
        )
        if self.dataset_name == "cifar10":
            self.federated_dataset = make_cifar10_iid_datasets(
                world_size=self.concurrency,
                local_rank=self.local_rank,
                device=str(self.device),
            )
        elif self.dataset_name == "flair":
            self.federated_dataset = make_flair_federated_dataset(
                hdf5_path=Path("/datasets/flair/flair_federated.hdf5"),
                partition="train",
                use_fine_grained_labels=True,
                max_num_user_images=512,
                world_size=self.concurrency,
                local_rank=self.local_rank,
                device=str(self.device),
            )
        elif self.dataset_name == "openimage":
            # Openimage with dataloader with num_workers 1
            self.federated_dataset = make_openimage_natural_partition(
                world_size=self.concurrency,
                local_rank=self.local_rank,
                n_workers=1,
                client_dataset_type=ClientDatasetOpenImageDataloader,
                device=str(self.device),
            )
            # Openimage with list-based pfl-style loading
            self.federated_dataset = make_openimage_natural_partition(
                world_size=self.concurrency,
                local_rank=self.local_rank,
                client_dataset_type=ClientDatasetOpenImageNoDataloader,
                device=str(self.device),
            )
        elif "shakespeare" in self.dataset_name:
            # Shakespeare with dataloader with num_workers 0
            self.federated_dataset = make_shakespeare_natural_partition(
                world_size=self.concurrency,
                local_rank=self.local_rank,
                n_workers=0,
                client_dataset_type=ClientDatasetShakespeareDataloader,
                device=str(self.device),
            )

            # Shakespeare with list-based pfl-style loading
            self.federated_dataset = make_shakespeare_natural_partition(
                world_size=self.concurrency,
                local_rank=self.local_rank,
                client_dataset_type=ClientDataset,
                device=str(self.device),
            )

        # Create global model
        # if self.local_rank == 0:
        self.worker_global_model = get_model(self.dataset_name)
        self.worker_global_model = self.worker_global_model.to(self.device)
        # Create client model
        self.client_model = get_model(self.dataset_name)
        self.client_model = self.client_model.to(self.device)
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
            parameters=self.client_fn(0).get_parameters({}),  # type: ignore[reportArgumentType]
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
            parameters=self.client_fn(0).get_parameters({}),  # type: ignore[reportArgumentType]
            name=self.worker_id,
            create=True,
        )
        elapsed_time = time.time() - aggregation_time
        if self.workers_system_metrics_queue is not None:
            self.workers_system_metrics_queue.put(("workers/init_time", elapsed_time))
        # Task loop
        task: list[int]
        for task in iter(task_queue.get, None):
            log(INFO, "Worker %s received task %s.", self.worker_id, task)
            self.process_task_pytorch_distributed(task)
        # Put the closing task's results in the result queue
        self.result_queue.put(
            WorkerResult(
                -1,
                0.0,
                "",
            )
        )

    def __del__(self) -> None:
        """Implement the deletion of the Worker."""
        log(DEBUG, "Closing Worker %s...", self.worker_id)
        # Free shared memories
        worker_shm = SharedMemory(name=self.worker_id)
        worker_shm.close()
        worker_shm.unlink()
        log(DEBUG, "Shared memories closed")
        # Empty worker's system metrics queue
        if self.workers_system_metrics_queue is not None:
            while not self.workers_system_metrics_queue.empty():
                self.workers_system_metrics_queue.get()
