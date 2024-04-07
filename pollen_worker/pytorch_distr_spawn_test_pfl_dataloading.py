"""PyTorch CIFAR-10 distributed training example.

From: https://leimao.github.io/blog/PyTorch-Distributed-Training/
Also seen:
 - https://pytorch.org/docs/stable/elastic/run.html#launcher-api
"""

import time
import torch
from torch.distributed import init_process_group, dist_backend
from torch import nn
from torch import optim
import torch.multiprocessing as mp

from flwr.common import log
from logging import INFO

from pollen_worker.horovod_utils import (
    all_reduce,
    get_model_difference,
    get_parameters,
    get_variable_map,
    make_cifar10_iid_datasets,
    prepare_batch,
    set_parameters,
    set_random_seeds,
)
from pollen_worker.pollen_utils import get_model


def main(
    local_rank: int,
    world_size: int,
    master_addr: str = "localhost",
    master_port: int = 51550,
) -> None:
    """Execute the main function."""
    num_rounds = 10000
    n_clients_per_round = 50
    batch_size = 10
    n_samples_per_client = 50
    learning_rate = 0.1
    random_seed = 51550

    # local_rank = int(os.environ["LOCAL_RANK"])
    # world_size = int(os.environ["WORLD_SIZE"])
    # master_addr = os.environ["MASTER_ADDR"]
    # master_port = os.environ["MASTER_PORT"]

    # We need to use seeds to make sure that the models initialized in different
    # processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of synchronizing
    # nodes/GPUs
    backend = dist_backend.NCCL if torch.cuda.is_available() else dist_backend.GLOO
    backend = dist_backend.GLOO if world_size > torch.cuda.device_count() else backend
    # We assume that the master is always in the first position.
    init_method = f"tcp://{master_addr}:{master_port}"
    init_process_group(
        backend=backend, init_method=init_method, rank=local_rank, world_size=world_size
    )

    # Encapsulate the model on the GPU assigned to the current process
    global_model = get_model("cifar10")
    global_model_variable_map = get_variable_map(global_model)
    central_optimizer = optim.SGD(global_model.parameters(), lr=1.0)
    model = get_model("cifar10")

    gpu_id = local_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)
    log(INFO, "local rank %i use GPU: %i", local_rank, gpu_id)
    model = model.to(device)

    # Data loading
    federated_dataset = make_cifar10_iid_datasets(
        user_dataset_len_sampler=lambda: n_samples_per_client,
        world_size=world_size,
        local_rank=local_rank,
    )

    buffer: dict[str, torch.Tensor] | None = None
    cache: dict[str, torch.Tensor] | None = None
    # Loop over the dataset multiple times
    for federated_round in range(num_rounds):
        start_time = time.time()

        log(INFO, "Local Rank: %s, Round: %s", local_rank, federated_round)

        j = 0
        for client_dataset in federated_dataset.get_cohort(n_clients_per_round):
            # log(
            #     INFO,
            #     "Local Rank: %s, Round: %s, Training client: %s",
            #     local_rank,
            #     federated_round,
            #     client_id,
            # )

            initial_model_variable_map = get_parameters(get_variable_map(model))
            optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
            )
            criterion = nn.CrossEntropyLoss(reduction="mean")

            model.train()

            cumulative_loss: torch.Tensor = torch.tensor(0.0, device=device)
            cumulative_acc: torch.Tensor = torch.tensor(0.0, device=device)
            cumulative_n_samples: torch.Tensor = torch.tensor(0.0, device=device)
            for data in client_dataset.iter(batch_size):
                prepared_batch = prepare_batch(data)
                inputs, labels = prepared_batch[0].to(device), prepared_batch[1].to(
                    device
                )
                if labels.dim() > 1:
                    labels = labels.squeeze()
                optimizer.zero_grad()
                outputs: torch.Tensor = model(inputs)
                loss: torch.Tensor = criterion(outputs, labels.long())
                cumulative_loss += loss.item()
                cumulative_acc += (outputs.argmax(-1) == labels.long()).sum().item()
                cumulative_n_samples += labels.size(0)
                loss.backward()
                optimizer.step()

            if j == 0:
                log(
                    INFO,
                    "Local Rank: %s, Round: %s, Loss: %s, Acc: %s, N Samples: %s",
                    local_rank,
                    federated_round,
                    cumulative_loss.cpu().item(),
                    cumulative_acc.cpu().item(),
                    cumulative_n_samples.cpu().item(),
                )

            deltas = get_model_difference(
                variable_map=get_variable_map(model),
                other_variable_map=initial_model_variable_map,
                cache=cache,
            )

            if buffer is None:
                buffer = get_parameters(deltas)
            else:
                for tensor_name, tensor_value in get_parameters(deltas).items():
                    buffer[tensor_name].add_(tensor_value)
            set_parameters(initial_model_variable_map, get_variable_map(model))
            j += 1

        if buffer is not None:
            log(
                INFO,
                "Local Rank: %s, Round: %s, Averaging %s clients.",
                local_rank,
                federated_round,
                j,
            )
            # Scale the cumulative update on the number of clients trained on this rank
            for tensor_value in buffer.values():
                tensor_value.div_(j)
            # All reduce the updates
            reduced_buffer = all_reduce(world_size, list(buffer.values()), average=True)
            # Update the global model applying the updates with the optimizer
            central_optimizer.zero_grad()
            for variable_name, difference in dict(
                zip(buffer.keys(), reduced_buffer, strict=False)
            ).items():
                if global_model_variable_map[variable_name].grad is None:
                    global_model_variable_map[variable_name].grad = torch.zeros_like(
                        global_model_variable_map[variable_name]
                    )
                # Interpret the model updates as gradients.
                global_model_variable_map[
                    variable_name
                ].grad.data.copy_(  # type: ignore[union-attr]
                    -1 * difference
                )
            central_optimizer.step()
            # Set the new model parameters to the local model
            set_parameters(
                get_parameters(global_model_variable_map),
                get_variable_map(model),
            )
            # Reset the buffer
            for tensor_name, tensor_value in buffer.items():
                buffer[tensor_name] = torch.empty(
                    tensor_value.shape, dtype=torch.float32, device=tensor_value.device
                )

        if local_rank == 0:
            log(
                INFO,
                "Round: %s, Done in %s seconds",
                federated_round,
                time.time() - start_time,
            )


if __name__ == "__main__":
    """
    How to run this script:
        ```console
        $ poetry shell
        $ python pollen_worker/pytorch_distr_spawn_test_pfl_dataloading.py
        ```
    """
    mp.spawn(main, args=(4,), nprocs=4, join=True)
