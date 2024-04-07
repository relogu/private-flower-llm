#!/usr/bin/env python
"""PyTorch CIFAR-10 distributed training example.

From: https://leimao.github.io/blog/PyTorch-Distributed-Training/
Also seen:
 - https://pytorch.org/docs/stable/elastic/run.html#launcher-api
"""

import time
import torch
from torch import nn
from torch import optim
import horovod.torch as hvd

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


def main() -> None:
    """Execute the main function."""
    num_rounds = 10000
    n_clients_per_round = 50
    batch_size = 10
    n_samples_per_client = 50
    learning_rate = 0.1
    random_seed = 51550

    # We need to use seeds to make sure that the models initialized in different
    # processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of synchronizing
    # nodes/GPUs
    hvd.init()
    log(
        INFO,
        "local_rank=%i local_size=%i rank=%i size=%i",
        hvd.local_rank(),
        hvd.local_size(),
        hvd.rank(),
        hvd.size(),
    )
    if hvd.size() == 1:
        raise RuntimeError(
            "You are running Horovod backend but number of "
            "worker and processes are 1. If you intend to only run on a "
            "single worker and process, don't use Horovod."
            "If you intend to run multiple processes/workers, then your "
            "run command was incorrect"
        )
    if torch.cuda.is_available():
        gpu_id = hvd.local_rank() % torch.cuda.device_count()
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(gpu_id)
        log(INFO, "local rank %i use GPU: %i", hvd.local_rank(), gpu_id)

    # Encapsulate the model on the GPU assigned to the current process
    model = get_model("cifar10")
    model = model.to(device)

    # Data loading
    federated_dataset = make_cifar10_iid_datasets(
        user_dataset_len_sampler=lambda: n_samples_per_client,
        world_size=hvd.size(),
        local_rank=hvd.local_rank(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-5,
    )

    buffer: dict[str, torch.Tensor] | None = None
    cache: dict[str, torch.Tensor] | None = None
    # Loop over the dataset multiple times
    for federated_round in range(num_rounds):
        start_time = time.time()

        log(INFO, "Local Rank: %s, Round: %s", hvd.local_rank(), federated_round)

        j = 0
        for client_dataset in federated_dataset.get_cohort(n_clients_per_round):
            # log(
            #     INFO,
            #     "Local Rank: %s, Round: %s, Training client: %s",
            #     hvd.local_rank(),
            #     federated_round,
            #     client_id,
            # )

            initial_model_variable_map = get_parameters(get_variable_map(model))

            model.train()

            for data in client_dataset.iter(batch_size):
                prepared_batch = prepare_batch(data)
                inputs, labels = prepared_batch[0].to(device), prepared_batch[1].to(
                    device
                )
                if labels.dim() > 1:
                    labels = labels.squeeze()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                loss.backward()
                optimizer.step()

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
                hvd.local_rank(),
                federated_round,
                j,
            )
            for tensor_value in buffer.values():
                tensor_value.div_(j)
            reduced_buffer = all_reduce(hvd, list(buffer.values()), average=True)
            set_parameters(
                dict(zip(buffer.keys(), reduced_buffer, strict=False)),
                get_variable_map(model),
            )
            for tensor_name, tensor_value in buffer.items():
                buffer[tensor_name] = torch.empty(
                    tensor_value.shape, dtype=torch.float32, device=tensor_value.device
                )
        if hvd.local_rank() == 0:
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
        $ horovodrun -cb # Check basic Horovod functionality
        $ horovodrun --gloo -np 2 -H localhost:2 \
            python pollen_worker/horovod_test_pfl_dataloading.py
        $ horovodrun --gloo -np 4 -H localhost:4 \
            python pollen_worker/horovod_test_pfl_dataloading.py
        ```
    """
    main()
