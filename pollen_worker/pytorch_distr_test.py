"""PyTorch CIFAR-10 distributed training example.

From: https://leimao.github.io/blog/PyTorch-Distributed-Training/
Also seeen:
 - https://pytorch.org/docs/stable/elastic/run.html#launcher-api
"""

import os
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group, dist_backend
from torch import nn
from torch import optim

import torchvision
from torchvision import transforms
from flwr.common import log
from logging import INFO
import argparse
import random
import numpy as np

from pollen_worker.datasets.federated_dataset import FederatedDataset
from pollen_worker.pollen_utils import get_client_ds_fn, get_model


def set_random_seeds(random_seed: int = 0) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def evaluate(model: torch.nn.Module, device: str, test_loader: DataLoader) -> float:
    """Evaluate the model on the test set."""
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total

    return accuracy


def get_variable_map(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Get the variable map from the model."""
    return {
        name: variable
        for name, variable in model.named_parameters()
        if variable.requires_grad
    }


def set_parameters(
    source_variable_map: dict[str, torch.Tensor],
    destination_variable_map: dict[str, torch.Tensor],
) -> None:
    """Set the parameters of the destination variables.

    It uses the values of the source tensors.
    """
    for variable_name, variable in destination_variable_map.items():
        new_value = source_variable_map[variable_name]
        variable.data.copy_(new_value)


def get_parameters(
    variable_map: dict[str, torch.Tensor],
    placeholder_variable_map: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Get the parameters from the variable map.

    If placeholders is not None, set the parameters of the placeholders to the values
    of the variable map.
    """
    if placeholder_variable_map is None:
        return {k: v.detach().clone() for k, v in variable_map.items()}
    else:
        set_parameters(variable_map, placeholder_variable_map)
        return placeholder_variable_map


@torch.no_grad()
def get_model_difference(
    variable_map: dict[str, torch.Tensor],
    other_variable_map: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return the difference between the model and other parameters."""
    model_diff: dict[str, torch.Tensor] = {}
    cache = model_diff if cache is None else cache
    for variable_name, variable in variable_map.items():
        if variable_name not in cache:
            # Up to here, we've used float16 or float32.
            # From here on, we always use float32 for numerical stability in
            # norm clipping and normalization.
            cache[variable_name] = torch.empty(
                variable.shape, dtype=torch.float32, device=variable.device
            )
        model_diff[variable_name] = (
            cache[variable_name]
            .data.copy_(variable)
            .sub_(other_variable_map[variable_name])
        )
    return model_diff, cache


def main() -> None:
    """Execute the main function."""
    num_epochs_default = 10000
    n_clients_per_round = 50
    batch_size_default = 10  # 256  # 1024
    n_samples_per_client = 50
    learning_rate_default = 0.1
    random_seed_default = 0
    model_dir_default = "saved_models"
    model_filename_default = "resnet_distributed.pth"

    # Each process runs on 1 GPU device specified by the local_rank argument.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        help="Local rank. Necessary for using the torch.distributed.launch utility.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs.",
        default=num_epochs_default,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size for one process.",
        default=batch_size_default,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate.",
        default=learning_rate_default,
    )
    parser.add_argument(
        "--random_seed", type=int, help="Random seed.", default=random_seed_default
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory for saving models.",
        default=model_dir_default,
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        help="Model filename.",
        default=model_filename_default,
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved checkpoint."
    )
    argv = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    # local_rank = argv.local_rank
    num_epochs = argv.num_epochs
    batch_size = argv.batch_size
    learning_rate = argv.learning_rate
    random_seed = argv.random_seed
    model_dir = argv.model_dir
    model_filename = argv.model_filename
    resume = argv.resume

    # Create directories outside the PyTorch program
    # Do not create directory here because it is not multiprocess safe
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    """

    model_filepath = Path(model_dir) / model_filename

    # We need to use seeds to make sure that the models initialized in different
    # processes are the same
    set_random_seeds(random_seed=random_seed)

    # Initializes the distributed backend which will take care of synchronizing
    # nodes/GPUs
    backend = dist_backend.NCCL if torch.cuda.is_available() else dist_backend.GLOO
    # We assume that the master is always in the first position.
    init_method = f"tcp://{master_addr}:{master_port}"
    init_process_group(
        backend=backend, init_method=init_method, rank=local_rank, world_size=world_size
    )

    # Encapsulate the model on the GPU assigned to the current process
    # model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = get_model("cifar10")

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    ddp_model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # We only save the model who uses device "cuda:0"
    # To resume, the device for the saved model would also be "cuda:0"
    if resume is True:
        map_location = {"cuda:0": f"cuda:{local_rank}"}
        ddp_model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data should be prefetched
    # Download should be set to be False, because it is not multiprocess safe
    # train_set = torchvision.datasets.CIFAR10(
    #     root="/datasets/cifar10/raw_data",
    #     train=True,
    #     download=False,
    #     transform=transform,
    # )
    test_set = torchvision.datasets.CIFAR10(
        root="/datasets/cifar10/raw_data",
        train=False,
        download=False,
        transform=transform,
    )

    def dataset_generator(cid: int) -> Dataset[Any]:
        """Return the dataset for the client."""
        return get_client_ds_fn(name="cifar10")(cid)[0]

    federated_dataset = FederatedDataset(
        dataset_generator=dataset_generator,
        list_of_clients=[],
    )

    # # Restricts data loading to a subset of the dataset
    # # exclusive to the current process
    # train_sampler = DistributedSampler(dataset=train_set)

    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=batch_size,
    #     sampler=train_sampler,
    #     num_workers=0,  # , prefetch_factor=50
    # )
    # Test loader does not have to follow distributed sampling strategy
    test_loader = DataLoader(
        dataset=test_set, batch_size=128, shuffle=False, num_workers=8
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        ddp_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5
    )

    # buffer: list[dict[str, torch.Tensor]] = []
    buffer: dict[str, torch.Tensor] | None = None
    cache: dict[str, torch.Tensor] | None = None
    # data_iterator = iter(train_loader)
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        if epoch % 10 == 0:
            # Save and evaluate model routinely
            accuracy = evaluate(model=ddp_model, device=device, test_loader=test_loader)
            # torch.save(ddp_model.state_dict(), model_filepath)
            log(INFO, "-" * 75)
            log(INFO, "Round: %s, Accuracy: %s", epoch, accuracy)
            log(INFO, "-" * 75)

        log(INFO, "Local Rank: %s, Round: %s", local_rank, epoch)

        federated_dataset.list_of_clients = list(range(n_clients_per_round))

        for client_id, tmp_client_dataset in zip(
            list(range(n_clients_per_round)),
            DataLoader(
                federated_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x[0],
                num_workers=10,  # self.client_prefetch_num_workers,
                prefetch_factor=5,
            ),
            strict=True,
        ):
            # for i in range(n_clients_per_round):
            data_iterator = iter(
                DataLoader(
                    dataset=tmp_client_dataset,
                    batch_size=batch_size,
                    num_workers=0,  # , prefetch_factor=50
                )
            )

            log(
                INFO,
                "Local Rank: %s, Round: %s, Training client %i",
                local_rank,
                epoch,
                client_id,
            )

            initial_model_variable_map = get_parameters(get_variable_map(model))

            ddp_model.train()

            i = 0
            while i < n_samples_per_client // batch_size:
                data = next(data_iterator)
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = ddp_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                i += 1

            deltas, cache = get_model_difference(
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

        if buffer is not None:
            for tensor_value in buffer.values():
                tensor_value.div_(n_clients_per_round)
            set_parameters(buffer, get_variable_map(model))
            for tensor_name, tensor_value in buffer.items():
                buffer[tensor_name] = torch.empty(
                    tensor_value.shape, dtype=torch.float32, device=tensor_value.device
                )


if __name__ == "__main__":
    """
    How to run this script:
        ```console
        $ mkdir -p data
        $ mkdir -p saved_models
        $ cd data
        $ wget -c --quiet https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
        $ tar -xvzf cifar-10-python.tar.gz
        $ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0
            --master_addr="192.168.0.1" --master_port=1234 pytorch_distr_test.py
        $ torchrun --nproc-per-node 2 --nnodes 1 --node-rank 0 --master-addr="127.0.0.1"
            --master-port=51550 pollen_worker/pytorch_distr_test.py
        $ torchrun --nproc-per-node 1 --nnodes 1 --node-rank 0 --master-addr="127.0.0.1"
            --master-port=51550 pollen_worker/pytorch_distr_test.py
        $ kill $(ps aux | grep pytorch_distr_test.py | grep -v grep | awk '{print $2}')
        ```
    """
    main()
