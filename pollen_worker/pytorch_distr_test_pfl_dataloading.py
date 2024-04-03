"""PyTorch CIFAR-10 distributed training example.

From: https://leimao.github.io/blog/PyTorch-Distributed-Training/
Also seen:
 - https://pytorch.org/docs/stable/elastic/run.html#launcher-api
"""

from collections.abc import Callable, Iterable
import os
from pathlib import Path
import pickle
import time
import torch
from torch.utils.data import DataLoader
from torch.distributed import init_process_group, dist_backend
from torch import nn
from torch import optim

from flwr.common import log
from logging import INFO
import random
import numpy as np

from pollen_worker.datasets.federated_dataset import FederatedDataset
from pollen_worker.pollen_utils import get_model


def in_memory_reshape(
    flat: torch.Tensor, tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Reshape a flat tensor into a list of tensors."""
    offset = 0
    for t in tensors:
        t.data.copy_(flat[offset : offset + t.numel()].reshape(t.shape))
        offset += t.numel()
    return tensors


def flatten(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, list[tuple], list[torch.dtype]]:
    """Flatten a list of PyTorch tensors into a single vector.

    :param tensors:
        A list of tensors to flatten.
    :return:
        `(vector, shapes, dtypes)`, where `vector` is the flattened tensor,
        `shapes` is a list of shapes of the input arrays and `dtypes` is a
        list of types of the input arrays. `shapes` and `dtypes` can be used
        with the `reshape` function to recover the original list of weights.
    """
    shapes = [tuple(v.shape) for v in tensors]
    dtypes = [v.dtype for v in tensors]
    vector = torch.cat([t.reshape(-1).type(torch.float32) for t in tensors])

    return vector, shapes, dtypes


def all_reduce(
    world_size: int, tensors: list[torch.Tensor], average: bool = False
) -> list[torch.Tensor]:
    """Execute an all-reduce operation on the provided tensors."""
    if world_size <= 1:
        return tensors

    flat, _, _ = flatten(tensors)
    torch.distributed.all_reduce(flat, op=torch.distributed.reduce_op.SUM)
    if average:
        flat /= world_size
    return in_memory_reshape(flat, tensors)


def to_tensor(values: list | np.ndarray, dtype: str | None = "float32") -> torch.Tensor:
    """Convert a list of values or a numpy array to a float32 Torch tensor."""
    torch_dtype = (
        getattr(torch, dtype) if dtype is not None and isinstance(dtype, str) else dtype
    )

    tensor = torch.as_tensor(values, dtype=torch_dtype)
    return tensor


def prepare_batch(batch) -> dict[int, torch.Tensor] | list[torch.Tensor]:
    """Transform a batch of data to tensors."""
    if isinstance(batch, dict):
        return {k: to_tensor(v) for k, v in batch.items()}
    else:
        return [to_tensor(data) for data in batch]


class ClientDataset(object):

    def __init__(self, data: tuple[np.ndarray, np.ndarray], client_id: int) -> None:
        self.data = data
        self.client_id = client_id
        self._batches: dict[int, list] = {}

    def __len__(self) -> int:
        return len(self.data[0])

    def iter(self, batch_size: int | None):
        if batch_size is None:
            yield self.data
            return

        def get_slice(
            data: np.ndarray | tuple[np.ndarray, np.ndarray], start_ix: int, end_ix: int
        ):
            if isinstance(data, list | tuple):
                # Input is a list of tensors.
                sliced = [get_slice(tensor, start_ix, end_ix) for tensor in data]
            else:
                assert isinstance(data, np.ndarray)
                # No list of multiple inputs.
                sliced = data[start_ix:end_ix]
            return sliced

        if batch_size not in self._batches:
            # Only keep a cache size of 1 to limit memory growth.
            # Only 1 is needed anyway if program uses static batch size.
            self._batches = {
                batch_size: [
                    get_slice(self.data, start_index, start_index + batch_size)
                    for start_index in range(0, len(self), batch_size)
                ]
            }
        yield from self._batches[batch_size]


class FederatedDataset(object):

    def __init__(
        self,
        data: (
            dict[int, tuple[np.ndarray, np.ndarray]]
            | list[int, tuple[np.ndarray, np.ndarray]]
        ),
        list_of_client_ids: list[int],
        world_size: int | None = None,
        local_rank: int | None = None,
    ) -> None:
        self.data = data
        self.list_of_client_ids = list_of_client_ids
        self.world_size = (
            int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
        )
        self.local_rank = (
            int(os.environ["LOCAL_RANK"]) if local_rank is None else local_rank
        )

    def make_dataset_fn(self, client_id: int) -> ClientDataset:
        return ClientDataset(data=self.data[client_id], client_id=client_id)

    def get_cohort(self, cohort_size: int) -> Iterable[ClientDataset]:
        for i in range(cohort_size):
            if (i % self.world_size) == self.local_rank:
                client_id = self.list_of_client_ids[i]
                yield self.make_dataset_fn(client_id)


def load_and_preprocess(
    pickle_file_path: str,
    channel_means: np.ndarray | None = None,
    channel_stddevs: np.ndarray | None = None,
    exclude_classes=None,
):
    with open(pickle_file_path, "rb") as f:
        images, labels = pickle.load(f)
    images = images.astype(np.float32)

    # Normalize per-channel.
    if channel_means is None:
        channel_means = images.mean(axis=(0, 1, 2), dtype="float64")
    if channel_stddevs is None:
        channel_stddevs = images.std(axis=(0, 1, 2), dtype="float64")
    images = (images - channel_means) / channel_stddevs

    if exclude_classes is not None:
        for exclude_class in exclude_classes:
            mask = (labels != exclude_class).reshape(-1)
            labels = labels[mask]
            images = images[mask]

    return images, labels, channel_means, channel_stddevs


def make_iid_federated_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    user_dataset_len_sampler: Callable[[], int],
    numpy_to_tensor: Callable = lambda x: x,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> FederatedDataset:
    """
    Create a federated dataset with IID users from the CIFAR10 dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the datapoints IID.
    """
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    start_ix = 0
    users_to_data: dict = {}
    while True:
        dataset_len = user_dataset_len_sampler()
        user_slice = slice(start_ix, start_ix + dataset_len)
        users_to_data[len(users_to_data)] = (images[user_slice], labels[user_slice])
        start_ix += dataset_len
        if start_ix >= len(images):
            break
    return FederatedDataset(
        data=users_to_data,
        list_of_client_ids=list(users_to_data),
        world_size=world_size,
        local_rank=local_rank,
    )


def make_cifar10_iid_datasets(
    data_dir: Path = Path("/datasets/cifar10"),
    user_dataset_len_sampler: Callable[[], int] = lambda: 50,
    numpy_to_tensor: Callable = lambda x: x,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> FederatedDataset:
    train_images, train_labels, _, _ = load_and_preprocess(data_dir / "cifar10_train.p")

    # create artificial federated training and val datasets
    # from central training and val data.
    return make_iid_federated_dataset(
        train_images,
        train_labels,
        user_dataset_len_sampler,
        numpy_to_tensor,
        world_size=world_size,
        local_rank=local_rank,
    )


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
) -> dict[str, torch.Tensor]:
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
    return model_diff


def main() -> None:
    """Execute the main function."""
    num_rounds = 10000
    n_clients_per_round = 50
    batch_size = 10
    n_samples_per_client = 50
    learning_rate = 0.1
    random_seed = 51550

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

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
    model = get_model("cifar10")

    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    # Data loading
    federated_dataset = make_cifar10_iid_datasets(
        user_dataset_len_sampler=lambda: n_samples_per_client,
        world_size=world_size,
        local_rank=local_rank,
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
                local_rank,
                federated_round,
                j,
            )
            for tensor_value in buffer.values():
                tensor_value.div_(j)
            reduced_buffer = all_reduce(world_size, list(buffer.values()), average=True)
            set_parameters(
                dict(zip(buffer.keys(), reduced_buffer, strict=False)),
                get_variable_map(model),
            )
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
        $ torchrun --nproc-per-node 1 --nnodes 1 --node-rank 0 --master-addr="127.0.0.1" \
            --master-port=51550 pollen_worker/pytorch_distr_test_pfl_dataloading.py
        $ torchrun --nproc-per-node 2 --nnodes 1 --node-rank 0 --master-addr="127.0.0.1" \
            --master-port=51550 pollen_worker/pytorch_distr_test_pfl_dataloading.py
        ```
    """
    main()
