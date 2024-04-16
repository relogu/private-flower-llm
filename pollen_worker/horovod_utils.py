"""Functions for Horovod-based training."""

from collections import OrderedDict
from collections.abc import Callable, Iterable
import os
from pathlib import Path
import pickle
import random
import socket
from typing import Any
import torch
from torch import device
from torch.nn import Module
import torch.distributed as dist
import torch.backends as torch_backends
import h5py
from torch.utils.data import DataLoader
from PIL import Image
from pollen_worker.datasets.openimage import train_transform
from torch.utils.data import Dataset

import numpy as np
from flwr.common import NDArrays
import torch.utils
from pollen_worker.datasets.openimage import OpenImage, _load_openimage_meta_data
from pollen_worker.datasets.shakespeare import (
    SHAKESPEARE_DTYPES,
)
from pollen_worker.datasets.shakespeare import (
    load_shakespeare_file,
)
from pollen_worker.datasets.shakespeare import LEAF_CHARACTERS
from pollen_worker.models.pfl_cnns import MultiLabelCNN
from pollen_worker.pollen_utils import get_clients_population_dict

from pollen_worker.datasets.pfl_datasets_common import (
    get_label_mapping,
    get_multi_hot_targets,
    get_user_num_images,
)


def get_free_tcp_port() -> int:
    """Get free socket port to use as MASTER_PORT."""
    # from https://www.programcreek.com/python/?CodeExample=get+free+port
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp.bind(("", 0))
    _, port = tcp.getsockname()
    tcp.close()
    return port


def set_random_seeds(random_seed: int = 0) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(random_seed)
    torch_backends.cudnn.deterministic = True  # type: ignore[reportAttributeAccessIssue]
    torch_backends.cudnn.benchmark = False  # type: ignore[reportAttributeAccessIssue]
    np.random.seed(random_seed)
    random.seed(random_seed)


def get_default_device() -> torch.device:
    """Return the default device for PyTorch.

    The default device can be manually set with the environment variable:
    POLLEN_PYTORCH_DEVICE
    """
    manual_device = os.environ.get("POLLEN_PYTORCH_DEVICE", None)
    if manual_device:
        # Default device can be overridden with env var.
        default_device = torch.device(manual_device)
    elif torch.cuda.is_available():
        default_device = torch.device("cuda")
    elif hasattr(torch_backends, "mps") and torch_backends.mps.is_available():  # type: ignore[reportAttributeAccessIssue]
        default_device = torch.device("mps")
    else:
        default_device = torch.device("cpu")
    return default_device


def get_ndarrays_from_model(
    net: Module,
    device: str = "cpu",
    to_numpy: bool = True,
) -> NDArrays | list[torch.Tensor]:
    """Return NDArrays from a PyTorch model."""
    # Put the model in eval mode
    net.eval()
    # Get the model parameters dictionary
    model_parameter_dict = net.named_parameters()
    # Sort the dictionary by key to ensure consistent order
    model_parameter_dict = sorted(model_parameter_dict)
    if device == "cpu" and to_numpy:
        tmp = [
            val.detach().to(device).numpy()
            for name, val in model_parameter_dict
            if "bn" not in name and val.requires_grad
        ]
    elif device == "cpu" and not to_numpy:
        tmp = [
            val.detach().to(device)
            for name, val in model_parameter_dict
            if "bn" not in name and val.requires_grad
        ]
    else:
        tmp = [
            val.detach().to(device)
            for name, val in model_parameter_dict
            if "bn" not in name and val.requires_grad
        ]
    return tmp


def set_model_parameters_from_ndarrays(
    parameters: NDArrays,
    model: Module,
    device: str | device = "cpu",
) -> None:
    """Set the model parameters from a NDArrays."""
    # Put the model in eval mode
    model.eval()
    # Get the model parameters dictionary
    model_parameter_dict = model.named_parameters()
    # Sort the dictionary by key to ensure consistent order
    model_parameter_dict = sorted(model_parameter_dict)
    module_state_dict = {
        k: val for k, val in model_parameter_dict if "bn" not in k and val.requires_grad
    }
    params_dict = zip(module_state_dict.keys(), parameters, strict=True)
    state_dict = OrderedDict(
        {k: torch.as_tensor(v, device=device) for k, v in params_dict}
    )
    model.load_state_dict(state_dict, strict=False)


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


def reshape(
    vector: torch.Tensor, shapes: list[tuple], dtypes: list[torch.dtype] | None = None
) -> list[torch.Tensor]:
    """
    Split and reshape a vector into a list of PyTorch tensors.

    :param vector:
        A 1-dimensional tensor to split and reshape.
    :param shapes:
        A list of tuples of integers, representing the shapes of multiple
        target weights to construct.
    :param dtypes:
        A list of types for the new weights.
    :return:
        A list of PyTorch tensors constructed from the inputs.
    """
    separating_index = 0
    weights = []
    for i, shape in enumerate(shapes):
        weight_len = np.prod(shape)
        new_flat_weight = vector[separating_index : separating_index + weight_len]
        if dtypes is not None:
            new_flat_weight = new_flat_weight.type(dtypes[i])
        weights.append(new_flat_weight.reshape(shape))
        separating_index += weight_len
    return weights


def in_memory_reshape(
    flat: torch.Tensor, tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Reshape a flat tensor into a list of tensors."""
    offset = 0
    for t in tensors:
        t.data.copy_(flat[offset : offset + t.numel()].reshape(t.shape))
        offset += t.numel()
    return tensors


def all_reduce(
    world_size: int, tensors: list[torch.Tensor], average: bool = False
) -> list[torch.Tensor]:
    """Execute an all-reduce operation on the provided tensors."""
    if world_size <= 1:
        return tensors

    flat, _, _ = flatten(tensors)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    if average:
        flat /= world_size
    return in_memory_reshape(flat, tensors)


def broadcast(world_size: int, tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Execute an broadcast operation on the provided tensors."""
    if world_size <= 1:
        return tensors

    flat, _, _ = flatten(tensors)
    dist.broadcast(flat, 0)
    return in_memory_reshape(flat, tensors)


def to_tensor(
    values: list | np.ndarray, _device: device, dtype: str | None = "float32"
) -> torch.Tensor:
    """Convert a list of values or a numpy array to a float32 Torch tensor."""
    torch_dtype = (
        getattr(torch, dtype) if dtype is not None and isinstance(dtype, str) else dtype
    )

    tensor = torch.as_tensor(values, dtype=torch_dtype)
    tensor = tensor.to(_device)
    return tensor


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


def prepare_batch(
    batch: Any, _device: device
) -> dict[int, torch.Tensor] | list[torch.Tensor]:
    """Transform a batch of data to tensors."""
    if isinstance(batch, dict):
        return {
            k: v if isinstance(v, torch.Tensor) else to_tensor(v, _device=_device)
            for k, v in batch.items()
        }
    else:
        return [
            data if isinstance(data, torch.Tensor) else to_tensor(data, _device=_device)
            for data in batch
        ]


class ClientDataset:
    """Implementation of a client dataset for federated learning."""

    def __init__(
        self,
        data: tuple[np.ndarray, np.ndarray],
        client_id: int,
        device: str,
        n_workers: int = 1,
        cap_batch_size: int | None = 256,
    ) -> None:
        self.data = data
        self.client_id = client_id
        self._batches: dict[int, list] = {}
        self.num_accumulation_steps: int = 1
        self.n_workers = n_workers
        self.cap_batch_size = cap_batch_size
        self.device = device

    def __len__(self) -> int:
        """Return the number of samples in the client dataset."""
        return len(self.data[1])

    def set_num_accumulation_steps(self, batch_size: int) -> None:
        """Set the number of accumulation steps for the client dataset."""
        # Get the 10% of the dataset
        threshold = max(1, int(len(self) * 1.0))
        # Round to the closest power of two
        preferred_batch_size = 2 ** int(np.log2(threshold))
        # Cap the batch size if needed
        if self.cap_batch_size:
            preferred_batch_size = min(preferred_batch_size, self.cap_batch_size)
        # Set the number of accumulation steps
        self.num_accumulation_steps = max(1, preferred_batch_size // batch_size)

    def iter(self, batch_size: int | None) -> Iterable[Any]:
        """Implement an iterator over the client dataset for a given batch size.

        It returns the entire dataset if batch_size is None.
        """
        if batch_size is None:
            yield self.data
            return

        def get_slice(
            data: np.ndarray | tuple[np.ndarray, np.ndarray], start_ix: int, end_ix: int
        ) -> list | np.ndarray[Any, Any]:
            """Return a slice of the data."""
            if isinstance(data, list | tuple):
                # Input is a list of tensors.
                return [get_slice(tensor, start_ix, end_ix) for tensor in data]
            else:
                assert isinstance(data, np.ndarray)
                # No list of multiple inputs.
                return data[start_ix:end_ix]

        if batch_size not in self._batches:
            # self.set_num_accumulation_steps(batch_size)
            # log(
            #     DEBUG,
            #     "Client %s num_accumulation_steps: %s",
            #     self.client_id, self.num_accumulation_steps,
            # )
            # Only keep a cache size of 1 to limit memory growth.
            # Only 1 is needed anyway if program uses static batch size.
            self._batches = {
                batch_size: [
                    get_slice(self.data, start_index, start_index + batch_size)
                    for start_index in range(0, len(self), batch_size)
                ]
            }
        yield from self._batches[batch_size]


class FakeShakespeareDataset(Dataset):
    """Fake Shakespeare dataset object."""

    def __init__(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """Initialize the dataset."""
        self.data = data

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the data at the given index."""
        return self.data[0][index], self.data[1][index]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data[0])


class ClientDatasetShakespeareDataloader(ClientDataset):
    """Implementation of a client dataset for federated learning."""

    def __init__(
        self,
        data: tuple[np.ndarray, np.ndarray],
        client_id: int,
        n_workers: int,
        device: str,
    ) -> None:
        self.data = data
        self.client_id = client_id
        self._batches: dict[int, list] = {}
        self.n_workers = n_workers
        self.dataset = FakeShakespeareDataset(data)
        self.device = device

    def __len__(self) -> int:
        """Return the number of samples in the client dataset."""
        return len(self.data[0])

    def iter(self, batch_size: int | None) -> Iterable[Any]:
        """Implement an iterator over the client dataset for a given batch size.

        It returns the entire dataset if batch_size is None.
        """
        if batch_size is None:
            yield self.data
            return
        train_loader = DataLoader(
            # NOTE: Non-default arguments
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            # NOTE: Prevent runtime error related to BatchNorm, apparently
            drop_last=False,
            # copy Tensors into CUDA pinned memory before returning them
            pin_memory=True,
            # the device to be used for pinning the memory
            pin_memory_device=self.device,
            # builds batches from samples
            collate_fn=None,
            # NOTE: Default arguments
            # how to draw sample from the dataset
            sampler=None,
            # like the above but for batches
            batch_sampler=None,
            # if positive, the timeout value for collecting a batch from workers
            timeout=0,
            # init function for worker processes
            worker_init_fn=None,
            multiprocessing_context=None,
            # This allows to maintain the workers
            prefetch_factor=2 if self.n_workers > 0 else None,
            # PRNG to use for random sampling
            generator=None,
            persistent_workers=False,
        )

        yield from train_loader


class ClientDatasetOpenImageNoDataloader:
    """Implementation of a client dataset for federated learning."""

    def __init__(
        self,
        data: tuple[list, list],
        client_id: int,
        path: Path,
        root: Path,
        n_workers: int,
        device: str,
        cap_batch_size: int = 256,
    ) -> None:
        self.data = data
        self.client_id = client_id
        self.path = path
        self._batches: dict[int, list] = {}
        self.dataset = OpenImage(
            root,
            data,
            client_id,
        )
        self.n_workers = n_workers
        self.device = device
        self.cap_batch_size = cap_batch_size
        self.num_accumulation_steps = 1

    def __len__(self) -> int:
        """Return the number of samples in the client dataset."""
        return len(self.data[0])

    def set_num_accumulation_steps(self, batch_size: int) -> None:
        """Set the number of accumulation steps for the client dataset."""
        # Get the 10% of the dataset
        threshold = max(1, int(len(self) * 1.0))
        # Round to the closest power of two
        preferred_batch_size = 2 ** int(np.log2(threshold))
        # Cap the batch size if needed
        if self.cap_batch_size:
            preferred_batch_size = min(preferred_batch_size, self.cap_batch_size)
        # Set the number of accumulation steps
        self.num_accumulation_steps = max(1, preferred_batch_size // batch_size)

    def iter(self, batch_size: int | None) -> Iterable[Any]:
        """Implement an iterator over the client dataset for a given batch size.

        It returns the entire dataset if batch_size is None.
        """
        if batch_size is None:
            yield self.data
            return

        def get_slice(data: tuple[list, list], start_ix: int, end_ix: int) -> list:
            """Return a slice of the data."""
            img_names, targets = data
            features, labels = [], []
            for i in range(start_ix, min(end_ix, len(img_names))):
                img_name = img_names[i]
                target = targets[i]
                img = Image.open(self.path / img_name)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                features.append(train_transform(img))
                labels.append(torch.tensor(target))
            return [torch.stack(features), torch.stack(labels)]

        if batch_size not in self._batches:
            # self.set_num_accumulation_steps(batch_size)
            # Only keep a cache size of 1 to limit memory growth.
            # Only 1 is needed anyway if program uses static batch size.
            self._batches = {
                batch_size: [
                    get_slice(self.data, start_index, start_index + batch_size)
                    for start_index in range(0, len(self), batch_size)
                ]
            }
        yield from self._batches[batch_size]


class ClientDatasetOpenImageDataloader(ClientDatasetOpenImageNoDataloader):
    """Implementation of a client dataset for federated learning."""

    def __init__(
        self,
        data: tuple[list, list],
        client_id: int,
        path: Path,
        root: Path,
        n_workers: int,
        device: str,
    ) -> None:
        self.data = data
        self.client_id = client_id
        self.path = path
        self._batches: dict[int, list] = {}
        self.dataset = OpenImage(
            root,
            data,
            client_id,
        )
        self.n_workers = n_workers
        self.device = device

    def __len__(self) -> int:
        """Return the number of samples in the client dataset."""
        return len(self.data[0])

    def iter(self, batch_size: int | None) -> Iterable[Any]:
        """Implement an iterator over the client dataset for a given batch size.

        It returns the entire dataset if batch_size is None.
        """
        if batch_size is None:
            yield self.data
            return
        n_workers = self.n_workers
        train_loader = DataLoader(
            # NOTE: Non-default arguments
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            # NOTE: Prevent runtime error related to BatchNorm, apparently
            drop_last=False,
            # copy Tensors into CUDA pinned memory before returning them
            pin_memory=True,
            # the device to be used for pinning the memory
            pin_memory_device=self.device,
            # builds batches from samples
            collate_fn=None,
            # NOTE: Default arguments
            # how to draw sample from the dataset
            sampler=None,
            # like the above but for batches
            batch_sampler=None,
            # if positive, the timeout value for collecting a batch from workers
            timeout=0,
            # init function for worker processes
            worker_init_fn=None,
            multiprocessing_context=None,
            # This allows to maintain the workers
            prefetch_factor=2 if n_workers > 0 else None,
            # PRNG to use for random sampling
            generator=None,
            persistent_workers=False,
        )

        yield from train_loader


class FederatedDataset:
    """Implementation of a federated dataset of clients for federated learning."""

    def __init__(
        self,
        data: (
            dict[int, tuple[np.ndarray, np.ndarray]]
            | list[tuple[np.ndarray, np.ndarray]]
        ),
        list_of_client_ids: list[int],
        device: str,
        n_workers: int = 0,
        world_size: int | None = None,
        local_rank: int | None = None,
        client_dataset_type: type[ClientDataset] = ClientDataset,
    ) -> None:
        self.data = data
        self.list_of_client_ids = list_of_client_ids
        self.world_size = (
            int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
        )
        self.local_rank = (
            int(os.environ["LOCAL_RANK"]) if local_rank is None else local_rank
        )
        self.client_dataset_type = client_dataset_type
        self.n_workers = n_workers
        self.device = device

    def make_dataset_fn(self, client_id: int) -> ClientDataset:
        """Return a client dataset for the given client_id."""
        return self.client_dataset_type(
            data=self.data[client_id],
            client_id=client_id,
            n_workers=self.n_workers,
            device=self.device,
        )

    def get_cohort(self, cohort_size: int | list[int]) -> Iterable[ClientDataset]:
        """Iterate over a cohort of clients."""
        if isinstance(cohort_size, int):
            for i in range(cohort_size):
                if (i % self.world_size) == self.local_rank:
                    client_id = self.list_of_client_ids[i]
                    yield self.make_dataset_fn(client_id)
        if isinstance(cohort_size, list):
            for client_id in cohort_size:
                yield self.make_dataset_fn(client_id)


class FederatedDatasetOpenImage:
    """Implementation of a federated dataset of clients for federated learning."""

    def __init__(
        self,
        path: Path,
        root: Path,
        data: dict[int, tuple[list, list]],
        list_of_client_ids: list[int],
        device: str,
        n_workers: int = 1,
        world_size: int | None = None,
        local_rank: int | None = None,
        client_dataset_type: type[
            ClientDatasetOpenImageNoDataloader
        ] = ClientDatasetOpenImageDataloader,
    ) -> None:
        self.data = data
        self.path = path
        self.list_of_client_ids = list_of_client_ids
        self.world_size = (
            int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
        )
        self.local_rank = (
            int(os.environ["LOCAL_RANK"]) if local_rank is None else local_rank
        )
        self.client_dataset_type = client_dataset_type
        self.root = root
        self.n_workers = n_workers
        self.device = device

    def make_dataset_fn(self, client_id: int) -> ClientDatasetOpenImageNoDataloader:
        """Return a client dataset for the given client_id."""
        return self.client_dataset_type(
            data=self.data[client_id],
            client_id=client_id,
            path=self.path,
            root=self.root,
            n_workers=self.n_workers,
            device=self.device,
        )

    def get_cohort(
        self, cohort_size: int | list[int]
    ) -> Iterable[ClientDatasetOpenImageNoDataloader]:
        """Iterate over a cohort of clients."""
        if isinstance(cohort_size, int):
            for i in range(cohort_size):
                if (i % self.world_size) == self.local_rank:
                    client_id = self.list_of_client_ids[i]
                    yield self.make_dataset_fn(client_id)
        if isinstance(cohort_size, list):
            for client_id in cohort_size:
                yield self.make_dataset_fn(client_id)


class FLAIRFederatedDataset:
    """Implementation of the FLAIR federated dataset."""

    def __init__(
        self,
        hdf5_path: Path,
        list_of_client_ids: list[str],
        max_num_user_images: int,
        partition: str,
        use_fine_grained_labels: bool,
        num_classes: int,
        device: str,
        world_size: int | None = None,
        local_rank: int | None = None,
    ) -> None:
        self.hdf5_path = hdf5_path
        self.list_of_client_ids = list_of_client_ids
        self.max_num_user_images = max_num_user_images
        self.partition = partition
        self.use_fine_grained_labels = use_fine_grained_labels
        self.num_classes = num_classes
        self.device = device
        self.world_size = (
            int(os.environ["WORLD_SIZE"]) if world_size is None else world_size
        )
        self.local_rank = (
            int(os.environ["LOCAL_RANK"]) if local_rank is None else local_rank
        )

    def make_dataset_fn(self, client_id: str) -> ClientDataset:
        """Return a client dataset for the given client_id."""
        with h5py.File(self.hdf5_path, "r") as h5:
            inputs = np.array(h5[f"/{self.partition}/{client_id}/images"])
            targets = get_multi_hot_targets(
                (len(inputs), self.num_classes),
                h5,
                self.partition,
                client_id,
                self.use_fine_grained_labels,
            )
            user_slice = slice(0, self.max_num_user_images)
            inputs, targets = inputs[user_slice], targets[user_slice]
            data_order = np.random.permutation(len(inputs))
        return ClientDataset(
            data=(inputs[data_order], targets[data_order]),
            client_id=self.list_of_client_ids.index(client_id),
            device=self.device,
        )

    def get_cohort(self, cohort_size: int | list[int]) -> Iterable[ClientDataset]:
        """Iterate over a cohort of clients."""
        if isinstance(cohort_size, int):
            for i in range(cohort_size):
                if (i % self.world_size) == self.local_rank:
                    client_id = self.list_of_client_ids[i]
                    yield self.make_dataset_fn(client_id)
        if isinstance(cohort_size, list):
            for i in cohort_size:
                client_id = self.list_of_client_ids[i]
                yield self.make_dataset_fn(client_id)


def make_flair_federated_dataset(
    hdf5_path: Path,
    partition: str,
    use_fine_grained_labels: bool,
    max_num_user_images: int,
    device: str,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> FLAIRFederatedDataset:
    """Create federated dataset from the flair dataset, to use in simulations.

    The federated dataset samples user datasets. A user dataset is
    made from data points of one us er.

    :param hdf5_path:
        A h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :param max_num_user_images:
        Maximum number of images each user can have.
    :param numpy_to_tensor:
        Function that convert numpy array to ML framework tensor.
    :return:
        Federated dataset from the HDF5 data file.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    user_num_images = get_user_num_images(hdf5_path, partition)
    user_ids = sorted(user_num_images.keys())

    return FLAIRFederatedDataset(
        hdf5_path=hdf5_path,
        list_of_client_ids=user_ids,
        max_num_user_images=max_num_user_images,
        partition=partition,
        use_fine_grained_labels=use_fine_grained_labels,
        num_classes=num_classes,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def load_and_preprocess_cifar10(
    pickle_file_path: Path,
    channel_means: np.ndarray | None = None,
    channel_stddevs: np.ndarray | None = None,
    exclude_classes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load and preprocess CIFAR10 data from a pickle file."""
    images: np.ndarray
    labels: np.ndarray
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
    device: str,
    numpy_to_tensor: Callable = lambda x: x,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> FederatedDataset:
    """
    Create a federated dataset with IID users from the CIFAR10 dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the data points IID.
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
        device=device,
    )


def make_cifar10_iid_datasets(
    device: str,
    data_dir: Path = Path("/datasets/cifar10"),
    user_dataset_len_sampler: Callable[[], int] = lambda: 50,
    numpy_to_tensor: Callable = lambda x: x,
    world_size: int | None = None,
    local_rank: int | None = None,
) -> FederatedDataset:
    """Construct the CIFAR10 IID federated dataset."""
    train_images, train_labels, _, _ = load_and_preprocess_cifar10(
        data_dir / "cifar10_train.p"
    )
    return make_iid_federated_dataset(
        images=train_images,
        labels=train_labels,
        user_dataset_len_sampler=user_dataset_len_sampler,
        numpy_to_tensor=numpy_to_tensor,
        device=device,
        world_size=world_size,
        local_rank=local_rank,
    )


def classification_training_loop(
    client_model: Module | MultiLabelCNN,
    local_optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    _device: device,
    client_dataset: ClientDataset | ClientDatasetOpenImageNoDataloader,
    batch_size: int,
    dataset_name: str,
    n_local_epochs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the local training loop for CIFAR10, OpenImage and Shakespeare."""
    client_model.train()

    cumulative_loss: torch.Tensor = torch.tensor(0.0, device=_device)
    cumulative_accuracy: torch.Tensor = torch.tensor(0.0, device=_device)
    cumulative_num_samples: torch.Tensor = torch.tensor(0.0, device=_device)
    for _ in range(n_local_epochs):
        for data in client_dataset.iter(batch_size):
            prepared_batch = prepare_batch(data, _device=_device)
            inputs, labels = prepared_batch[0].to(_device), prepared_batch[1].to(
                _device
            )
            if dataset_name == "cifar10" and labels.dim() > 1:
                labels = labels.squeeze()
            local_optimizer.zero_grad()
            outputs: torch.Tensor = client_model(inputs)
            loss: torch.Tensor = criterion(outputs, labels.long())
            loss.backward()
            local_optimizer.step()
            cumulative_loss += loss.detach().clone().item()
            cumulative_accuracy += (
                (outputs.argmax(-1) == labels.long()).detach().clone().sum().item()
            )
            cumulative_num_samples += len(labels)
    return cumulative_loss, cumulative_accuracy, cumulative_num_samples


def flair_training_loop(
    client_model: Module | MultiLabelCNN,
    local_optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    _device: device,
    client_dataset: ClientDataset | ClientDatasetOpenImageNoDataloader,
    batch_size: int,
    dataset_name: str,
    n_local_epochs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the local training loop for FLAIR."""
    client_model.train()

    cumulative_loss: torch.Tensor = torch.tensor(0.0, device=_device)
    cumulative_accuracy: torch.Tensor = torch.tensor(0.0, device=_device)
    cumulative_num_samples: torch.Tensor = torch.tensor(0.0, device=_device)
    for _ in range(n_local_epochs):
        # TODO: Assess if this is really necessary
        local_optimizer.zero_grad()
        for i, data in enumerate(client_dataset.iter(batch_size)):
            prepared_batch = prepare_batch(data, _device=_device)
            inputs, labels = prepared_batch[0].to(_device), prepared_batch[1].to(
                _device
            )
            # local_optimizer.zero_grad()
            outputs: torch.Tensor = client_model(inputs)
            loss: torch.Tensor = criterion(outputs, labels)
            num_data = len(inputs)
            num_predictions = np.ones(client_model._num_outputs) * num_data  # type: ignore[reportArgumentType]
            scores = torch.sigmoid(outputs)
            predictions = torch.round(scores)
            cumulative_loss += loss.item()
            cumulative_accuracy += (
                torch.sum(torch.eq(labels, predictions), dim=0).sum().item()
                / num_predictions.sum()
            )
            cumulative_num_samples += num_predictions.sum()
            loss.backward()
            if (i + 1) % client_dataset.num_accumulation_steps:  # type: ignore[reportArgumentType]
                local_optimizer.step()
                local_optimizer.zero_grad()
        # TODO: Assess if this is an overhead whenever the gradients are actually zero
        # because the accumulation steps split evenly the number of iterations performed
        local_optimizer.step()
    return cumulative_loss, cumulative_accuracy, cumulative_num_samples


def make_shakespeare_natural_partition(
    device: str,
    root: Path = Path("/datasets/FedScale/leaf_shakespeare"),
    dataset_type: str = "train",
    n_workers: int = 0,
    world_size: int | None = None,
    local_rank: int | None = None,
    client_samples_dict: dict[str | int, int] | None = None,
    client_dataset_type: type[ClientDataset] = ClientDataset,
) -> FederatedDataset:
    """Load and preprocess SHAKESPEARE data from the shakespeare files."""
    if client_samples_dict is None:
        client_samples_dict = get_clients_population_dict(
            name="shakespeare_memory", batch_size=10, seed=1337, dataset="train"
        )
    path_to_mapping = Path(root, "client_data_mapping")
    path_to_data = Path(root, "data")

    users_to_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for client_id in client_samples_dict:
        path = Path(path_to_mapping / dataset_type / f"{client_id}.parquet")
        data, labels = load_shakespeare_file(
            path, path_to_data, dataset_type, SHAKESPEARE_DTYPES
        )
        pre_processed_data = np.array(
            [[LEAF_CHARACTERS.find(c) for c in word] for word in data]
        )
        pre_processed_labels = np.array([LEAF_CHARACTERS.find(c) for c in labels])
        users_to_data[int(client_id)] = (pre_processed_data, pre_processed_labels)

    return FederatedDataset(
        data=users_to_data,
        n_workers=n_workers,
        list_of_client_ids=list(users_to_data),
        world_size=world_size,
        local_rank=local_rank,
        client_dataset_type=client_dataset_type,
        device=device,
    )


def make_openimage_natural_partition(
    device: str,
    root: Path = Path("/datasets/FedScale/openImg"),
    dataset_type: str = "train",
    n_workers: int = 1,
    world_size: int | None = None,
    local_rank: int | None = None,
    client_samples_dict: dict[str | int, int] | None = None,
    client_dataset_type: type[
        ClientDatasetOpenImageNoDataloader
    ] = ClientDatasetOpenImageDataloader,
) -> FederatedDatasetOpenImage:
    """Load and preprocess SHAKESPEARE data from the shakespeare files."""
    if client_samples_dict is None:
        client_samples_dict = get_clients_population_dict(
            name="openimage", batch_size=20, seed=1337, dataset="train"
        )

    path_to_mapping = Path(root, "client_data_mapping")
    path_to_data = Path(root, dataset_type)

    users_to_data: dict[int, tuple[list, list]] = {}

    for client_id in client_samples_dict:
        path = Path(path_to_mapping / dataset_type / f"{client_id}.parquet")
        data, labels = _load_openimage_meta_data(path)
        users_to_data[int(client_id)] = (data, labels)

    return FederatedDatasetOpenImage(
        path=path_to_data,
        root=root,
        data=users_to_data,
        n_workers=n_workers,
        list_of_client_ids=list(users_to_data),
        world_size=world_size,
        local_rank=local_rank,
        client_dataset_type=client_dataset_type,
        device=device,
    )
