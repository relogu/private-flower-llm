"""Module for the FLAIR dataset.

The script has been slightly modified from: TODO: add link to original script (numpy.py)
Also, some functions has been added from: TODO: add link to original script (_init_.py)
"""

# Copyright © 2023-2024 Apple Inc.

from collections.abc import Callable
from logging import INFO
from pathlib import Path
import time
from typing import Any

import h5py
from matplotlib import pyplot as plt
import numpy as np
from flwr.common import log
import pandas as pd

from pfl.data import ArtificialFederatedDataset, FederatedDataset
from pfl.data.dataset import Dataset
from pfl.data.sampling import get_data_sampler, get_user_sampler
import torch

from pollen_worker.datasets.pfl_datasets_common import (
    get_label_mapping,
    get_multi_hot_targets,
    get_user_num_images,
    get_channel_mean_stddevs,
)


def get_metadata(data_path: Path, use_fine_grained_labels: bool) -> dict[str, Any]:
    """Return metadata for the FLAIR dataset."""
    channel_mean, channel_stddevs = get_channel_mean_stddevs()
    label_mapping = get_label_mapping(data_path, use_fine_grained_labels)

    metadata = {
        "channel_mean": channel_mean,
        "channel_stddevs": channel_stddevs,
        "label_mapping": label_mapping,
    }
    return metadata


def make_flair_datasets(
    data_path: Path,
    use_fine_grained_labels: bool,
    max_num_user_images: int,
    numpy_to_tensor: Callable,
) -> tuple[FederatedDataset, FederatedDataset, dict[str, Any]]:
    """Create a train and val ``FederatedDataset`` from the FLAIR dataset."""
    training_federated_dataset = make_federated_dataset(
        data_path,
        "train",
        use_fine_grained_labels,
        max_num_user_images,
        numpy_to_tensor,
    )
    val_federated_dataset = make_federated_dataset(
        data_path, "val", use_fine_grained_labels, max_num_user_images, numpy_to_tensor
    )

    metadata = get_metadata(data_path, use_fine_grained_labels)
    return (training_federated_dataset, val_federated_dataset, metadata)


def make_flair_iid_datasets(
    data_path: Path,
    use_fine_grained_labels: bool,
    user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable,
) -> tuple[ArtificialFederatedDataset, ArtificialFederatedDataset, dict[str, Any]]:
    """Create a train and val ``ArtificialFederatedDataset` from the FLAIR dataset."""
    log(INFO, "Creating FLAIR with artificial I.I.D. data distribution")
    training_federated_dataset = make_artificial_federated_dataset(
        data_path,
        "train",
        use_fine_grained_labels,
        user_dataset_len_sampler,
        numpy_to_tensor,
    )
    val_federated_dataset = make_artificial_federated_dataset(
        data_path,
        "val",
        use_fine_grained_labels,
        user_dataset_len_sampler,
        numpy_to_tensor,
    )
    metadata = get_metadata(data_path, use_fine_grained_labels)
    return (training_federated_dataset, val_federated_dataset, metadata)


def make_federated_dataset(
    hdf5_path: Path,
    partition: str,
    use_fine_grained_labels: bool,
    max_num_user_images: int,
    numpy_to_tensor: Callable = lambda x: x,
) -> FederatedDataset:
    """Create federated dataset from the flair dataset, to use in simulations.

    The federated dataset samples user datasets. A user dataset is
    made from data points of one user.

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
    sampler = get_user_sampler("random", user_ids)

    def _make_dataset_fn(user_id: str) -> Dataset:
        with h5py.File(hdf5_path, "r") as h5:
            inputs = np.array(h5[f"/{partition}/{user_id}/images"])
            targets = get_multi_hot_targets(
                (len(inputs), num_classes),
                h5,
                partition,
                user_id,
                use_fine_grained_labels,
            )
            user_slice = slice(0, max_num_user_images)
            inputs, targets = inputs[user_slice], targets[user_slice]
            data_order = np.random.permutation(len(inputs))
            inputs = numpy_to_tensor(inputs[data_order])
            targets = numpy_to_tensor(targets[data_order])
        return Dataset(
            raw_data=[inputs, targets],
            train_kwargs={"eval": False},
            eval_kwargs={"eval": True},
            user_id=user_id,
        )

    return FederatedDataset(
        _make_dataset_fn, sampler, user_id_to_weight=user_num_images
    )


def make_artificial_federated_dataset(
    hdf5_path: Path,
    partition: str,
    use_fine_grained_labels: bool,
    user_dataset_len_sampler: Callable,
    numpy_to_tensor: Callable = lambda x: x,
) -> ArtificialFederatedDataset:
    """Create artificial I.I.D. federated dataset from the flair dataset.

    I.I.D. dataset is created by sampling data from
    all data points and ignoring the existing user IDs.

    :param hdf5_path:
        A h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :param user_dataset_len_sampler:
        A callable that should sample the dataset length of a user, i.e.
        `callable() -> dataset_length`.
    :param numpy_to_tensor:
        Function that convert numpy array to ML framework tensor.
    :return:
        Artificial I.I.D. federated dataset from the HDF5 data file.
    """
    num_classes = len(get_label_mapping(hdf5_path, use_fine_grained_labels))
    user_num_images = get_user_num_images(hdf5_path, partition)
    user_image_id = []
    for user_id in sorted(user_num_images.keys()):
        num_images = user_num_images[user_id]
        for image_id in range(num_images):
            user_image_id.append((user_id, image_id))

    def _make_dataset_fn(indices: list[int]) -> Dataset:
        inputs, targets = [], []
        with h5py.File(hdf5_path, "r") as h5:
            for index in indices:
                user_id, image_id = user_image_id[index]
                image = h5[f"/{partition}/{user_id}/images"][image_id]
                targets_shape = (user_num_images[user_id], num_classes)
                target = get_multi_hot_targets(
                    targets_shape, h5, partition, user_id, use_fine_grained_labels
                )[image_id]
                inputs.append(np.expand_dims(image, axis=0))
                targets.append(np.expand_dims(target, axis=0))
        inputs = numpy_to_tensor(np.vstack(inputs))
        targets = numpy_to_tensor(np.vstack(targets))
        return Dataset(
            raw_data=[inputs, targets],
            train_kwargs={"eval": False},
            eval_kwargs={"eval": True},
        )

    data_sampler = get_data_sampler("random", len(user_image_id))
    return ArtificialFederatedDataset(
        _make_dataset_fn, data_sampler, user_dataset_len_sampler
    )


def get_client_features_from_hdf5(
    hdf5_path: Path,
    partition: str,
    use_fine_grained_labels: bool,
    max_num_user_images: int | None,
    num_classes: int,
    user_id: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the client features and targets from the HDF5 file."""
    with h5py.File(hdf5_path, "r") as h5:
        inputs = np.array(h5[f"/{partition}/{user_id}/images"])
        targets = get_multi_hot_targets(
            (len(inputs), num_classes),
            h5,
            partition,
            user_id,
            use_fine_grained_labels,
        )
        if max_num_user_images:
            user_slice = slice(0, max_num_user_images)
            inputs, targets = inputs[user_slice], targets[user_slice]
        data_order = np.random.permutation(len(inputs))
        inputs, targets = inputs[data_order], targets[data_order]
        inputs = torch.as_tensor(inputs)
        targets = torch.as_tensor(targets)
        return inputs, targets


class FLAIRDataset(torch.utils.data.Dataset):
    """FLAIR Pytorch Dataset, each item contains the data tensors for a user ID."""

    def __init__(
        self,
        hdf5_path: Path,
        user_ids: list[str],
        user_id: int,
        partition: str,
        use_fine_grained_labels: bool,
        max_num_user_images: int | None = None,
    ) -> None:
        self._hdf5_path = hdf5_path
        self._user_id = user_ids[user_id]
        # Added for compatibility with other tasks
        self.client_id = self._user_id
        self._partition = partition
        self._use_fine_grained_labels = use_fine_grained_labels
        self._num_classes = len(
            get_label_mapping(self._hdf5_path, self._use_fine_grained_labels)
        )
        self._max_num_user_images = max_num_user_images

        self.inputs, self.targets = get_client_features_from_hdf5(
            hdf5_path=self._hdf5_path,
            partition=self._partition,
            use_fine_grained_labels=self._use_fine_grained_labels,
            max_num_user_images=self._max_num_user_images,
            num_classes=self._num_classes,
            user_id=str(self._user_id),
        )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the sample at current `index`."""
        return self.inputs[index], self.targets[index]


def _create_parquet_client_samples_dict(
    hdf5_path: Path = Path("/datasets/flair/flair_federated.hdf5"),
    dataset: str = "train",
) -> None:
    # Get the user_num_images
    user_num_images = get_user_num_images(hdf5_path, dataset)
    # Convert the dictionary to a DataFrame
    clients_samples_df = pd.DataFrame.from_dict(user_num_images, "index").reset_index()
    # Rename the columns
    clients_samples_df.columns = ["client_id", "samples"]
    log(INFO, clients_samples_df)
    if not Path(
        f"/datasets/flair/client_data_mapping/{dataset}_clients_dict.parquet"
    ).exists():
        Path("/datasets/flair/client_data_mapping").mkdir(parents=True, exist_ok=True)
        clients_samples_df.to_parquet(
            f"/datasets/flair/client_data_mapping/{dataset}_clients_dict.parquet"
        )


if __name__ == "__main__":
    # Example usage
    hdf5_path = Path("/datasets/flair/flair_federated.hdf5")
    dataset = "train"
    start_time = time.time()
    user_num_images = get_user_num_images(hdf5_path, dataset)
    log(INFO, f"Time to get user_num_images: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    metadata = get_metadata(hdf5_path, True)
    # log(INFO, "Metadata of FLAIR dataset: %s", metadata)
    log(INFO, f"Time to get metadata: {time.time() - start_time:.2f} seconds")
    # Create mapping user - number of samples
    start_time = time.time()
    user_image_id = []
    for user_id in sorted(user_num_images.keys()):
        num_images = user_num_images[user_id]
        for image_id in range(num_images):
            user_image_id.append((user_id, image_id))
    log(INFO, f"Time to get user_image_id: {time.time() - start_time:.2f} seconds")
    # # Create each user dataset
    # start_time = time.time()
    # for user_id in sorted(user_num_images.keys()):
    #     user_dataset = FLAIRDataset(
    #         hdf5_path=hdf5_path,
    #         user_id=user_id,
    #         partition=dataset,
    #         use_fine_grained_labels=True,
    #         max_num_user_images=None,
    #     )
    #     # Check that the length of the dataset is the same as the number of samples
    #     assert (
    #         len(user_dataset) == user_num_images[user_id]
    #     ), f"Length mismatch for {user_id}"
    log(INFO, f"Time to create user_dataset: {time.time() - start_time:.2f} seconds")
    for dataset in ["train", "test"]:
        if not Path(
            f"/datasets/flair/client_data_mapping/{dataset}_clients_dict.parquet"
        ).exists():
            _create_parquet_client_samples_dict(hdf5_path=hdf5_path, dataset=dataset)
        # Plot the number of samples per client
        clients_samples_df = pd.read_parquet(
            f"/datasets/flair/client_data_mapping/{dataset}_clients_dict.parquet"
        )
        log(INFO, clients_samples_df.describe())
        cap_samples = 200
        clients_samples_df[clients_samples_df["samples"] < cap_samples]["samples"].hist(
            bins=100
        )
        plt.savefig(f"{dataset}_chiappe.png")
        # plt.show()
