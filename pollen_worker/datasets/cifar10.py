"""Module for the CIFAR10 dataset.

The script has been slightly modified from: TODO: add link to original script (numpy.py)
"""

# Copyright © 2023-2024 Apple Inc.
from logging import INFO
from pathlib import Path
import pickle
from collections.abc import Callable
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from flwr.common import log
from pfl.data.partition import partition_by_dirichlet_class_distribution

from pollen_worker.datasets.pfl_datasets_common import (
    parse_draw_num_datapoints_per_user,
)


def numpy_to_tensor(x: Any) -> Any:
    """Identity."""
    return x


def load_and_preprocess(
    pickle_file_path: Path,
    channel_means: np.ndarray | None = None,
    channel_stddevs: np.ndarray | None = None,
    exclude_classes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load and preprocess the CIFAR10 dataset."""
    images: np.ndarray
    labels: np.ndarray
    with open(pickle_file_path, "rb") as f:
        images, labels = pickle.load(f)
    images = images.astype(np.float32)

    # Normalize per-channel.
    if channel_means is None:
        channel_means = images.mean(axis=(0, 1, 2), dtype="float64").astype(np.float32)
    if channel_stddevs is None:
        channel_stddevs = images.std(axis=(0, 1, 2), dtype="float64").astype(np.float32)
    images = (images - channel_means) / channel_stddevs

    if exclude_classes is not None:
        for exclude_class in exclude_classes:
            mask = (labels != exclude_class).reshape(-1)
            labels = labels[mask]
            images = images[mask]

    return images, labels, channel_means, channel_stddevs


def make_federated_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    user_dataset_len_sampler: Callable,
    alpha: float = 0.1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create a federated dataset from the CIFAR10 dataset.

    Users are created as proposed by Hsu et al. https://arxiv.org/abs/1909.06335,
    by sampling each user's class distribution from Dir(0.1).
    """
    # Randomise the order of the data.
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    # Mapping from user_id to indices of images and labels for each user.
    users_to_indices = partition_by_dirichlet_class_distribution(
        labels, alpha, user_dataset_len_sampler
    )
    images = numpy_to_tensor(images)
    labels = numpy_to_tensor(labels)
    # Mapping from user_id to (images, labels) for each user.
    users_to_data = [(images[indices], labels[indices]) for indices in users_to_indices]
    return users_to_data


def make_iid_federated_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    user_dataset_len_sampler: Callable,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Create a federated dataset with IID users from the CIFAR10 dataset.

    Users are created by first sampling the dataset length from
    ``user_dataset_len_sampler`` and then sampling the datapoints IID.
    """
    # Randomise the order of the data.
    data_order = np.random.permutation(len(images))
    images, labels = images[data_order], labels[data_order]
    # Make the numpy arrays into tensors.
    images = torch.tensor(images)
    labels = torch.tensor(labels)
    # Mapping from user_id to (images, labels) for each user.
    users_to_data: dict = {}
    # Build the partitions for each user
    start_ix = 0
    while True:
        dataset_len = user_dataset_len_sampler()
        user_slice = slice(start_ix, start_ix + dataset_len)
        users_to_data[len(users_to_data)] = (images[user_slice], labels[user_slice])
        start_ix += dataset_len
        if start_ix >= len(images):
            break
    return users_to_data


def make_cifar10_datasets(
    data_dir: Path,
    user_dataset_len_sampler: Callable,
    alpha: float = 0.1,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Create the CIFAR10 federated datasets with non-IID users."""
    (train_images, train_labels, channel_means, channel_stddevs) = load_and_preprocess(
        data_dir / "cifar10_train.p"
    )
    val_images, val_labels, _, _ = load_and_preprocess(
        data_dir / "cifar10_test.p",
        channel_means,
        channel_stddevs,
    )

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_federated_dataset(
        train_images,
        train_labels,
        user_dataset_len_sampler,
        alpha,
    )
    val_federated_dataset = make_federated_dataset(
        val_images,
        val_labels,
        user_dataset_len_sampler,
        alpha,
    )

    return training_federated_dataset, val_federated_dataset


def make_cifar10_iid_datasets(
    data_dir: Path, user_dataset_len_sampler: Callable
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]]
]:
    """Create the CIFAR10 federated datasets with IID users."""
    train_images, train_labels, channel_means, channel_stddevs = load_and_preprocess(
        data_dir / "cifar10_train.p"
    )
    val_images, val_labels, _, _ = load_and_preprocess(
        data_dir / "cifar10_test.p",
        channel_means,
        channel_stddevs,
    )

    # create artificial federated training and val datasets
    # from central training and val data.
    training_federated_dataset = make_iid_federated_dataset(
        train_images, train_labels, user_dataset_len_sampler
    )
    val_federated_dataset = make_iid_federated_dataset(
        val_images, val_labels, user_dataset_len_sampler
    )

    return training_federated_dataset, val_federated_dataset


def static_user_datalen_sampler() -> int:
    """Sample 50 samples per user."""
    return 50


class Cifar10(Dataset):
    """Cifar10 dataset object."""

    def __init__(
        self,
        data_dir: Path,
        client_id: int | None = None,
        dataset: str = "train",
        user_dataset_len_sampler: Callable = static_user_datalen_sampler,
        alpha: float | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.client_id = client_id
        self.dataset = dataset
        self.user_dataset_len_sampler = user_dataset_len_sampler
        self.alpha = alpha

        if self.dataset == "train":
            (self.images, self.labels, _, _) = load_and_preprocess(
                self.data_dir / "cifar10_train.p"
            )
        else:
            self.images, self.labels, _, _ = load_and_preprocess(
                self.data_dir / "cifar10_test.p",
            )
        self.data: (
            list[tuple[np.ndarray, np.ndarray]]
            | dict[int, tuple[np.ndarray, np.ndarray]]
        )
        if self.alpha:
            self.data = make_federated_dataset(
                self.images,
                self.labels,
                self.user_dataset_len_sampler,
                self.alpha,
            )
        else:
            self.data = make_iid_federated_dataset(
                self.images, self.labels, self.user_dataset_len_sampler
            )

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.client_id is not None:
            return len(self.data[self.client_id][0])
        else:
            return len(self.data[0])

    def __getitem__(self, index: int) -> tuple[Any, Any | int]:
        """Return the sample at current `index`."""
        if self.client_id is not None:
            return (
                self.data[self.client_id][0][index],
                self.data[self.client_id][1][index][0],
            )
        else:
            return self.data[0][index], self.data[1][index][0]


if __name__ == "__main__":
    # Example usage
    data_dir = Path("/datasets/cifar10")
    alpha = 0.1
    user_dataset_len_sampler = parse_draw_num_datapoints_per_user(
        "constant",
        50,
    )

    start_time = time.time()
    training_fed_dataset, val_fed_dataset = make_cifar10_datasets(
        data_dir, user_dataset_len_sampler, alpha
    )
    log(INFO, f"Time to create datasets: {time.time() - start_time:.2f} seconds")
    log(INFO, training_fed_dataset[0][0].shape)
    log(INFO, len(training_fed_dataset[0][0]))
    log(INFO, training_fed_dataset[0][1].shape)
    log(INFO, len(training_fed_dataset[0][1]))
    log(INFO, training_fed_dataset[1][0].shape)
    log(INFO, len(training_fed_dataset[1][0]))
    log(INFO, training_fed_dataset[1][1].shape)
    log(INFO, len(training_fed_dataset[1][1]))
    training_fed_dataset_iid, val_fed_dataset_iid = make_cifar10_iid_datasets(
        data_dir, user_dataset_len_sampler
    )
    start_time = time.time()
    log(INFO, f"Time to create iid datasets: {time.time() - start_time:.2f} seconds")
    log(INFO, training_fed_dataset[0][0].shape)
    log(INFO, len(training_fed_dataset[0][0]))
    log(INFO, training_fed_dataset[0][1].shape)
    log(INFO, len(training_fed_dataset[0][1]))
    log(INFO, training_fed_dataset[1][0].shape)
    log(INFO, len(training_fed_dataset[1][0]))
    log(INFO, training_fed_dataset[1][1].shape)
    log(INFO, len(training_fed_dataset[1][1]))
