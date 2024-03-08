"""Module for the helper functions of the FLAIR dataset.

The script has been slightly modified from:
TODO: add link to original script (common.py)
"""

# Copyright © 2023-2024 Apple Inc.

from collections.abc import Callable
import json
from pathlib import Path
import random

import h5py
import numpy as np
import tqdm


def get_multi_hot_targets(
    shape: tuple[int, int],
    h5: h5py.File,
    partition: str,
    use_id: str,
    use_fine_grained_labels: bool,
) -> np.ndarray:
    """Return multi-hot targets for a user."""
    prefix = "fine_grained_labels" if use_fine_grained_labels else "labels"
    row_indices = np.array(h5[f"/{partition}/{use_id}/{prefix}_row"])
    col_indices = np.array(h5[f"/{partition}/{use_id}/{prefix}_col"])
    vec = np.zeros(shape, dtype=np.float32)
    vec[row_indices, col_indices] = 1
    return vec


def get_label_mapping(hdf5_path: Path, use_fine_grained_labels: bool) -> dict[str, int]:
    """
    Get the mapping of labels to indices.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param use_fine_grained_labels:
        Whether to use fine-grained label taxonomy.
    :return:
        A dictionary with label as key and index as value
    """
    with h5py.File(hdf5_path, "r") as h5:
        if use_fine_grained_labels:
            return json.loads(h5["/metadata/fine_grained_label_mapping"][()])
        else:
            return json.loads(h5["/metadata/label_mapping"][()])


def get_training_channel_mean_stddevs(
    hdf5_path: Path, data_fraction: float = 0.1, seed: int | None = None
) -> tuple[list[float], list[float]]:
    """Compute the averaged channel mean and std from training data.

    Following ImageNet implementation from:
    https://github.com/soumith/imagenet-multiGPU.torch/blob/master/donkey.lua
    Technically there is a small privacy leakage if no noise added.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param data_fraction:
        Fraction of users for estimating the statistics.
    :param seed:
        Random seed for sampling a fraction of users.
    :return:
        Two list of three numbers representing channel mean and std.
    """
    num_channels = 3
    mean, std = np.zeros(num_channels), np.zeros(num_channels)
    n = 0
    if seed is not None:
        random.seed(seed)

    with h5py.File(hdf5_path, "r") as h5:
        user_ids = h5["/train"].keys()
        num_users = int(len(user_ids) * data_fraction)

        random.shuffle(user_ids)
        for user_id in tqdm.tqdm(user_ids[:num_users]):
            inputs = np.array(h5[f"/train/{user_id}/images"]) / 255.0
            mean += np.mean(inputs, axis=(1, 2)).sum(axis=0)
            std += np.std(inputs, axis=(1, 2)).sum(axis=0)
            n += len(inputs)

    return (mean / n).tolist(), (std / n).tolist()


def get_channel_mean_stddevs(
    source: str = "imagenet",
    hdf5_path: Path | None = None,
    data_fraction: float | None = None,
    seed: int | None = None,
) -> tuple[list[float], list[float]]:
    """
    Get image channel mean and standard deviations for input transformation.

    :param source:
        A string indicating where the mean and std estimation came from.
        'flair' will estimate the statistics from training data and
        'imagenet' will use the statistics from ImageNet data.
    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param data_fraction:
        Fraction of users for estimating the statistics.
    :param seed:
        Random seed for sampling a fraction of users.
    :return:
        Two list of three numbers representing channel mean and std.
    """
    if source == "flair":
        assert hdf5_path is not None and data_fraction is not None
        return get_training_channel_mean_stddevs(hdf5_path, data_fraction, seed)
    elif source == "imagenet":
        return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def get_user_num_images(hdf5_path: Path, partition: str) -> dict[str, int]:
    """
    Get the number of images per user.

    :param hdf5_path:
        The FLAIR h5py dataset object.
    :param partition:
        Whether it is a "train", "val" or "test" partition.
    :return:
        A dictionary with user IDs as keys and number of images
        as values.
    """
    with h5py.File(hdf5_path, "r") as h5:
        user_num_images = {}
        for key in tqdm.tqdm(
            h5[f"/{partition}"].keys(), desc=f"Creating user_num_images for {partition}"
        ):
            user_num_images[key] = h5[f"/{partition}/{key}/image_ids"].shape[0]
    return user_num_images


def parse_draw_num_datapoints_per_user(
    datapoints_per_user_distribution: str,
    mean_datapoints_per_user: float,
    minimum_num_datapoints_per_user: int = 1,
) -> Callable[[], int]:
    """
    Get a user dataset length sampler.

    :param datapoints_per_user_distribution:
        'constant' or 'poisson'.
    :param mean_datapoints_per_user:
        If 'constant' distribution, this is the value to always return.
        If 'poisson' distribution, this is the mean of the poisson
        distribution.
    :param minimum_num_datapoints_per_user:
        Only accept return values that are at least as large as this argument.
        If rejected, value is resampled until above the threshold, which will
        result in a truncated distribution.
    :return:
        A callable that samples lengths for artificial user datasets
        yet to be created.
    """
    if datapoints_per_user_distribution == "constant":
        assert minimum_num_datapoints_per_user < mean_datapoints_per_user

        def _draw_num_datapoints_per_user() -> int:
            return int(mean_datapoints_per_user)

        draw_num_datapoints_per_user = _draw_num_datapoints_per_user
    else:
        assert datapoints_per_user_distribution == "poisson"

        def _draw_truncated_poisson() -> int:
            while True:
                num_datapoints = np.random.poisson(mean_datapoints_per_user)
                # Try again if less than minimum specified.
                if num_datapoints >= minimum_num_datapoints_per_user:
                    return num_datapoints

        draw_num_datapoints_per_user = _draw_truncated_poisson
    return draw_num_datapoints_per_user
