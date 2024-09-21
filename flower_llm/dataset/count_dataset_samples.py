"""
Dataset Sample Counting Script.

This module provides functionality to count the number of samples and calculate the
average number of characters per sample in specified splits of a Hugging Face dataset.
It uses command-line arguments to specify the dataset path, configuration names, and
splits to process.

Functions
---------
- parse_args() -> Namespace
    Parses command-line arguments for the dataset sample counting script.
- main(args: Namespace) -> None
    Processes and counts samples in specified splits of a Hugging Face dataset.

Usage
-----
To use this script, run it from the command line with the appropriate arguments:
    python count_dataset_samples.py \
        --path <dataset_path> \
        --names <config1> <config2> \
        --splits <split1> <split2>

Example
-------
    python count_dataset_samples.py \
        --path dataset_path \
        --names config1 config2 \
        --splits train test

Dependencies
------------
- argparse
- logging
- tempfile
- flwr.common.logger
- tqdm
- datasets (Hugging Face)
- flower_llm.dataset.utils (build_dataloader, NoConcatDatasetString)
"""

from argparse import ArgumentParser, Namespace
from logging import INFO
from tempfile import TemporaryDirectory
from flwr.common.logger import log
from tqdm import tqdm

import datasets as hf_datasets

from flower_llm.dataset.utils import build_dataloader, NoConcatDatasetString


def parse_args() -> Namespace:
    """
    Parse command-line arguments for the dataset sample counting script.

    This function sets up an argument parser to receive a dataset with a set of
    configurations and counts the number of samples and average characters per sample.
    It parses the command-line arguments and returns them as a Namespace object.

    Parameters
    ----------
    None

    Returns
    -------
    Namespace
        An object containing the parsed command-line arguments:
        - path (str): Path or name of the dataset.
        - names (set[str] | None): Set of dataset configuration names, or None.
        - splits (set[str] | None): Set of dataset splits to process, or None.

    Notes
    -----
    - The `--path` argument is required and specifies the path or name of the dataset.
    - The `--names` argument is optional and specifies the dataset configuration names.
      It can be a list of names, e.g., "all" or "en".
    - The `--splits` argument is optional and specifies the dataset splits to process.
      It can be a list of splits, e.g., "train" or "validation".
    - The function logs the parsed arguments at the INFO level.

    Example
    -------
    >>> args = parse_args()
    >>> print(args.path)
    >>> print(args.names)
    >>> print(args.splits)
    """
    parser = ArgumentParser(
        description=(
            "Receive a dataset with a set of configurations and counts the number of "
            "samples and average char per sample."
        )
    )
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--names", nargs="+", default=None, help='E.g. "all" or "en"')
    parser.add_argument(
        "--splits", nargs="+", default=None, help='E.g. "train" or "validation"'
    )
    parsed = parser.parse_args()
    if parsed.names is not None:
        parsed.names = set(parsed.names)
    if parsed.splits is not None:
        parsed.splits = set(parsed.splits)
    log(INFO, "Arguments parsed: %s", parsed)
    return parsed


def main(args: Namespace) -> None:
    """
    Process and count samples in specified splits of a Hugging Face (HF) dataset Hub.

    This function processes specified splits of a HF dataset Hub, counts the number
    of samples, and calculates the average number of characters per sample. It logs the
    progress and results at each step.

    Parameters
    ----------
    args : Namespace
        The arguments for the function, expected to have the following attributes:
        - names (list[str]): List of dataset configuration names.
        - splits (list[str]): List of dataset splits to process.
        - path (str): Path or name of the dataset.

    Returns
    -------
    None

    Notes
    -----
    - The function uses a temporary directory for caching the dataset.
    - The dataset is streamed and processed in batches to take advantage of
        multiprocessing (when the dataset possesses multiple shards).
    - The function logs the number of samples and the average number of characters per
        sample.

    Example
    -------
    >>> from argparse import Namespace
    >>> args = Namespace(names=['en'], splits=['train'], path='allenai/c4')
    >>> main(args)
    """
    for name in args.names:
        for split in args.splits:
            temp_dir = TemporaryDirectory()
            log(
                INFO,
                "Processing split %s of dataset %s with configuration %s",
                split,
                args.path,
                name,
            )
            n_chars_per_sample = 0
            hf_dataset = NoConcatDatasetString(
                hf_datasets.load_dataset(  # type: ignore[attr-defined]
                    path=args.path,  # Path or name of the dataset
                    name=name,  # Defining the name of the dataset configuration.
                    data_dir=None,  # We are getting it from the Hub
                    split=split,  # Which split to use
                    streaming=True,
                    cache_dir=temp_dir.name,
                    keep_in_memory=False,
                    save_infos=False,
                    trust_remote_code=True,
                )
            )
            # Build a batched dataloader for streaming the HF dataset in batches so that
            # we can actually take advantage of multiprocessing
            loader = build_dataloader(
                dataset=hf_dataset, batch_size=1, num_workers=None  # type: ignore[reportArgumentType]
            )
            log(
                INFO,
                "Looping over  %s-%s-%s to obtain general statistics.",
                args.path,
                name,
                split,
            )
            n_samples = 0
            for batch in tqdm(loader):
                n_samples += len(batch)
                for sample in batch:
                    n_chars_per_sample += len(sample)
            log(INFO, "Number of samples: %d", n_samples)
            log(
                INFO,
                "Number of characters per sample: %d",
                n_chars_per_sample // n_samples,
            )


if __name__ == "__main__":
    main(parse_args())
