"""
Tokenizer Training Script.

This module provides functionality to train a tokenizer on specified dataset splits
from the Hugging Face Hub. It supports various configurations and special tokens,
and saves the trained tokenizer to a specified output directory.

Functions
---------
- parse_args() -> Namespace
    Parse command-line arguments for the tokenizer training script.
- main(args: Namespace) -> None
    Train a tokenizer on specified dataset splits and save the trained tokenizer.

Usage
-----
To use this script, run it from the command line with the appropriate arguments:
    python train_tokenizer.py \
        --path <dataset_path> \
        --names <config1> <config2> \
        --splits <split1> <split2> \
        --output_root_dir <output_dir>

Example
-------
    python train_tokenizer.py \
        --path allenai/c4 \
        --names en \
        --splits train validation \
        --output_root_dir ./output

Dependencies
------------
- argparse
- datetime
- logging
- pathlib
- tempfile
- flwr.common.logger
- datasets (Hugging Face)
- tokenizers
- transformers
- flower_llm.dataset.utils (NoConcatDatasetString, generate_samples)
"""

from argparse import ArgumentParser, Namespace
from datetime import datetime
from logging import INFO
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from flwr.common.logger import log

import datasets as hf_datasets
from tokenizers.implementations import SentencePieceUnigramTokenizer
from transformers import PreTrainedTokenizerFast

from flower_llm.dataset.constants import CONSTANTS, DataSplitConstants
from flower_llm.dataset.utils import (
    NoConcatDatasetString,
    build_dataloader,
    generate_samples_from_hf_dataloader,
)


def parse_args() -> Namespace:
    """
    Parse command-line arguments for the tokenizer training script.

    This function sets up an argument parser to receive a dataset with a set of
    configurations, special tokens, and other parameters required for training a
    tokenizer. It parses the command-line arguments and returns them as a Namespace
    object.

    Parameters
    ----------
    None

    Returns
    -------
    Namespace
        An object containing the parsed command-line arguments:
        - path (str): Path or name of the dataset (e.g., "allenai/c4").
        - names (set[str] | None): Set of dataset configuration names (e.g., "en" or
            "it"), or None if not provided.
        - splits (set[str] | None): Set of dataset splits to process (e.g., "train" or
            "validation"), or None if not provided.
        - special_tokens (list[str] | None): List of special tokens (e.g., "<unk>" or
            "</s>"), or None if not provided.
        - output_root_dir (str): Directory where the output will be saved.
        - max_length (int): Maximum length of the sequences. Default is 2048.
        - vocab_size (int): Size of the vocabulary. Default is 32000.
        - truncate_num_samples (int): Number of samples to truncate to. Default is -1
            (no truncation).
        - num_workers (int | None): Number of worker processes to use for data loading.
         Default is None.

    Notes
    -----
    - The `--path` argument is required and specifies the path or name of the dataset.
    - The `--output_root_dir` argument is required and specifies the directory where
        the output will be saved.
    - The `--names`, `--splits`, and `--special_tokens` arguments are optional and can
        be used to specify dataset configurations, splits, and special tokens,
        respectively.
    - The function logs the parsed arguments at the INFO level.

    Example
    -------
    >>> args = parse_args()
    >>> print(args.path)
    >>> print(args.names)
    >>> print(args.splits)
    >>> print(args.special_tokens)
    >>> print(args.output_root_dir)
    >>> print(args.max_length)
    >>> print(args.vocab_size)
    >>> print(args.num_workers)
    >>> print(args.truncate_num_samples)
    """
    parser = ArgumentParser(description=("Tokenizer Training Script."))
    parser.add_argument("--path", type=str, required=True, help='E.g. "allenai/c4"')
    parser.add_argument("--names", nargs="+", default=None, help='E.g. "en" or "it"')
    parser.add_argument(
        "--splits", nargs="+", default=None, help='E.g. "train" or "validation"'
    )
    parser.add_argument(
        "--special_tokens", nargs="+", default=None, help='E.g. "<unk>" or "</s>"'
    )
    parser.add_argument("--output_root_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--truncate_num_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    parsed = parser.parse_args()
    if parsed.names is not None:
        parsed.names = set(parsed.names)
    if parsed.splits is not None:
        parsed.splits = set(parsed.splits)
    if parsed.special_tokens is not None:
        parsed.special_tokens = list(set(parsed.special_tokens))
    log(INFO, "Arguments parsed: %s", parsed)
    return parsed


def main(args: Namespace) -> None:
    """
    Train a tokenizer on specified dataset splits and save the trained tokenizer.

    This function processes specified splits of a Hugging Face dataset, trains a
    SentencePieceUnigramTokenizer on the dataset, converts it to a fast tokenizer,
    and saves the trained tokenizer to the specified output directory.

    Parameters
    ----------
    args : Namespace
        The arguments for the function, expected to have the following attributes:
        - path (str): Path or name of the dataset.
        - names (set[str]): Set of dataset configuration names.
        - splits (set[str]): Set of dataset splits to process.
        - special_tokens (list[str]): List of special tokens.
        - output_root_dir (str): Directory where the output will be saved.
        - max_length (int): Maximum length of the sequences.
        - vocab_size (int): Size of the vocabulary.
        - num_workers (int | None): Number of worker processes to use for data loading.
        - truncate_num_samples (int): Number of samples to truncate to.

    Returns
    -------
    None

    Example
    -------
    >>> from argparse import Namespace
    >>> args = Namespace(
    ...     path="dataset_path",
    ...     names={"config1", "config2"},
    ...     splits={"train", "test"},
    ...     special_tokens=["<unk>", "</s>"],
    ...     output_root_dir="output_dir",
    ...     max_length=512,
    ...     num_workers=4,
    ...     vocab_size=32000,
    ...     truncate_num_samples=-1
    ... )
    >>> main(args)
    """
    for name in args.names:
        for split in args.splits:
            start_time = time.time()
            temp_dir = TemporaryDirectory()
            # Retrieve the DataSplitConstants object
            dataset_split_constants: DataSplitConstants | None = None
            if "c4" in args.path:
                dataset_split_constants = CONSTANTS[f"c4_{name}"].splits[split]
            assert dataset_split_constants is not None, (
                "Dataset constants not found for "
                f"{args.path}-{name}-{split}. "
                f"Available constants: {CONSTANTS}"
            )
            log(
                INFO,
                "Processing split %s of dataset %s with configuration %s",
                split,
                args.path,
                name,
            )
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
            log(
                INFO,
                "Training tokenizer on %s samples",
                dataset_split_constants.raw_samples,
            )
            # Build a batched dataloader for streaming the HF dataset in batches so that
            # we can actually take advantage of multiprocessing and pre-fetching
            loader = build_dataloader(
                dataset=hf_dataset, batch_size=512, num_workers=args.num_workers
            )
            # Create a tokenizer object to be trained
            tokenizer = SentencePieceUnigramTokenizer()
            log(INFO, "Training tokenizer %s", tokenizer)
            # Train the tokenizer
            tokenizer.train_from_iterator(
                iterator=(
                    generate_samples_from_hf_dataloader(
                        loader, args.truncate_num_samples
                    )
                ),
                vocab_size=args.vocab_size,
                show_progress=True,
                special_tokens=args.special_tokens,
                length=(
                    dataset_split_constants.raw_samples
                    if args.truncate_num_samples < 0
                    else args.truncate_num_samples
                ),
                unk_token="<unk>",
            )
            log(INFO, "Tokenizer has trained")
            # Convert the tokenizer to a fast tokenizer
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                model_max_length=args.max_length,
                special_tokens=args.special_tokens,
            )
            # Create the filename
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            save_directory = (
                f"tokenizer_{timestamp}_v-{args.vocab_size}_l-{args.max_length}"
                f"_d-{args.path}_n-{name}_s-{split}"
            ).replace("/", "-")
            # Save the tokenizer to the specified path
            tokenizer.save_pretrained(
                save_directory=str(Path(args.output_root_dir) / save_directory)
            )
            tokenizer.push_to_hub(save_directory, private=True)
            log(
                INFO,
                "Tokenizer has been saved to %s",
                (Path(args.output_root_dir) / save_directory),
            )
            log(INFO, "Time taken: %s", time.time() - start_time)


if __name__ == "__main__":
    main(parse_args())
