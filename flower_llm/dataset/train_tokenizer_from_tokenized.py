"""
Train Tokenizer from Tokenized Streaming Text Datasets.

This module provides functionality to train a tokenizer on tokenized streaming text
datasets. It processes specified splits of a dataset, trains a
SentencePieceUnigramTokenizer on the tokenized streaming text dataset, converts it to a
fast tokenizer, and saves the trained tokenizer to the specified output directory. The
tokenizer is also pushed to the Hugging Face Hub.

Functions
---------
- parse_args() -> Namespace
    Parse command-line arguments for the tokenizer training script.
- main(args: Namespace) -> None
    Train a tokenizer on tokenized decoded streaming text datasets and save the trained
    tokenizer.

Usage
-----
To use this script, run it from the command line with the appropriate arguments:
    python train_tokenizer_from_tokenized.py \
        --names <config1> <config2> --splits <split1> <split2> \
        --decode_tokenizer <tokenizer_path> --decode_tokenizer_kwargs <kwargs> \
        --streams_config_file <streams_config> --dataset_config_file <dataset_config> \
        --max_length <max_length> --vocab_size <vocab_size> \
        --special_tokens <token1> <token2> --truncate_num_samples <num_samples> \
        --num_workers <num_workers> --output_root_dir <output_dir> \
        --s3_endpoint_url <s3_url>

Example
-------
    python train_tokenizer_from_tokenized.py --names wikipedia arxiv --splits train \
        --decode_tokenizer path/to/decode_tokenizer --decode_tokenizer_kwargs '{}' \
        --streams_config_file path/to/streams_config.yaml \
        --dataset_config_file path/to/dataset_config.yaml --max_length 512 \
        --vocab_size 32000 --special_tokens "<unk>" "</s>" --truncate_num_samples 1000 \
            --num_workers 4 --output_root_dir ./output --s3_endpoint_url https://s3.endpoint.url

Dependencies
------------
- argparse
- dataclasses
- datetime
- json
- logging
- os
- pathlib
- tempfile
- time
- uuid
- flwr.common.logger
- llmfoundry
- omegaconf
- tokenizers
- transformers
- flower_llm
"""

from argparse import ArgumentParser, Namespace
from dataclasses import asdict
from datetime import datetime
import json
from logging import INFO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from uuid import uuid4
from flwr.common.logger import log

from llmfoundry import StreamingTextDataset
from omegaconf import OmegaConf
from tokenizers.implementations import SentencePieceUnigramTokenizer
import torch
from transformers import PreTrainedTokenizerFast


from llmfoundry.utils.builders import (
    build_tokenizer,
)
from llmfoundry.data.text_data import build_streams

from flower_llm.clients.llm_config_functions import StreamDict
from flower_llm.dataset.constants import THE_PILE_CLIENT_MAP
from flower_llm.dataset.utils import (
    build_dataloader,
    generate_samples_tokenized_streaming_text_dataset,
)


def parse_args() -> Namespace:
    """
    Parse command-line arguments for the tokenizer training script.

    This function sets up an argument parser to receive various parameters for training
    a tokenizer on tokenized streaming text datasets. It parses the command-line
    arguments and returns them as a Namespace object.

    Parameters
    ----------
    None

    Returns
    -------
    Namespace
        An object containing the parsed command-line arguments:
        - decode_tokenizer (str): Path or name of the decode tokenizer to use. Default
         is "EleutherAI/gpt-neox-20b".
        - decode_tokenizer_kwargs (dict): Additional keyword arguments for the decode
         tokenizer.
        - s3_endpoint_url (str): S3 endpoint URL for remote storage. Default is "http://128.232.115.0:9000".
        - max_length (int): Maximum length of the sequences. Default is 2048.
        - names (set[str]): Set of dataset configuration names (e.g., "wikipedia" or
         "arxiv").
        - splits (set[str]): Set of dataset splits to process (e.g., "train" or
         "validation").
        - streams_config_file (str): Path to the streams configuration file. Default is
         "/nfs-share/ls985/projects/flower_llm/flower_llm/conf/dataset/streams/the_pile_8_clients.yaml".
        - dataset_config_file (str): Path to the dataset configuration file. Default is
         "/nfs-share/ls985/projects/flower_llm/flower_llm/conf/dataset/fed-the_pile.yaml".
        - special_tokens (list[str]): List of special tokens (e.g., "<unk>" or "</s>").
        - output_root_dir (str): Directory where the output will be saved.
        - vocab_size (int): Size of the vocabulary. Default is 32000.
        - truncate_num_samples (int): Number of samples to truncate to. Default is -1
         (no truncation).
        - num_workers (int): Number of worker processes to use for data loading.

    Raises
    ------
    SystemExit
        If required arguments are not provided or if there is an error in parsing
        arguments.

    Example
    -------
    >>> args = parse_args()
    >>> print(args.decode_tokenizer)
    >>> print(args.decode_tokenizer_kwargs)
    >>> print(args.s3_endpoint_url)
    >>> print(args.max_length)
    >>> print(args.names)
    >>> print(args.splits)
    >>> print(args.streams_config_file)
    >>> print(args.dataset_config_file)
    >>> print(args.special_tokens)
    >>> print(args.output_root_dir)
    >>> print(args.vocab_size)
    >>> print(args.truncate_num_samples)
    >>> print(args.num_workers)
    """
    parser = ArgumentParser(
        description=("Train Tokenizer from Tokenized Streaming Text Datasets.")
    )
    parser.add_argument(
        "--decode_tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        required=False,
    )
    parser.add_argument("--decode_tokenizer_kwargs", type=str, required=False)
    parser.add_argument(
        "--s3_endpoint_url",
        type=str,
        default="http://128.232.115.0:9000",
        required=False,
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--names",
        nargs="+",
        default=None,
        help='E.g. "wikipedia" or "arxiv"',
        required=True,
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help='E.g. "train" or "validation"',
        required=True,
    )
    parser.add_argument(
        "--streams_config_file",
        type=str,
        default="/nfs-share/ls985/projects/flower_llm/flower_llm/conf/dataset/streams/the_pile_16_clients.yaml",
        required=False,
    )
    parser.add_argument(
        "--dataset_config_file",
        type=str,
        default="/nfs-share/ls985/projects/flower_llm/flower_llm/conf/dataset/fed-the_pile.yaml",
        required=False,
    )
    parser.add_argument(
        "--special_tokens", nargs="+", default=None, help='E.g. "<unk>" or "</s>"'
    )
    parser.add_argument("--output_root_dir", type=str, required=False)
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--truncate_num_samples", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    parsed = parser.parse_args()

    # Parse decode_tokenizer_kwargs
    if parsed.decode_tokenizer_kwargs is not None:
        parsed.decode_tokenizer_kwargs = json.loads(parsed.decode_tokenizer_kwargs)
    else:
        parsed.decode_tokenizer_kwargs = {}
    parsed.decode_tokenizer_kwargs["model_max_length"] = parsed.max_length
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
    Train a tokenizer on tokenized decoded datasets and save the trained tokenizer.

    This function processes specified splits of a dataset, trains a
    SentencePieceUnigramTokenizer on the tokenized streaming text dataset, converts it
    to a fast tokenizer, and saves the trained tokenizer to the specified output
    directory. The tokenizer is also pushed to the Hugging Face Hub.

    Parameters
    ----------
    args : Namespace
        The arguments for the function, expected to have the following attributes:
        - names (list[str]): List of dataset configuration names.
        - splits (list[str]): List of dataset splits to process (e.g., "train" or
         "val").
        - decode_tokenizer (str): Path or name of the decode tokenizer to use for
         decoding the dataset.
        - decode_tokenizer_kwargs (dict): Additional keyword arguments for the decode
         tokenizer.
        - streams_config_file (str): Path to the streams configuration file.
        - dataset_config_file (str): Path to the dataset configuration file.
        - max_length (int): Maximum length of the sequences.
        - vocab_size (int): Size of the vocabulary.
        - special_tokens (list[str]): List of special tokens (e.g., "<unk>" or "</s>").
        - truncate_num_samples (int): Number of samples to truncate to.
        - num_workers (int): Number of worker processes to use for data loading.
        - output_root_dir (str): Directory where the output will be saved.
        - s3_endpoint_url (str): S3 endpoint URL for remote storage.

    Returns
    -------
    None

    Example
    -------
    >>> from argparse import Namespace
    >>> args = Namespace(
    ...     names=["config1", "config2"],
    ...     splits=["train", "val"],
    ...     decode_tokenizer="path/to/decode_tokenizer",
    ...     decode_tokenizer_kwargs={},
    ...     streams_config_file="path/to/streams_config.yaml",
    ...     dataset_config_file="path/to/dataset_config.yaml",
    ...     max_length=512,
    ...     vocab_size=32000,
    ...     special_tokens=["<unk>", "</s>"],
    ...     truncate_num_samples=1000,
    ...     num_workers=4,
    ...     output_root_dir="output_dir",
    ...     s3_endpoint_url="https://s3.endpoint.url"
    ... )
    >>> main(args)
    """
    torch.multiprocessing.set_sharing_strategy("file_system")
    general_temp_dir = TemporaryDirectory()
    os.environ["TMPDIR"] = general_temp_dir.name
    os.environ["S3_ENDPOINT_URL"] = args.s3_endpoint_url
    os.environ["RUN_UUID"] = str(uuid4())
    for name in args.names:
        assert name in THE_PILE_CLIENT_MAP, f"Name {name} not in THE_PILE_CLIENT_MAP"
        for split in args.splits:
            assert split in {"train", "val"}, f"Split {split} not in {'train', 'val'}"
            start_time = time.time()
            temp_dir = TemporaryDirectory()
            log(
                INFO,
                "Processing split %s with configuration %s",
                split,
                name,
            )
            # Build the tokenizer to use for decoding the dataset
            decode_tokenizer = build_tokenizer(
                args.decode_tokenizer, args.decode_tokenizer_kwargs
            )
            # Load the dataset and streams configuration files
            streams_config = OmegaConf.load(args.streams_config_file)
            # Set some useful defaults
            dataset_config = OmegaConf.load(args.dataset_config_file)
            split_config = dataset_config[split]
            split_config.max_seq_len = args.max_length
            split_config.streams = streams_config
            split_config.root_local = temp_dir.name
            split_config.shuffle = False
            split_config.cache_limit = "10gb"
            # Retrieve the streams for the client
            selected_stream = streams_config[THE_PILE_CLIENT_MAP[name]][
                "client_streams"
            ]
            # Set streams dictionary for the train loader
            actual_streams = {
                key: StreamDict(**value) for key, value in selected_stream.items()
            }
            # Get the root path for remote and local data
            root_remote = split_config.pop("root_remote", "")
            root_remote = root_remote + "/" if root_remote else root_remote
            root_local = split_config.pop("root_local", "")
            root_local = root_local + "/" if root_local else root_local
            # Propagate the split and the remote and local paths to each stream
            for stream in actual_streams.values():
                # Set the split, remote, and local paths
                stream.split = split or stream.split
                if root_local:
                    stream.local = (
                        root_local + stream.local if stream.local else root_local
                    )
                if root_remote:
                    stream.remote = (
                        root_remote + stream.remote if stream.remote else root_remote
                    )
                # Remove potential trailing slashes
                stream.local = (
                    stream.local.rstrip("/") if stream.local else stream.local
                )
                stream.remote = (
                    stream.remote.rstrip("/") if stream.remote else stream.remote
                )
            # Convert the streams to dictionaries
            streams_dict = {
                stream_name: asdict(stream)
                for stream_name, stream in actual_streams.items()
            }
            # Retrieve only the first element in the dictionary
            stream_name, first_element = next(iter(streams_dict.items()))
            # Construct the streaming dataset
            streaming_text_dataset = StreamingTextDataset(
                tokenizer=decode_tokenizer,
                max_seq_len=args.max_length,
                streams=build_streams({stream_name: first_element}),
                batch_size=512,
            )
            # Build a batched dataloader for streaming the HF dataset in batches so that
            # we can actually take advantage of multiprocessing and pre-fetching
            loader = build_dataloader(
                dataset=streaming_text_dataset,
                batch_size=512,
                num_workers=args.num_workers,
            )
            # Create a tokenizer object to be trained
            tokenizer = SentencePieceUnigramTokenizer()
            log(INFO, "Training tokenizer %s", tokenizer)
            # Train the tokenizer
            tokenizer.train_from_iterator(
                iterator=(
                    generate_samples_tokenized_streaming_text_dataset(
                        loader,
                        decode_tokenizer,
                        truncate_num_samples=args.truncate_num_samples,
                    )
                ),
                vocab_size=args.vocab_size,
                show_progress=True,
                special_tokens=args.special_tokens,
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
                f"_d-pile_n-{name}_s-{split}"
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
