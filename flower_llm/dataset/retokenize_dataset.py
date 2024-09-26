"""
Retokenize Dataset and Convert to MDS Format.

This module provides functionality to download a dataset from the Hugging Face Hub,
retokenize it using specified decode and encode tokenizers, and convert it into MDS
(Multi-Document Summarization) format. The resulting MDS files can be saved to a
specified remote bucket, with support for multiple clients.

Functions
---------
- parse_args() -> Namespace
    Parse command-line arguments for retokenizing and converting a dataset into MDS
    format.
- main(args: Namespace) -> None
    Retokenize a dataset from the Hugging Face Hub and save it in MDS format.

Usage
-----
To use this script, run it from the command line with the appropriate arguments:
    python retokenize_dataset.py --name <dataset_name> --splits <split1> <split2> \
        --decode_tokenizer <decode_tokenizer_path> \
        --decode_tokenizer_kwargs <decode_kwargs> \
        --encode_tokenizer <encode_tokenizer_path> \
        --encode_tokenizer_kwargs <encode_kwargs> \
        --streams_config_file <streams_config> --dataset_config_file <dataset_config> \
        --max_length <max_length> --concat_tokens <num_tokens> --bos_text <bos_text> \
        --eos_text <eos_text> --no_wrap --num_workers <num_workers> \
        --remote_bucket <remote_bucket> --compression <compression_method> \
        --s3_endpoint_url <s3_url>

Example
-------
    python retokenize_dataset.py --name wikipedia --splits train val \
        --decode_tokenizer path/to/decode_tokenizer --decode_tokenizer_kwargs '{}' \
        --encode_tokenizer path/to/encode_tokenizer --encode_tokenizer_kwargs '{}' \
        --streams_config_file path/to/streams_config.yaml \
        --dataset_config_file path/to/dataset_config.yaml --max_length 512 \
        --concat_tokens 2048 --bos_text "<s>" --eos_text "</s>" --no_wrap \
        --num_workers 4 --remote_bucket s3://mybucket --compression zstd \
        --s3_endpoint_url https://s3.endpoint.url

Dependencies
------------
- argparse
- dataclasses
- json
- logging
- os
- tempfile
- uuid
- flwr.common.logger
- llmfoundry
- omegaconf
- streaming
- torch
- tqdm
- flower_llm
"""

from dataclasses import asdict
import json
from argparse import ArgumentParser, Namespace
from logging import INFO
import os
from tempfile import TemporaryDirectory
from uuid import uuid4

from flwr.common.logger import log
from llmfoundry import StreamingTextDataset
from llmfoundry.utils.builders import build_tokenizer
from omegaconf import OmegaConf
from streaming import MDSWriter
import torch
from tqdm import tqdm
from llmfoundry.data.text_data import build_streams
from flower_llm.clients.llm_config_functions import StreamDict

from flower_llm.dataset.constants import THE_PILE_CLIENT_NAMING_MAP, THE_PILE_CLIENT_MAP
from flower_llm.dataset.utils import (
    build_dataloader,
    generate_samples_retokenized_streaming_text_dataset,
)


def parse_args() -> Namespace:
    """
    Parse command-line arguments for retokenizing and converting a dataset into MDS.

    This function sets up an argument parser to receive various parameters for
    downloading a dataset from the Hugging Face Hub, retokenizing it using specified
    decode and encode tokenizers, and converting it into MDS format. It parses the
    command-line arguments and returns them as a Namespace object.

    Parameters
    ----------
    None

    Returns
    -------
    Namespace
        An object containing the parsed command-line arguments:
        - name (str): Name of the dataset configuration (e.g., "wikipedia" or "arxiv").
        - splits (set[str]): Set of dataset splits to process (e.g., "train" or "val").
        - streams_config_file (str): Path to the streams configuration file.
        - dataset_config_file (str): Path to the dataset configuration file.
        - compression (str): Compression method for the output MDS dataset. Default is
         "zstd".
        - concat_tokens (int): Number of tokens to concatenate. Default is 2048.
        - decode_tokenizer (str): Path or name of the tokenizer to use for decoding the
         dataset. Default is "EleutherAI/gpt-neox-20b".
        - max_length (int): Maximum length of the sequences. Default is 2048.
        - s3_endpoint_url (str): S3 endpoint URL for remote storage. Default is "http://128.232.115.0:9000".
        - decode_tokenizer_kwargs (dict): Additional keyword arguments for the decode
         tokenizer.
        - encode_tokenizer (str): Path or name of the tokenizer to use for encoding the
         dataset.
        - encode_tokenizer_kwargs (dict): Additional keyword arguments for the encode
         tokenizer.
        - bos_text (str): Text representing the Beginning of Sequence token. Default is
         an empty string.
        - eos_text (str): Text representing the End of Sequence token. Default is
         "</s>".
        - pad_text (str): Text representing the Padding token. Default is "<s>".
        - no_wrap (bool): Whether to disable wrapping of tokens. Default is False.
        - num_workers (int): Number of worker processes to use for data loading.
        - remote_bucket (str): Name of the remote bucket to upload the files to. Default
         is "s3://iclr2025datasets".

    Raises
    ------
    SystemExit
        If required arguments are not provided or if there is an error in parsing
        arguments.

    Example
    -------
    >>> args = parse_args()
    >>> print(args.name)
    >>> print(args.splits)
    >>> print(args.streams_config_file)
    >>> print(args.dataset_config_file)
    >>> print(args.compression)
    >>> print(args.concat_tokens)
    >>> print(args.decode_tokenizer)
    >>> print(args.s3_endpoint_url)
    >>> print(args.max_length)
    >>> print(args.decode_tokenizer_kwargs)
    >>> print(args.encode_tokenizer)
    >>> print(args.encode_tokenizer_kwargs)
    >>> print(args.bos_text)
    >>> print(args.eos_text)
    >>> print(args.pad_text)
    >>> print(args.no_wrap)
    >>> print(args.num_workers)
    >>> print(args.remote_bucket)
    """
    parser = ArgumentParser(
        description=(
            "Convert dataset into MDS format, optionally concatenating and tokenizing."
        )
    )
    # Parameters for downloading the dataset from HF Hub
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='E.g. "wikipedia" or "arxiv"',
        required=True,
    )
    parser.add_argument(
        "--splits", nargs="+", default=None, help='E.g. "train" or "val"'
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
    # Parameters to creating the output MDS dataset
    parser.add_argument("--compression", type=str, default="zstd")
    # Parameters for the tokenization and (potentially) concatenation
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
        default=2048,
    )
    parser.add_argument(
        "--decode_tokenizer",
        type=str,
        default="EleutherAI/gpt-neox-20b",
        required=False,
    )
    parser.add_argument(
        "--s3_endpoint_url",
        type=str,
        default="http://128.232.115.0:9000",
        required=False,
    )
    parser.add_argument("--decode_tokenizer_kwargs", type=str, required=False)
    parser.add_argument("--encode_tokenizer", type=str, required=True)
    parser.add_argument("--encode_tokenizer_kwargs", type=str, required=False)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default="</s>")
    parser.add_argument("--pad_text", type=str, required=False, default="<s>")
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    # Arguments to use our S3-stored dataset when concatenating tokens
    parser.add_argument(
        "--remote_bucket",
        type=str,
        default="s3://iclr2025datasets",
        help="Name of the remote bucket to upload the files to",
    )

    # Parse arguments
    parsed = parser.parse_args()

    # Parse decode_tokenizer_kwargs
    if parsed.decode_tokenizer_kwargs is not None:
        parsed.decode_tokenizer_kwargs = json.loads(parsed.decode_tokenizer_kwargs)
    else:
        parsed.decode_tokenizer_kwargs = {}
    # Parse encode_tokenizer_kwargs
    if parsed.encode_tokenizer_kwargs is not None:
        parsed.encode_tokenizer_kwargs = json.loads(parsed.encode_tokenizer_kwargs)
    else:
        parsed.encode_tokenizer_kwargs = {}

    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.encode_tokenizer is None
    ):
        parser.error(
            "When setting --concat_tokens, you must specify a --encode_tokenizer"
        )

    # Change BOS/EOS/pad to strings if they are None
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    if parsed.pad_text is None:
        parsed.pad_text = ""
    # Add BOS/EOS/pad tokens to encode_tokenizer_kwargs
    parsed.encode_tokenizer_kwargs["bos_token"] = parsed.bos_text
    parsed.encode_tokenizer_kwargs["eos_token"] = parsed.eos_text
    parsed.encode_tokenizer_kwargs["pad_token"] = parsed.pad_text

    # Parse splits
    if parsed.splits is not None:
        parsed.splits = set(parsed.splits)
        log(INFO, f"Converting splits: {parsed.splits}")
    return parsed


def main(args: Namespace) -> None:
    """
    Retokenize a dataset from the Hugging Face Hub and save it in MDS format.

    This function processes specified splits of a dataset, retokenizes the data using
    specified decode and encode tokenizers, and converts it into MDS format. The
    resulting MDS files are saved to a specified remote bucket, with support for
    multiple clients.

    Parameters
    ----------
    args : Namespace
        The arguments for the function, expected to have the following attributes:
        - name (str): Name of the dataset configuration.
        - splits (list[str]): List of dataset splits to process, e.g., "train" or "val".
        - decode_tokenizer (str): Path or name of the tokenizer to use for decoding the
         dataset.
        - decode_tokenizer_kwargs (dict): Additional keyword arguments for the decode
         tokenizer.
        - encode_tokenizer (str): Path or name of the tokenizer to use for encoding the
         dataset.
        - encode_tokenizer_kwargs (dict): Additional keyword arguments for the encode
         tokenizer.
        - streams_config_file (str): Path to the streams configuration file.
        - dataset_config_file (str): Path to the dataset configuration file.
        - max_length (int): Maximum length of the sequences.
        - concat_tokens (int): Number of tokens to concatenate.
        - bos_text (str): Text representing the Beginning of Sequence token.
        - eos_text (str): Text representing the End of Sequence token.
        - no_wrap (bool): Whether to disable wrapping of tokens.
        - num_workers (int): Number of worker processes to use for data loading.
        - remote_bucket (str): Name of the remote bucket to upload the files to.
        - compression (str): Compression method for the output MDS dataset.
        - s3_endpoint_url (str): S3 endpoint URL for remote storage.

    Returns
    -------
    None

    Example
    -------
    >>> from argparse import Namespace
    >>> args = Namespace(
    ...     name="wikipedia",
    ...     splits=["train", "val"],
    ...     decode_tokenizer="path/to/decode_tokenizer",
    ...     decode_tokenizer_kwargs={},
    ...     encode_tokenizer="path/to/encode_tokenizer",
    ...     encode_tokenizer_kwargs={},
    ...     streams_config_file="path/to/streams_config.yaml",
    ...     dataset_config_file="path/to/dataset_config.yaml",
    ...     max_length=512,
    ...     concat_tokens=2048,
    ...     bos_text="<s>",
    ...     eos_text="</s>",
    ...     no_wrap=False,
    ...     num_workers=4,
    ...     remote_bucket="s3://mybucket",
    ...     compression="zstd",
    ...     s3_endpoint_url="https://s3.endpoint.url"
    ... )
    >>> main(args)
    """
    log(INFO, "Arguments received: %s", args)
    torch.multiprocessing.set_sharing_strategy("file_system")
    general_temp_dir = TemporaryDirectory()
    os.environ["TMPDIR"] = general_temp_dir.name
    os.environ["S3_ENDPOINT_URL"] = args.s3_endpoint_url
    os.environ["RUN_UUID"] = str(uuid4())
    # Create temporary directory
    temp_dir = TemporaryDirectory()
    assert (
        args.name in THE_PILE_CLIENT_MAP
    ), f"Name {args.name} not in THE_PILE_CLIENT_MAP"
    # Build decode tokenizer
    decode_tokenizer = build_tokenizer(
        args.decode_tokenizer, args.decode_tokenizer_kwargs
    )
    # Build encode tokenizer
    encode_tokenizer = build_tokenizer(
        args.encode_tokenizer, args.encode_tokenizer_kwargs
    )
    vocab_size = len(encode_tokenizer.get_vocab())
    # We will enforce length because it suppress warnings about sequences too long
    # for the model
    encode_tokenizer.model_max_length = int(1e30)
    # Set the columns for the MDS file
    columns = {"tokens": "ndarray:int32"}
    # Loop over passed splits
    for split_name in args.splits:
        assert split_name in {
            "train",
            "val",
        }, f"Split {split_name} not in {'train', 'val'}"
        # Create temporary directory for caching the dataset
        temp_dir = TemporaryDirectory()
        # Load the dataset and streams configuration files
        streams_config = OmegaConf.load(args.streams_config_file)
        # Set some useful defaults
        dataset_config = OmegaConf.load(args.dataset_config_file)
        split_config = dataset_config[split_name]
        split_config.max_seq_len = args.max_length
        split_config.streams = streams_config
        split_config.root_local = temp_dir.name
        split_config.shuffle = False
        split_config.cache_limit = "10gb"
        # Retrieve the streams for the client
        selected_stream = streams_config[THE_PILE_CLIENT_MAP[args.name]][
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
            stream.split = split_name or stream.split
            if root_local:
                stream.local = root_local + stream.local if stream.local else root_local
            if root_remote:
                stream.remote = (
                    root_remote + stream.remote if stream.remote else root_remote
                )
            # Remove potential trailing slashes
            stream.local = stream.local.rstrip("/") if stream.local else stream.local
            stream.remote = (
                stream.remote.rstrip("/") if stream.remote else stream.remote
            )
        # Convert the streams to dictionaries
        streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}
        # Loop over clients
        for i, (name, stream_inner) in enumerate(streams_dict.items()):
            # Construct the streaming dataset
            streaming_text_dataset = StreamingTextDataset(
                tokenizer=decode_tokenizer,
                max_seq_len=args.max_length,
                streams=build_streams({name: stream_inner}),
                batch_size=512,
            )
            # Build a batched dataloader for streaming the HF dataset in batches so that
            # we can actually take advantage of multiprocessing
            loader = build_dataloader(
                dataset=streaming_text_dataset,
                batch_size=512,
                num_workers=args.num_workers,
            )
            # Build a generator that yields samples from the batched dataloader,
            # truncating if needed
            samples = generate_samples_retokenized_streaming_text_dataset(
                loader,
                decode_tokenizer=decode_tokenizer,
                encode_tokenizer=encode_tokenizer,
                max_length=args.concat_tokens,
                no_wrap=args.no_wrap,
                bos_text=args.bos_text,
                eos_text=args.eos_text,
            )

            # Write samples
            log(
                INFO,
                "Converting %s-%s for client %s to MDS format...",
                args.name,
                split_name,
                i,
            )
            # Create temporary directory for the client
            client_temp_dir = TemporaryDirectory()
            # Define the remote path for the client
            remote_path = (
                f"{args.remote_bucket}/{vocab_size}"
                f"/fed-the-pile"
                f"/{THE_PILE_CLIENT_NAMING_MAP[args.name]}_{i}/{split_name}"
            )
            # Write the samples to the MDS file
            with MDSWriter(
                columns=columns,
                out=(client_temp_dir.name, remote_path),
                compression=args.compression,
            ) as out:
                for sample in tqdm(
                    samples,
                    desc=f"client_{i}_{args.name}_{split_name}",
                ):
                    # Writing the sample to the MDS file
                    out.write(sample)


if __name__ == "__main__":
    main(parse_args())
