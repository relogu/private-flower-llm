# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

import json
import os
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable
from logging import INFO
from pathlib import Path
from tempfile import TemporaryDirectory

from flwr.common.logger import log
from llmfoundry.utils.builders import build_tokenizer
from streaming import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


from flower_llm.dataset.constants import CONSTANTS, ConcatMode, DatasetConstants
from flower_llm.dataset.text_data import StreamingTextDataset
from flower_llm.dataset.utils import build_dataloader, build_hf_dataset


def parse_args() -> Namespace:  # noqa: D103
    parser = ArgumentParser(
        description=(
            "Convert dataset into MDS format, optionally concatenating and tokenizing."
        )
    )
    # Parameters for downloading the dataset from HF Hub
    parser.add_argument(
        "--path", type=str, required=True, help='E.g. "allenai/c4" or ""'
    )
    parser.add_argument("--names", nargs="+", default=None, help='E.g. "all" or "en"')
    parser.add_argument(
        "--splits", nargs="+", default=None, help='E.g. "train" or "validation"'
    )
    # Parameters to creating the output MDS dataset
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)
    # Parameters for the tokenization and (potentially) concatenation
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--concat_tokens",
        type=int,
        help="Convert text to tokens and concatenate up to this many tokens",
    )
    parser.add_argument("--tokenizer", type=str, required=False, default=None)
    parser.add_argument("--tokenizer_kwargs", type=str, required=False)
    parser.add_argument("--bos_text", type=str, required=False, default=None)
    parser.add_argument("--eos_text", type=str, required=False, default=None)
    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, required=False, default=None)
    # Number of clients to compose the federated dataset (this is done at a
    # dataset/configuration/split level)
    parser.add_argument("--num_clients", type=int, required=False, default=1)
    # Arguments to use our S3-stored dataset when concatenating tokens
    parser.add_argument(
        "--local",
        type=str,
        default="/local/scratch/tmp",
        help="Local path to centralized dataset",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default="s3://c4-dataset",
        help="Remote path to centralized dataset",
    )
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=17)

    # Parse arguments
    parsed = parser.parse_args()

    # Parse tokenizer_kwargs
    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    # Check that the output root does not contain any of the requested splits
    if (
        Path.is_dir(Path(parsed.out_root))
        and len(set(os.listdir(Path(parsed.out_root))).intersection(set(parsed.splits)))
        > 0
    ):
        raise ValueError(
            f"--out_root={Path(parsed.out_root)} contains"
            f"{os.listdir(Path(parsed.out_root))} which"
            f"cannot overlap with the requested splits {parsed.splits}."
        )

    # Make sure we have needed concat options
    if (
        parsed.concat_tokens is not None
        and isinstance(parsed.concat_tokens, int)
        and parsed.tokenizer is None
    ):
        parser.error("When setting --concat_tokens, you must specify a --tokenizer")

    # Change BOS/EOS to strings if they are None
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""

    # Parse splits and names
    if parsed.names is not None:
        parsed.names = set(parsed.names)
    if parsed.splits is not None:
        parsed.splits = set(parsed.splits)
    return parsed


def _est_progress_denominator(
    total_samples: int,
    chars_per_sample: int,
    chars_per_token: int,
    mode: ConcatMode,
    max_length: int,
) -> int | float | None:
    est_tokens_per_sample = chars_per_sample / chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return (total_samples * est_tokens_per_sample) / max_length
    return None


def generate_samples(
    loader: DataLoader, truncate_num_samples: int | None = None
) -> Iterable[dict[str, bytes]]:
    """Build a Generator over samples of a dataloader.

    Args:
       loader (DataLoader):
        A dataloader emitting batches like
        {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields
    ------
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {k: v[idx] for k, v in batch.items()}


def main(args: Namespace) -> None:  # noqa: D103
    log(INFO, "Arguments received: %s", args)
    # Create temporary directory
    temp_dir = TemporaryDirectory()
    # Retrieve constants for the dataset
    try:
        dataset_constants: DatasetConstants = CONSTANTS[args.dataset]
    except KeyError as e:
        raise ValueError(
            f'Constants for dataset "{args.dataset}" not found. Currently only'
            '"the_pile" and "c4" are supported.'
        ) from e
    # Elaborate over whether to concatenate tokens or not
    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        # TODO: Add support for custom pre-trained tokenizers saved as files
        tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)
        # We will enforce length because it suppress warnings about sequences too long
        # for the model
        tokenizer.model_max_length = int(1e30)
        columns = {"tokens": "bytes"}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {"text": "str"}
    # Loop over passed splits
    for split_name in args.splits:
        # Retrieving info about the current split
        try:
            split_constants = dataset_constants.splits[split_name]
        except KeyError as e:
            raise KeyError(f"Constants not defined for split {split_name}.") from e
        folder_split = split_constants.folder_split
        expected_num_samples = split_constants.raw_samples
        truncate_num_samples = split_constants.truncated_samples
        # Create the dataset given the parameters
        # NOTE: We can't know how many samples we will get from the dataset
        if mode == ConcatMode.NO_CONCAT:
            dataset = build_hf_dataset(
                path=args.path,
                name=args.name,
                split=split_constants.split,
                mode=mode,
                max_length=args.concat_tokens,
                bos_text=args.bos_text,
                eos_text=args.eos_text,
                no_wrap=args.no_wrap,
                tokenizer=tokenizer,
                temp_dir=temp_dir,
            )
        else:
            assert tokenizer is not None
            # Build dataset potentially with streams
            dataset = StreamingTextDataset(
                tokenizer=tokenizer,
                streams=None,
                batch_size=None,
                local=args.local,
                remote=args.remote,
                split=split_name,
                shuffle=args.shuffle,
                max_seq_len=args.concat_tokens,
                shuffle_seed=args.shuffle_seed,
                cache_limit=None,
            )
            # Substituting the dataset.__getitem__ method with its parent's method
            dataset.__getitem__ = super(dataset.__class__, dataset).__getitem__  # type: ignore[reportAttributeAccessIssue]

        # Build a batched dataloader for streaming the HF dataset in batches so that we
        # can actually take advantage of multiprocessing
        loader = build_dataloader(
            dataset=dataset, batch_size=512, num_workers=args.num_workers
        )
        # Build a generator that yields samples from the batched dataloader, truncating
        # if needed
        samples = generate_samples(loader, truncate_num_samples=truncate_num_samples)

        if split_constants.denominator is not None:
            denominator = split_constants.denominator
        else:
            denominator = 0
            for _ in tqdm(samples, desc=folder_split):
                denominator += 1
            # Build a batched dataloader for streaming the HF dataset in batches
            loader = build_dataloader(
                dataset=dataset, batch_size=512, num_workers=args.num_workers
            )
            # Build a generator that yields samples from the batched dataloader
            samples = generate_samples(
                loader, truncate_num_samples=truncate_num_samples
            )
        log(INFO, f"Number of samples in {folder_split} is {denominator}.")

        # Estimating the total number of samples
        if "small" in split_name:
            if expected_num_samples is not None:
                assert truncate_num_samples is not None
                denominator = (
                    truncate_num_samples
                    # if truncate_num_samples is not None
                    # else _est_progress_denominator(
                    #     total_samples=expected_num_samples,
                    #     chars_per_sample=dataset_constants.chars_per_sample,
                    #     chars_per_token=dataset_constants.chars_per_token,
                    #     mode=mode,
                    #     max_length=args.concat_tokens,
                    # )
                )
            else:
                raise ValueError(
                    "Expected number of samples must be set for partitioning to work."
                )
            log(INFO, f"Estimated number of total samples is {denominator}.")
        else:
            log(
                INFO,
                "Counting the number of samples w/ the current settings for split %s.",
                split_name,
            )
            if split_constants.denominator is not None:
                denominator = split_constants.denominator
            else:
                denominator = 0
                for _ in tqdm(samples, desc=folder_split):
                    denominator += 1
                # Re-build a batched dataloader for streaming the HF dataset in batches
                loader = build_dataloader(
                    dataset=dataset, batch_size=512, num_workers=args.num_workers
                )
                # Re-build a generator that yields samples from the batched dataloader
                samples = generate_samples(
                    loader, truncate_num_samples=truncate_num_samples
                )
            log(INFO, f"Number of samples in {folder_split} is {denominator}.")
        # Estimate the number of samples for the current client
        # NOTE: The last client will get the remainder of the samples
        expected_samples_per_client = denominator // args.num_clients
        log(INFO, f"Expected samples per client {expected_samples_per_client}.")
        remainder = int(denominator % args.num_clients)
        log(INFO, f"Remainder is {remainder}.")

        # Write samples
        log(INFO, f"Converting {folder_split} to MDS format...")
        log(
            INFO,
            "Note: the progress bar is based on the dataset length before"
            " tokenization, and may finish at a value before 100%.",
        )
        # Loop over the number of clients
        for i in range(args.num_clients):
            # Add the remainder to the last client
            if i == args.num_clients - 1:
                expected_samples_per_client += remainder
            # Set the output path given the client id
            out_path = (
                Path(args.out_root) / f"client_{i}" / folder_split
                if args.num_clients > 1
                else Path(args.out_root) / folder_split
            )
            log(
                INFO,
                "Writing client %s with %s expected samples on folder %s.",
                i,
                expected_samples_per_client,
                out_path,
            )
            with MDSWriter(
                columns=columns,
                out=str(out_path),
                compression=args.compression,
            ) as out:
                for j, sample in enumerate(
                    tqdm(samples, desc=folder_split, total=expected_samples_per_client)
                ):
                    # Writing the sample to the MDS file
                    out.write(sample)
                    # Break if we have reached the expected number of samples
                    if j == expected_samples_per_client - 1:
                        break


if __name__ == "__main__":
    main(parse_args())
