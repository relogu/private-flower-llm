# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import json
from multiprocessing.managers import ListProxy, SyncManager

from flower_llm.dataset.convert_dataset_hf import CONSTANTS
import os
import platform
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from torch.utils.data._utils.worker import _worker_loop  # noqa: PLC2701
from enum import Enum

from pathlib import Path
from typing import Any
from collections import Counter
from multiprocessing import Manager
import time

import psutil
from logging import INFO
from flwr.common.logger import log
from llmfoundry.utils.builders import build_tokenizer
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    get_worker_info,
)
from tqdm import tqdm

import numpy as np

from flower_llm.dataset.text_data import StreamingTextDataset
import operator


class ConcatMode(Enum):
    """Describe concatenation modes."""

    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse command line arguments."""
    parser = ArgumentParser(
        description=(
            "Convert dataset into MDS format, optionally concatenating and tokenizing"
        )
    )
    parser.add_argument("--dataset", type=str, required=True)

    parser.add_argument(
        "--data_subset", type=str, default=None, help='E.g. "all" or "en"'
    )
    # NOTE: Controls maximum n-gram length
    # We will count all n-grams up to this length
    parser.add_argument("--max_grams", type=int, default=1)

    # NOTE: This controls after how many samples a worker sends to file
    # It is appropriate for 64 workers with max_grams=8
    # Should result in a peak memory utilization of ~530 GB range
    parser.add_argument("--dump_frequency", type=int, default=int(10**3))

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "train_small", "val", "val_small", "val_xsmall"],
    )

    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

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

    parser.add_argument("--post_process_only", default=False, action="store_true")
    parser.add_argument("--shuffle", default=True, action="store_true")
    parser.add_argument("--shuffle_seed", type=int, default=17)

    parser.add_argument("--num_samples_freq_map", type=int, default=int(10**6))

    parsed = parser.parse_args()

    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    if (
        Path.is_dir(Path(parsed.out_root))
        and len(set(os.listdir(Path(parsed.out_root))).intersection(set(parsed.splits)))
        > 0
        and not parsed.post_process_only
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

    # now that we have validated them, change BOS/EOS to strings
    if parsed.bos_text is None:
        parsed.bos_text = ""
    if parsed.eos_text is None:
        parsed.eos_text = ""
    return parsed


def worker_loop_with_cleanup(
    cleanup_func: Callable[[Dataset], None],
    *args: Any,
    **kwargs: Any,
) -> None:
    """Wrap the worker loop to call cleanup function."""
    try:
        _worker_loop(*args, **kwargs)
    finally:
        worker_info = get_worker_info()
        assert worker_info is not None
        cleanup_func(worker_info.dataset)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int | None,
    max_grams: int,
    manager: SyncManager,
    out_folder: Path,
    dump_frequency: int,
) -> tuple[DataLoader, ListProxy]:
    """Return a DataLoader for a dataset."""
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        current_platform = platform.platform().lower()
        if "linux" in current_platform or "macos" in current_platform:
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers, configure each worker to prefetch as many samples as
    # it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for
    # prefetch_factor, which non-intuitively must be 2.
    dataset.global_counters = manager.list([manager.dict() for _ in range(num_workers)])  # type: ignore[reportAttributeAccessIssue]
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    barrier = manager.Barrier(num_workers)  # type: ignore[reportAttributeAccessIssue, attr-defined]

    # NOTE: For writing back the number of tokens in each sample
    global_sample_cnt_list = manager.list()

    def worker_init_fun(worker_id: int) -> None:
        """Initialize the worker with the dataset and counters."""
        worker_info = get_worker_info()
        assert worker_info is not None
        dataset = worker_info.dataset

        # NOTE: This is dataset agnostic
        # Should not require subclassing
        dataset.max_grams = max_grams  # type: ignore[reportAttributeAccessIssue]
        dataset.local_counters = [Counter() for _ in range(max_grams)]  # type: ignore[reportAttributeAccessIssue]
        dataset.sample_cnt_list = []  # type: ignore[reportAttributeAccessIssue]
        dataset.global_sample_cnt_list = global_sample_cnt_list  # type: ignore[reportAttributeAccessIssue]
        dataset.dump_frequency = dump_frequency  # type: ignore[reportAttributeAccessIssue]
        dataset.barrier = barrier  # type: ignore[reportAttributeAccessIssue]
        dataset.worker_id = worker_id  # type: ignore[reportAttributeAccessIssue]
        dataset.out_folder = out_folder  # type: ignore[reportAttributeAccessIssue]
        dataset.global_counters = dataset.global_counters  # type: ignore[reportAttributeAccessIssue]

    def cleanup_function(dataset: Dataset) -> None:
        """Cleanup function to merge local counters into global counters.

        NOTE: Any other solution including atexit and/or __del__ will fail
        because pytorch does not close subprocesses in a standard way.
        """
        dataset.global_sample_cnt_list.extend(dataset.sample_cnt_list)  # type: ignore[reportAttributeAccessIssue]

        # Dump the data to a file with pickle

        # Note only if any of the local counter is not empty

        for i in range(dataset.max_grams):
            # Make sure we output even the last batch
            length = len(dataset.sample_cnt_list)
            log(INFO, "Dumping %d-gram %d", i + 1, length)

            file_path = f"{i + 1}_gram_{dataset.worker_id}_{length + 1}.json"
            file: Path = dataset.out_folder / file_path

            json.dump(dataset.local_counters[i], file.open(mode="w"), indent=4)
            log(INFO, "Dumped %d-gram %d", i + 1, length)

        dataset.barrier.wait()  # type: ignore[reportAttributeAccessIssue]
        os.sync()

    # NOTE: partial binding to replace the worker loop
    # And call the proper cleanup function
    # Equivalent to creating a closure
    torch.utils.data._utils.worker._worker_loop = functools.partial(  # type: ignore[reportAttributeAccessIssue]
        worker_loop_with_cleanup, cleanup_function
    )

    def custom_get_item(self, idx: int) -> Any:  # noqa: ANN001
        """Get item and count n-grams."""
        sample = super(dataset.__class__, dataset).__getitem__(idx)  # type: ignore[reportAttributeAccessIssue]

        # Decoding a sample requires going from bytes to ndarrays
        decoded_sample = [
            int(k) for k in np.frombuffer(sample["tokens"], dtype=np.int64)
        ]

        # Add the number of tokens in the decoded sample
        self.sample_cnt_list.append(len(decoded_sample))

        # Count n-grams
        for n in range(self.max_grams):
            # NOTE: This groups n-grams together
            # For example: x = [1,2,3,4,5]
            # x[0:] = [1,2,3,4,5]
            # x[1:] = [2,3,4,5]
            # zip(*[x[i:] for i in range(2)]) = [(1,2), (2,3), (3,4), (4,5)]

            # NOTE: for local_counters, update is equivalent to
            # summing frequencies for a key
            self.local_counters[n].update(
                str(it)
                for it in zip(
                    *[decoded_sample[i:] for i in range(n + 1)],
                    strict=False,
                )
            )

            if len(self.sample_cnt_list) % self.dump_frequency == 0:
                # Dump the data to a file with pickle

                self.out_folder.mkdir(parents=True, exist_ok=True)

                file: Path = (
                    self.out_folder
                    / f"{n + 1}_gram_{self.worker_id}_{len(self.sample_cnt_list)}.json"
                )
                json.dump(
                    self.local_counters[n],
                    file.open(mode="w"),
                    indent=4,
                )
                self.local_counters[n] = Counter()

        # NOTE: No need to keep the sample in memory
        return torch.empty(1)

    # Replace the sampling function of the dataset to count n-grams
    dataset.__getitem__ = custom_get_item.__get__(dataset, dataset.__class__)

    # NOTE: persistent workers should be true
    # to avoid them getting killed before they can write back

    dataloader = DataLoader(
        dataset=dataset,
        sampler=None,
        persistent_workers=True,
        worker_init_fn=worker_init_fun,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    return dataloader, global_sample_cnt_list


def load_and_process_file(file: Path, delete: bool, load: bool) -> Counter:
    """Load and return the counters from a file."""
    if load:
        with file.open(mode="r") as f:
            counters = Counter(json.load(f))
        return counters
    if delete and not file.match("*_gram.json"):
        file.unlink()  # Remove the file after loading
    return Counter()


def build_complete_freq_map(
    out_folder: Path,
    gram: int,
    num_workers: int,
    write: bool = False,
    delete: bool = False,
    load: bool = True,
) -> Counter:
    """Build a complete frequency map from the shards."""
    # Glob all the files in the out_folder which contain the word shard
    # and load them
    files = list(out_folder.glob(f"{gram}_gram*"))
    counter: Counter | None = None
    cnt = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # load_and_process_file
        futures = {
            executor.submit(load_and_process_file, file, delete, load): file
            for file in files
        }

        for future in as_completed(futures):
            res = future.result()
            if counter is None:
                counter = res
            else:
                cnt += 1
                counter.update(res)
                if cnt % 100 == 0:
                    elapsed_time = time.time() - start_time
                    log(INFO, "Processed %d files in %.2f seconds", cnt, elapsed_time)

    log(INFO, "Loaded all files")
    assert counter is not None
    if out_folder is not None:
        counters_file = out_folder / f"{gram}_gram.json"
        if not counters_file.exists() and write:
            json.dump(
                counter,
                counters_file.open(mode="w"),
                indent=4,
            )
    return counter


def main(args: Namespace) -> None:
    """Create C4/pile streaming dataset.

    Args:
        args (Namespace): Command line arguments.
    """
    log(INFO, "Arguments received: %s", args)

    # Retrieve constants for the dataset
    try:
        dataset_constants = CONSTANTS[args.dataset]
    except KeyError as e:
        raise ValueError(
            f'Constants for dataset "{args.dataset}" not found. Currently only'
            '"the_pile" and "c4" are supported.'
        ) from e
    # Elaborate over whether to concatenate tokens or not
    max_grams = int(args.max_grams)

    tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)
    # We will enforce length because it suppress warnings about sequences too long
    # for the model
    tokenizer.model_max_length = int(1e30)

    # NOTE: The manager creates synchronized objects that can be shared across processes
    manager = Manager()

    # Loop over passed splits
    for split_name in args.splits:
        # Retrieving info about the current split

        try:
            split = dataset_constants.splits[split_name]
        except KeyError as e:
            raise KeyError(f"Constants not defined for split {split_name}.") from e
        folder_split = split.folder_split

        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Save the frequency maps
        out_folder = Path(args.out_root) / folder_split
        out_folder.mkdir(parents=True, exist_ok=True)
        if not args.post_process_only:
            # NOTE: we will never build metadata from source
            # Too slow
            assert tokenizer is not None
            # Build dataset potentially with streams

            dataset = StreamingTextDataset(
                tokenizer=tokenizer,
                streams=None,
                batch_size=512,
                remote=args.remote,
                local=args.local,
                split=split_name,
                shuffle=args.shuffle,
                max_seq_len=args.concat_tokens,
                shuffle_seed=args.shuffle_seed,
                cache_limit=None,
            )
            # Substituting the dataset.__getitem__ method with its parent's method
            dataset.__getitem__ = super(dataset.__class__, dataset).__getitem__  # type: ignore[reportAttributeAccessIssue,method-assign]

            # Build a batched dataloader for streaming the HF dataset in batches
            # and counting n-grams/obtaining statistics
            # NOTE: to avoid overloading RAM we save intermediary results to disk
            loader, sample_cnt_list = build_dataloader(
                dataset=dataset,
                batch_size=512,
                num_workers=args.num_workers,
                max_grams=max_grams,
                manager=manager,
                out_folder=out_folder,
                dump_frequency=args.dump_frequency,
            )

            # Looping through the dataloader to count n-grams
            for _ in tqdm(loader):
                pass

            # Allow some time to pass to make sure everyone
            # is in a consistent state and has time
            # to write back to the shared memory

            time.sleep(5)

            # We have to kill the workers
            # for the finally clause to be executed
            loader._iterator._shutdown_workers()
            time.sleep(5)

            # Write samples
            log(INFO, "Creating frequency map for dataset...")

            unigrams = build_complete_freq_map(
                out_folder, 1, args.num_workers, write=False, delete=True
            )

            sorted_unigrams = dict(
                sorted(unigrams.items(), key=operator.itemgetter(1), reverse=True)
            )

            unigram_total_tokens = sum(unigrams.values())

            assert tokenizer is not None

            # Sort keys by frequency rather than insertion order
            log(INFO, "Counter loading completed")

            # Total number of samples processed
            n_samples = len(sample_cnt_list)

            total_tokens_samples = sum(sample_cnt_list)

            assert (
                total_tokens_samples == unigram_total_tokens
            ), f"""Total tokens mismatch samples: {total_tokens_samples},
            unigrams: {unigram_total_tokens}"""

            samples_file = out_folder / "samples_stats.json"
            samples_dict = {
                "total_tokens_samples_counting": total_tokens_samples,
                "num_samples": n_samples,
            }

            log(INFO, "Writing statistics to %s", samples_file)
            # Where to store the number of tokens per sample
            # to take statistics, although I am pretty sure they
            # are all 2048 right until the end
            samples_raw = out_folder / "samples_raw.json"

            # Check if file already exists
            if not samples_file.exists():
                json.dump(samples_dict, samples_file.open(mode="w"), indent=4)

            if not samples_raw.exists():
                json.dump(
                    {"tokens_per_sample": list(sample_cnt_list)},
                    samples_raw.open(mode="w"),
                    indent=4,
                )

            decoded_unigrams = {k: (v, "") for k, v in sorted_unigrams.items()}

            #  Save the unigrams
            unigrams_file = out_folder / "1_gram.json"
            if not unigrams_file.exists():
                json.dump(decoded_unigrams, unigrams_file.open(mode="w"), indent=4)

            log(
                INFO,
                "Unigrams saved to %s, len %s",
                unigrams_file,
                len(decoded_unigrams),
            )
        else:
            for gram in range(3, max_grams + 1):
                build_complete_freq_map(
                    out_folder,
                    gram,
                    min(args.num_workers, 4),
                    write=True,
                    delete=False,
                    load=True,
                )
                build_complete_freq_map(
                    out_folder,
                    gram,
                    min(args.num_workers, 4),
                    write=False,
                    delete=True,
                    load=False,
                )


if __name__ == "__main__":
    main(parse_args())
