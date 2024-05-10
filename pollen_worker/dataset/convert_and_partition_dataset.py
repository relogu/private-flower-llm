# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
import json
import os
import platform
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from collections.abc import Iterable

from datasets import DatasetDict, IterableDatasetDict, Dataset as HFDataset


import datasets as hf_datasets
import psutil
from streaming import MDSWriter
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from llmfoundry.utils.builders import build_tokenizer
import numpy as np

import resource

# NOTE:Without this the script will be unable to partition
# A dataset tagged on a per-sample basis
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


class PartitionTokensDataset(ConcatTokensDataset):
    """An IterableDataset that returns token samples for MDSWriter.

    Returns dicts of {'tokens': bytes}

    To use data created by this class and written to MDS format:

    ```python
        import torch
        from streaming.base import StreamingDataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained('your/tokenizer')
        ds = StreamingDataset(local='mds-data-folder', split='val')

        # note, you need to copy the numpy array because the original is non-writeable
        # and torch does not support non-writeable tensors,
        # so you get a scary warning and
        # if you do try to write to the tensor you get undefined behavior
        tokens = torch.from_numpy(np.frombuffer(ds[0]['tokens'], dtype=np.int64).copy())
        print(tokenizer.decode(tokens))
    ```
    """

    def __init__(
        self,
        hf_dataset: hf_datasets.IterableDataset | hf_datasets.Dataset,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        bos_text: str,
        eos_text: str,
        no_wrap: bool,
        partitioner: Callable[[Any], str],
    ) -> None:
        super().__init__(hf_dataset, tokenizer, max_length, bos_text, eos_text, no_wrap)
        self.partitioner = partitioner

    def __iter__(self) -> Iterable[dict[str, bytes]]:
        """Iterate over the dataset, yielding tokenized samples and partitions."""
        buffer: dict[str, list] = defaultdict(list)
        for sample in self.hf_dataset:
            intended_key = self.partitioner(sample)
            encoded = self.tokenizer(sample["text"], truncation=False, padding=False)  # type: ignore[reportArgumentType]
            iids = encoded["input_ids"]
            buffer[intended_key] = (
                buffer[intended_key] + self.bos_tokens + iids + self.eos_tokens  # type: ignore[reportArgumentType]
            )
            while len(buffer[intended_key]) >= self.max_length:
                concat_sample = buffer[intended_key][: self.max_length]
                buffer[intended_key] = (
                    buffer[intended_key][self.max_length :] if self.should_wrap else []
                )
                yield {
                    # convert to bytes to store in MDS binary format
                    "tokens": np.asarray(concat_sample).tobytes(),
                    "partition": intended_key,  # type: ignore[reportReturnType]
                }


class ConcatMode(Enum):
    """Concatenation mode for tokenized text."""

    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="""Convert dataset into MDS format,
        optionally concatenating and tokenizing"""
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--data_subset", type=str, default=None, help='E.g. "all" or "en"'
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "train_small", "val", "val_small", "val_xsmall"],
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

    parser.add_argument("--num_clients", type=int, default=1)
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

    parsed = parser.parse_args()

    if parsed.tokenizer_kwargs is not None:
        parsed.tokenizer_kwargs = json.loads(parsed.tokenizer_kwargs)
    else:
        parsed.tokenizer_kwargs = {}

    if (
        os.path.isdir(parsed.out_root)  # noqa: PTH112
        and len(set(os.listdir(parsed.out_root)).intersection(set(parsed.splits))) > 0
    ):
        raise ValueError(
            f"""--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)}
            which cannot overlap with the requested splits {parsed.splits}."""
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


@dataclass
class DataSplitConstants:
    """Constants for a dataset split."""

    hf_split: str
    folder_split: str
    raw_samples: int | None
    truncated_samples: int | None


@dataclass
class DatasetConstants:
    """Constants for a dataset."""

    chars_per_sample: int
    chars_per_token: int
    partitioner: Callable[[Any], str]
    splits: dict[str, DataSplitConstants] = field(default_factory=dict)

    def __iter__(self):  # noqa: D105, ANN204
        for _, v in self.splits.items():  # noqa: PERF102
            yield v


class TrainSmallConstants(DataSplitConstants):
    """Constants for a small training split."""

    def __init__(
        self,
        hf_split: str = "train",
        folder_split: str = "train_small",
        raw_samples: int = 100000,
        truncated_samples: int = 100000,
    ) -> None:
        """Initialize TrainSmallConstants."""
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValSmallConstants(DataSplitConstants):
    """Constants for a small validation split."""

    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_small",
        raw_samples: int = 10000,
        truncated_samples: int = 10000,
    ) -> None:
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


class ValXSmallConstants(DataSplitConstants):
    """Constants for a very small validation split."""

    def __init__(
        self,
        hf_split: str = "validation",
        folder_split: str = "val_xsmall",
        raw_samples: int = 3000,
        truncated_samples: int = 3000,
    ) -> None:
        super().__init__(hf_split, folder_split, raw_samples, truncated_samples)


pileconstants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
    partitioner=lambda sample: sample["meta"]["pile_set_name"],
)
pileconstants.splits["train"] = DataSplitConstants(
    hf_split="train",
    folder_split="train",
    raw_samples=210607728,
    truncated_samples=None,
)
pileconstants.splits["train_small"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_small",
    raw_samples=100000,
    truncated_samples=100000,
)
pileconstants.splits["val"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val",
    raw_samples=214670,
    truncated_samples=None,
)
pileconstants.splits["val_small"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
pileconstants.splits["val_xsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)

c4constants = DatasetConstants(
    chars_per_sample=2163,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
    partitioner=lambda _: "",
)
c4constants.splits["train"] = DataSplitConstants(
    hf_split="train",
    folder_split="train",
    raw_samples=364868892,
    truncated_samples=None,
)
c4constants.splits["train_small"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_small",
    raw_samples=100000,
    truncated_samples=100000,
)
c4constants.splits["val"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val",
    raw_samples=364608,
    truncated_samples=None,
)
c4constants.splits["val_small"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_small",
    raw_samples=10000,
    truncated_samples=10000,
)
c4constants.splits["val_xsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xsmall",
    raw_samples=3000,
    truncated_samples=3000,
)
c4constants.splits["val_xxsmall"] = DataSplitConstants(
    hf_split="validation",
    folder_split="val_xxsmall",
    raw_samples=100,
    truncated_samples=100,
)

CONSTS = {
    "c4": c4constants,
    "monology/pile-uncopyrighted": pileconstants,
    "allenai/c4": c4constants,
}


def build_hf_dataset(
    dataset_name: str,
    split: str,
    mode: ConcatMode,
    partitioner: Callable[[Any], str],
    init_hf_dataset: (
        DatasetDict | HFDataset | IterableDatasetDict | IterableDataset | None
    ) = None,
    max_length: int | None = None,
    bos_text: str = "",
    eos_text: str = "",
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
    data_subset: str | None = None,
) -> IterableDataset:
    """Build an IterableDataset over the HF C4 or pile source data.

    Args:
        dataset_name (str): Dataset name
        split (str): Split name.
        mode (ConcatMode): NO_CONCAT, or CONCAT_TOKENS
        max_length (int): The length of concatenated tokens
        bos_text (str): text to insert at the beginning of each sequence
        eos_text (str): text to insert at the end of each sequence
        no_wrap (bool): if concatenating,
        whether to wrap text across `max_length` boundaries
        tokenizer (PreTrainedTokenizerBase): if mode is CONCAT_TOKENS,
        the tokenizer to use
        data_subset (str): Referred to as "name" in HuggingFace datasets.load_dataset.
            Typically "all" (The Pile) or "en" (c4).

    Returns
    -------
        An IterableDataset.
    """
    hf_dataset = (
        hf_datasets.load_dataset(
            path=dataset_name, name=data_subset, split=split, streaming=True
        )
        if init_hf_dataset is None
        else init_hf_dataset
    )

    if mode == ConcatMode.NO_CONCAT:
        dataset = NoConcatDataset(hf_dataset)  # type: ignore[reportArgumentType]
    else:
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise ValueError(f"{tokenizer=} must be of type PreTrainedTokenizerBase")
        if max_length is None:
            raise ValueError("max_length must be set.")
        if bos_text + eos_text == "":  # noqa: PLC1901
            test_tokens = tokenizer("test")
            if (
                test_tokens["input_ids"][0] != tokenizer.bos_token_id  # type: ignore[reportIndexIssue]
                and test_tokens["input_ids"][-1] != tokenizer.eos_token_id  # type: ignore[reportIndexIssue]
            ):
                tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
                tok_error_msg += (
                    "Concatenating with this tokenizer will result in sequences being "
                )
                tok_error_msg += """attached without a separating token
                Please use another tokenizer, """
                tok_error_msg += (
                    "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
                )
                tok_error_msg += "--bos_text=<|endoftext|>."
                raise ValueError(tok_error_msg)
        dataset = PartitionTokensDataset(
            hf_dataset=hf_dataset,  # type: ignore[reportArgumentType]
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
            partitioner=partitioner,
        )
    return dataset


def _est_progress_denominator(
    total_samples: int,
    chars_per_sample: int,
    chars_per_token: int,
    mode: ConcatMode,
    max_length: int,
) -> int | None:
    """Estimate the denominator for a progress bar."""
    est_tokens_per_sample = chars_per_sample // chars_per_token
    if mode == ConcatMode.NO_CONCAT:
        return total_samples
    elif mode == ConcatMode.CONCAT_TOKENS:
        return total_samples * est_tokens_per_sample // max_length
    else:
        return None


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int | None,
    collate_fn: Callable | None = None,
) -> DataLoader:
    """Build a DataLoader for a dataset."""
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        if "linux" or "macos" in platform.platform().lower():  # noqa: SIM222
            num_workers = max(1, psutil.cpu_count())
        else:
            num_workers = 0

    # If using multiple workers,
    # configure each worker to prefetch as many samples as it can, up to
    # the aggregate device batch size
    # If not using workers,
    # the torch DataLoader expects the default value for prefetch_factor,
    # which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else None

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn,
    )


def generate_samples(
    loader: DataLoader, truncate_num_samples: int | None = None
) -> Iterable[dict[str, bytes]]:
    """Generate samples from a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes,
       sample1_bytes, sample2_bytes, ...]}
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


def main(args: Namespace) -> None:
    """Create C4/pile streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    try:
        dataset_constants = CONSTS[args.dataset]
    except KeyError as e:
        raise ValueError(
            f"""Constants for dataset "{args.dataset}" not found
            Currently only "the_pile" and "c4" are supported."""
        ) from e

    if args.concat_tokens is not None:
        mode = ConcatMode.CONCAT_TOKENS
        tokenizer = build_tokenizer(args.tokenizer, args.tokenizer_kwargs)
        # we will enforce length,
        # so suppress warnings about sequences too long for the model
        tokenizer.model_max_length = int(1e30)
        columns = {"tokens": "bytes"}
    else:
        mode = ConcatMode.NO_CONCAT
        tokenizer = None
        columns = {"text": "str"}

    for split_name in args.splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError as e:
            raise KeyError(f"Constants not defined for split {split_name}.") from e
        hf_split = split.hf_split
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        dataset = build_hf_dataset(
            dataset_name=args.dataset,
            init_hf_dataset=None,
            data_subset=args.data_subset,
            split=hf_split,
            mode=mode,
            max_length=args.concat_tokens,
            bos_text=args.bos_text,
            eos_text=args.eos_text,
            no_wrap=args.no_wrap,
            tokenizer=tokenizer,
            partitioner=dataset_constants.partitioner,
        )

        loader = build_dataloader(
            dataset=dataset, batch_size=512, num_workers=args.num_workers
        )

        samples_generator = iter(
            generate_samples(loader, truncate_num_samples=truncate_num_samples)
        )

        if expected_num_samples is not None:
            denominator = (
                truncate_num_samples
                if truncate_num_samples is not None
                else _est_progress_denominator(
                    total_samples=expected_num_samples,
                    chars_per_sample=dataset_constants.chars_per_sample,
                    chars_per_token=dataset_constants.chars_per_token,
                    mode=mode,
                    max_length=args.concat_tokens,
                )
            )
        else:
            denominator = None

        # Write samples

        with ExitStack() as stack:
            # Create a dictionary to store the MDSWriter for each key
            writers = {}

            client = 0
            for sample in tqdm(samples_generator, desc=folder_split, total=denominator):
                intended_key = sample.pop("partition")
                if intended_key not in writers:
                    writers |= {
                        intended_key: {
                            client: stack.enter_context(
                                MDSWriter(
                                    columns=columns,
                                    out=os.path.join(  # noqa: PTH118
                                        args.out_root,
                                        (
                                            f"{intended_key}_{client}"
                                            if intended_key
                                            else f"client_{client}"
                                        ),
                                        folder_split,
                                    ),
                                    compression=args.compression,
                                )
                            )
                            for client in range(args.num_clients)
                        }
                    }
                writers[intended_key][client].write(sample)
                client += 1
                client %= args.num_clients


if __name__ == "__main__":
    main(parse_args())
