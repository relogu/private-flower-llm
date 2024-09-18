"""
Utility Functions for Dataset Handling and Tokenizer Configuration.

This module provides various utility functions and classes to facilitate dataset
handling, tokenizer configuration, and data loading for machine learning workflows. It
includes functions for building datasets, validating tokenizer configurations, and
creating data loaders. Additionally, it defines custom dataset classes for specific use
cases.

Classes
-------
- NoConcatDatasetString
    An IterableDataset that returns text samples for MDSWriter.

Functions
---------
- check_tokenizer_config(
        tokenizer: PreTrainedTokenizerBase,
        bos_text: str = "",
        eos_text: str = ""
    ) -> None
    Validate the configuration of a tokenizer for BOS and EOS token insertion.
- build_hf_dataset(
        path: str,
        split: str,
        mode: ConcatMode,
        temp_dir: TemporaryDirectory,
        max_length: int | None = None,
        bos_text: str = "",
        eos_text: str = "",
        no_wrap: bool = False,
        tokenizer: PreTrainedTokenizerBase | None = None,
        name: str | None = None
    ) -> IterableDataset
    Build a Hugging Face dataset with optional token concatenation.
- build_dataloader(
        dataset: Dataset,
        batch_size: int,
        num_workers: int | None
    ) -> DataLoader
    Build a DataLoader for a given dataset with specified batch size and number of
    workers.

Dependencies
------------
- collections.abc
- tempfile
- llmfoundry.data
- torch.utils.data
- transformers
- datasets (Hugging Face)
- flower_llm.dataset.constants
- flower_llm.utils

Usage
-----
This module is intended to be used as a utility module for handling datasets and
tokenizers in machine learning workflows. Import the necessary functions and classes as
needed.

Example
-------
    from utils import build_hf_dataset, build_dataloader, check_tokenizer_config

    # Example usage of build_hf_dataset
    dataset = build_hf_dataset(
        path="dataset_path",
        split="train",
        mode=ConcatMode.CONCAT,
        temp_dir=TemporaryDirectory(),
        max_length=512,
        bos_text="<s>",
        eos_text="</s>",
        tokenizer=AutoTokenizer.from_pretrained("facebook/opt-125m"),
        name="dataset_name"
    )

    # Example usage of build_dataloader
    dataloader = build_dataloader(dataset, batch_size=32, num_workers=4)

    # Example usage of check_tokenizer_config
    check_tokenizer_config(tokenizer, bos_text="<s>", eos_text="</s>")
"""

from collections.abc import Iterable, Iterator
from tempfile import TemporaryDirectory

from llmfoundry.data import ConcatTokensDataset, NoConcatDataset
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

import datasets as hf_datasets

from flower_llm.dataset.constants import ConcatMode
from flower_llm.utils import get_n_cpu_cores


class NoConcatDatasetString(IterableDataset):
    """An IterableDataset that returns text samples for MDSWriter.

    Returns dicts of {'text': bytes}
    """

    def __init__(
        self,
        hf_dataset: hf_datasets.IterableDataset | hf_datasets.Dataset,
    ) -> None:
        """Retrieve the reference to the HF dataset passed."""
        self.hf_dataset = hf_dataset

    def __iter__(self) -> Iterable[str]:  # type: ignore[reportIncompatibleMethodOverride]
        """Iterate over the dataset and yield text samples."""
        for sample in self.hf_dataset:
            # convert to bytes to store in MDS binary format
            yield str(sample["text"])  # type: ignore[reportArgumentType]


def check_tokenizer_config(
    tokenizer: PreTrainedTokenizerBase, bos_text: str = "", eos_text: str = ""
) -> None:
    """
    Validate the configuration of a tokenizer for BOS and EOS token insertion.

    This function checks if the provided tokenizer is an instance of
    PreTrainedTokenizerBase and validates whether it correctly inserts BOS (Beginning
    of Sequence) and EOS (End of Sequence) tokens. If both `bos_text` and `eos_text` are
    provided, the function ensures that the tokenizer inserts the BOS and EOS tokens
    correctly. If the tokenizer does not insert these tokens, a ValueError is raised
    with an appropriate error message.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer to be validated. It must be of type PreTrainedTokenizerBase.
    bos_text : str, optional
        The text representing the BOS token. Default is an empty string.
    eos_text : str, optional
        The text representing the EOS token. Default is an empty string.

    Raises
    ------
    TypeError
        If the provided tokenizer is not an instance of PreTrainedTokenizerBase.
    ValueError
        If both `bos_text` and `eos_text` are provided and the tokenizer does not insert
        BOS and EOS tokens correctly.

    Example
    -------
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    >>> check_tokenizer_config(tokenizer, bos_text="<|endoftext|>", eos_text="")
    """
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError(
            f"{tokenizer=} must be of type PreTrainedTokenizerBase "
            f"instead of {type(tokenizer)}"
        )
    if bos_text and eos_text:
        test_tokens = tokenizer("test")
        if (
            test_tokens["input_ids"][0] != tokenizer.bos_token_id  # type: ignore[reportIndexIssue]
            and test_tokens["input_ids"][-1] != tokenizer.eos_token_id  # type: ignore[reportIndexIssue]
        ):
            tok_error_msg = "This tokenizer does not insert an EOS nor BOS token. "
            tok_error_msg += (
                "Concatenating with this tokenizer will result in sequences being "
            )
            tok_error_msg += "attached without a separating token."
            "Please use another tokenizer, "
            tok_error_msg += (
                "such as facebook/opt-125m, or specify EOS/BOS text with e.g. "
            )
            tok_error_msg += "--bos_text=<|endoftext|>."
            raise ValueError(tok_error_msg)


def build_hf_dataset(
    path: str,
    split: str,
    mode: ConcatMode,
    temp_dir: TemporaryDirectory,
    max_length: int | None = None,
    bos_text: str = "",
    eos_text: str = "",
    no_wrap: bool = False,
    tokenizer: PreTrainedTokenizerBase | None = None,
    name: str | None = None,
) -> IterableDataset:
    """
    Build a Hugging Face dataset with optional token concatenation.

    This function loads a dataset from the Hugging Face Hub and wraps it in a dataset
    class based on the specified concatenation mode. If no concatenation is required,
    the dataset is wrapped in a NoConcatDataset. If concatenation is required, the
    dataset is wrapped in a ConcatTokensDataset, and the tokenizer configuration is
    validated.

    Parameters
    ----------
    path : str
        The path or name of the dataset to load from the Hugging Face Hub.
    split : str
        The dataset split to load (e.g., "train", "validation").
    mode : ConcatMode
        The concatenation mode to use. Must be an instance of ConcatMode.
    temp_dir : TemporaryDirectory
        A temporary directory for caching the dataset.
    max_length : int, optional
        The maximum length of concatenated tokens, required when concatenating.
    bos_text : str, optional
        The text representing the BOS token. Default is an empty string.
    eos_text : str, optional
        The text representing the EOS token. Default is an empty string.
    no_wrap : bool, optional
        Whether to disable wrapping of tokens. Default is False.
    tokenizer : PreTrainedTokenizerBase, optional
        The tokenizer to use for token concatenation, required when concatenating.
    name : str, optional
        The name of the dataset configuration. Default is None.

    Returns
    -------
    IterableDataset
        The wrapped dataset, either as a NoConcatDataset or ConcatTokensDataset.

    Raises
    ------
    AssertionError
        If concatenation is enabled but `tokenizer` or `max_length` is not provided.
    TypeError
        If the provided tokenizer is not an instance of PreTrainedTokenizerBase.
    ValueError
        If the tokenizer configuration is invalid for BOS and EOS token insertion.

    Example
    -------
    >>> from transformers import AutoTokenizer
    >>> from tempfile import TemporaryDirectory
    >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    >>> temp_dir = TemporaryDirectory()
    >>> dataset = build_hf_dataset(
    ...     path="allenai/c4",
    ...     split="train",
    ...     mode=ConcatMode.CONCAT,
    ...     temp_dir=temp_dir,
    ...     max_length=512,
    ...     bos_text="<s>",
    ...     eos_text="</s>",
    ...     tokenizer=tokenizer,
    ...     name="en"
    ... )
    """
    # Load dataset from HF Hub
    hf_dataset = hf_datasets.load_dataset(  # type: ignore[attr-defined]
        path=path,  # Path or name of the dataset
        name=name,  # Defining the name of the dataset configuration.
        data_dir=None,  # We are getting it from the Hub
        split=split,  # Which split to use
        streaming=True,
        cache_dir=temp_dir.name,
        keep_in_memory=False,
        save_infos=False,
        trust_remote_code=True,
    )
    # Wrap the dataset depending on the concatenation mode
    if mode == ConcatMode.NO_CONCAT:
        # Wrap the dataset in a NoConcatDataset
        dataset = NoConcatDataset(hf_dataset)  # type: ignore[reportArgumentType]
    else:
        assert tokenizer is not None, "Tokenizer must be provided for concatenation."
        assert max_length is not None, "Max length must be provided for concatenation."
        # Check that the tokenizer is properly configured
        check_tokenizer_config(tokenizer, bos_text, eos_text)
        # Wrap the dataset in a ConcatTokensDataset
        dataset = ConcatTokensDataset(
            hf_dataset=hf_dataset,  # type: ignore[reportArgumentType]
            tokenizer=tokenizer,
            max_length=max_length,
            bos_text=bos_text,
            eos_text=eos_text,
            no_wrap=no_wrap,
        )
    return dataset


def build_dataloader(
    dataset: Dataset, batch_size: int, num_workers: int | None
) -> DataLoader:
    """
    Build a DataLoader for a given dataset with batch size and number of workers.

    This function creates a DataLoader for the provided dataset, configuring the number
    of workers and prefetch factor based on the batch size. If `num_workers` is None,
    the function determines the number of CPU cores available and uses that value.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load data from.
    batch_size : int
        The number of samples per batch to load.
    num_workers : int | None
        The number of worker processes to use for data loading. If None, the number of
        CPU cores is used.

    Returns
    -------
    DataLoader
        A DataLoader instance configured with the specified dataset, batch size, and
        number of workers.

    Notes
    -----
    - Multiple workers are only supported on Linux machines.
    - The prefetch factor is configured based on the number of workers and batch size.
      If using multiple workers, each worker is configured to prefetch as many samples
      as it can, up to the aggregate device batch size. If not using workers, a default
      prefetch factor of 2 is used.

    Example
    -------
    >>> from torch.utils.data import Dataset, DataLoader
    >>> class MyDataset(Dataset):
    ...     def __len__(self):
    ...         return 100
    ...     def __getitem__(self, idx):
    ...         return idx
    >>> dataset = MyDataset()
    >>> dataloader = build_dataloader(dataset, batch_size=10, num_workers=4)
    >>> for batch in dataloader:
    ...     print(batch)
    """
    if num_workers is None:
        # Multiple workers is only supported on linux machines
        num_workers = get_n_cpu_cores()

    # If using multiple workers, configure each worker to prefetch as many samples as
    # it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for
    # prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size // num_workers) if num_workers > 0 else 2

    return DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )


def generate_samples(
    iterator: Iterable[str], truncate_num_samples: int | None = None
) -> Iterator[str]:
    """
    Generate samples from an iterator with optional truncation.

    This function takes an iterator of strings and yields items from it. If the
    `truncate_num_samples` parameter is provided, the function will yield up to
    that many items and then stop. If `truncate_num_samples` is None, all items
    from the iterator will be yielded.

    Parameters
    ----------
    iterator : Iterable[str]
        An iterator that yields strings.
    truncate_num_samples : int | None, optional
        The maximum number of samples to yield. If None, all items from the iterator
        will be yielded. Default is None.

    Returns
    -------
    Iterator[str]
        An iterator that yields strings from the input iterator, up to the specified
        number of samples.

    Example
    -------
    >>> data = ["sample1", "sample2", "sample3"]
    >>> for sample in generate_samples(data, truncate_num_samples=2):
    ...     print(sample)
    sample1
    sample2
    """
    if truncate_num_samples is None:
        truncate_num_samples = -1
    for item in iterator:
        if truncate_num_samples == 0:
            return
        truncate_num_samples -= 1
        yield item
