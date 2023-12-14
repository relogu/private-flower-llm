# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Build a StreamingTextDataset dataset and dataloader for training."""

import os
from itertools import islice
from logging import INFO
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union, cast

import hydra
import numpy as np
import torch
import transformers
from composer.core.data_spec import DataSpec
from composer.core.types import Batch
from flwr.common.logger import log
from llmfoundry.utils.config_utils import pop_config
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from streaming import Stream, StreamingDataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase


class StreamingTextDataset(StreamingDataset):
    """Generic text dataset using MosaicML's StreamingDataset.

    Args:
        tokenizer (Tokenizer): HuggingFace tokenizer to
            tokenize samples.
        max_seq_len (int): The max sequence length of each sample.
        streams (Sequence[Stream], optional): One or more Streams to stream/cache
            samples from, which may be upsampled or downsampled. StreamingDataset uses
            either ``streams`` or ``remote``/``local``.
            Defaults to ``None``.
        remote (str, optional): Remote path or directory to download the dataset from.
            If ``None``, its data must exist locally. StreamingDataset uses either
            ``streams`` or ``remote``/``local``.
            Defaults to ``None``.
        local (str, optional): Local working directory to download shards to. This is
            where shards are cached while they are being used. Uses a temp directory if
            not set. StreamingDataset uses either ``streams`` or ``remote``/``local``.
            Defaults to ``None``.
        split (str, optional): Which dataset split to use, if any. If provided, we
            stream from/to the ``split`` subdirs of  ``remote`` and ``local``.
            Defaults to ``None``.
        download_retry (int): Number of download re-attempts before giving up.
            Defaults to ``2``.
        download_timeout (float): Number of seconds to wait for a shard to download
            before raising an exception.
            Defaults to ``60``.
        validate_hash (str, optional): Optional hash or checksum algorithm to use to
            validate shards. Defaults to ``None``.
        keep_zip (bool): Whether to keep or delete the compressed form when
            decompressing downloaded shards. If ``False``, keep iff remote is local or
            no remote.
            Defaults to `False``.
        epoch_size (Union[int, str], optional): Number of samples to draw per epoch
            balanced across all streams. If ``None``, takes its value from the total
            number of underlying samples. Provide this field if you are weighting
            streams relatively to target a larger or smaller epoch size.
            Defaults to ``None``.
        predownload (int, optional): Target number of samples ahead to download the
            shards of while iterating. If ``None``, its value is set to
            ``8 * batch_size``.
            Defaults to ``None``.
        cache_limit (Union[int, str], optional) - Maximum size in bytes of this
            StreamingDataset's shard cache. Before downloading a shard, the least
            recently used resident shard(s) may be evicted (deleted from the local
            cache) in order to stay under the limit. Set to None to disable shard
            eviction. Supports integer bytes as well as string human-readable
            bytes (e.g., 100b, 64kb, 77mb, and so on).
            Defaults to None.
        partition_algo (str): Which partitioning algorithm to use. Defaults to ``orig``.
        num_canonical_nodes (int, optional): Canonical number of nodes for shuffling
            with resumption. If ``None``, this is interpreted as 64 times the number of
            physical nodes of the initial run if ``shuffle_algo`` is ``py1s`` or
            ``py2s``, and simply the number of physical nodes of the initial run
            otherwise.
            Defaults to ``None``.
        batch_size (int, optional): Batch size of its DataLoader, which affects how the
            dataset is partitioned over the workers.
            Defaults to ``None``.
        shuffle (bool): Whether to iterate over the samples in randomized order.
            Defaults to ``False``.
        shuffle_algo (str): Which shuffling algorithm to use. Defaults to ``py1e``.
        shuffle_seed (int): Seed for Deterministic data shuffling. Defaults to ``9176``.
        shuffle_block_size (int, optional): Unit of shuffle. A canonical node's samples
            are split into blocks of this size, and samples within each block are
            shuffled. If ``None``, its value is calculated as
            ``max(4_000_000 // num_canonical_nodes), 1 << 18)``.
            Defaults to ``None``.
        sampling_method (str): Which sampling method to use, either ``balanced`` or
            ``fixed``.
            Defaults to ``balanced``.
        sampling_granularity (int): When picking samples for a stream's final partial
            repeat, how many samples to pick from the same shard at a time (``1`` for
            evenly balanced across shards, ``1000`` to pick 1000 samples from the same
            shard at a time, etc).
            Defaults to ``1``.
        batching_method (str): Which batching method to use, either ``random``,
            ``stratified``, or ``per_stream``. Defaults to ``random``.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int,
        streams: Optional[Sequence[Stream]] = None,
        remote: Optional[str] = None,
        local: Optional[str] = None,
        split: Optional[str] = None,
        download_retry: int = 2,
        download_timeout: float = 60,
        validate_hash: Optional[str] = None,
        keep_zip: bool = False,
        epoch_size: Optional[Union[int, str]] = None,
        predownload: Optional[int] = None,
        cache_limit: Optional[Union[int, str]] = None,
        partition_algo: str = "relaxed",
        num_canonical_nodes: Optional[int] = None,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        shuffle_algo: str = "py1e",
        shuffle_seed: int = 9176,
        shuffle_block_size: Optional[int] = None,
        sampling_method: str = "balanced",
        sampling_granularity: int = 1,
        batching_method: str = "random",
        **kwargs: Any,
    ):
        group_method = kwargs.pop("group_method", None)
        if group_method is not None:
            raise NotImplementedError(
                "group_method is deprecated and has been removed.\nTo "
                "concatenate, use the --concat_tokens "
                "argument when creating your MDS dataset with concat_c4.py"
            )

        if len(kwargs) > 0:
            raise ValueError(
                f"StreamingTextDataset() got an unexpected keyword argument: {kwargs}"
            )

        if local is not None and (remote is None or (local == remote)):
            if os.path.isdir(local):
                contents = set(os.listdir(local))
                if split not in contents:
                    raise ValueError(
                        f"local directory {local} does not contain split {split}"
                    )

        # TODO: discover where yamls are being converted incorrect, but temporary
        # workaround
        if isinstance(shuffle_block_size, float):
            shuffle_block_size = int(shuffle_block_size)

        # Build Dataset
        super().__init__(
            streams=streams,
            remote=remote,
            local=local,
            split=split,
            download_retry=download_retry,
            download_timeout=download_timeout,
            validate_hash=validate_hash,
            keep_zip=keep_zip,
            epoch_size=epoch_size,
            predownload=predownload,
            cache_limit=cache_limit,
            partition_algo=partition_algo,
            num_canonical_nodes=num_canonical_nodes,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_algo=shuffle_algo,
            shuffle_seed=shuffle_seed,
            shuffle_block_size=shuffle_block_size,
            sampling_method=sampling_method,
            sampling_granularity=sampling_granularity,
            batching_method=batching_method,
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    # How to tokenize a text sample to a token sample
    def _tokenize(self, text_sample: Mapping) -> Dict[str, List[int]]:
        if self.tokenizer._pad_token is None:
            # Some tokenizers (e.g. GPT2 tokenizer) have no padding token which
            # causes bugs
            raise RuntimeError(
                "If tokenizing on-the-fly, tokenizer must have a pad_token_id"
            )

        return self.tokenizer(
            text_sample["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_seq_len,
        )

    def _read_binary_tokenized_sample(self, sample: Dict[str, Any]) -> torch.Tensor:
        return torch.from_numpy(
            np.frombuffer(sample["tokens"], dtype=np.int64)[: self.max_seq_len].copy()
        )

    # How to process a sample
    def __getitem__(
        self, idx: int | slice | List[int] | NDArray[np.int64]
    ) -> Union[Dict[str, List[int]], torch.Tensor]:
        """Implement dataset interface."""
        sample = super().__getitem__(idx)
        if "text" in sample:
            token_sample = self._tokenize(sample)
        elif "tokens" in sample:
            token_sample = self._read_binary_tokenized_sample(sample)
        else:
            raise RuntimeError(
                "StreamingTextDataset needs samples to have a `text` or `tokens` column"
            )
        return token_sample


class ConcatenatedSequenceCollatorWrapper:
    """Collator wrapper to add sequence_id to batch."""

    def __init__(
        self,
        base_collator: Callable,
        eos_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
    ):
        self.base_collator = base_collator
        if (eos_token_id is None) and (bos_token_id is None):
            raise ValueError(
                "Must supply a value for either eos_token_id or bos_token_id, but got"
                "None for both."
            )
        if (eos_token_id is not None) and (bos_token_id is not None):
            raise ValueError(
                "Cannot use *both* EOS and BOS tokens for detecting sequence boundaries"
                ". Please supply `eos_token_id` if sequences end with an EOS token, or"
                "use `bos_token_id` if sequences start with a BOS token."
            )

        if eos_token_id is None:
            self.split_token_id = cast(int, bos_token_id)
            self.bos_mode = True
        else:
            self.split_token_id = eos_token_id
            self.bos_mode = False

    def __call__(self, examples: List[Any]) -> Dict[str, torch.Tensor]:
        """Implement collator interface."""
        batch = self.base_collator(examples)
        batch["sequence_id"] = self.get_sequence_id_from_batch(batch)
        return batch

    def get_sequence_id_from_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Get sequence id from batch."""
        is_separator = torch.eq(batch["input_ids"], self.split_token_id)
        cumulative_sep = torch.cumsum(is_separator, dim=1).to(batch["input_ids"].dtype)
        # If separator token is bos, we're already done
        if self.bos_mode:
            return cumulative_sep

        # If separator token is eos, right shift 1 space
        left_zeros = cumulative_sep.new_zeros((cumulative_sep.shape[0], 1))
        return torch.cat([left_zeros, cumulative_sep[:, :-1]], dim=1)


def build_text_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int,
) -> DataSpec:
    """Build a dataloader for text data."""
    assert cfg.name == "text", f"Tried to build text dataloader w/ cfg.name={cfg.name}"
    if cfg.dataset.get("group_method", None) is not None:
        raise NotImplementedError(
            "group_method is deprecated and has been removed.\nTo "
            "concatenate, use the --concat_tokens "
            "argument when creating your MDS dataset with convert_dataset_hf.py"
        )

    # get kwargs
    streams_dict = cfg.dataset.pop("streams", None)
    mlm_probability = cfg.dataset.pop("mlm_probability", None)
    eos_token_id = cfg.dataset.pop("eos_token_id", None)
    bos_token_id = cfg.dataset.pop("bos_token_id", None)

    # build streams
    streams = None
    if streams_dict is not None:
        streams = []
        for _, stream in streams_dict.items():
            # stream is the streams kwargs
            # fwd all kwargs with **stream allows streaming to check args
            streams.append(Stream(**stream))

    # build dataset potentially with streams
    dataset = StreamingTextDataset(
        tokenizer=tokenizer,
        streams=streams,
        batch_size=device_batch_size,
        **cfg.dataset,
    )

    collate_fn = transformers.DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer,
        mlm=mlm_probability is not None,
        mlm_probability=mlm_probability,
    )

    if (eos_token_id is not None) or (bos_token_id is not None):
        # Note: Will raise an error if both are non-None
        collate_fn = ConcatenatedSequenceCollatorWrapper(
            base_collator=collate_fn,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
        )

    dl = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
    )

    # If we pretokenized, we may not have padding, in which case the
    # tokenizer may not have a pad_token_id. In this case, we can
    # just use the default token counting function. This is correct
    # because we do not support training on pretokenized data with padding,
    # and if tokenizing on the fly, we require that the tokenizer has a pad token.
    token_counting_func = None
    if tokenizer.pad_token_id is not None:
        token_counting_func = get_tokens_per_batch_func(
            pad_token_id=tokenizer.pad_token_id
        )

    return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)


def get_tokens_per_batch_func(
    pad_token_id: int, decoder_only: bool = True
) -> Callable[[Batch], int]:
    """Return a callable that counts the number of tokens in a batch.

    Args:
        pad_token_id (int): The id of the padding token.
        decoder_only (bool, optional): Whether to expect the batch to just contain
            ``input_ids`` (decoder only)or to also contain ``decoder_input_ids``
            (encoder decoder). Defaults to ``True``.

    Returns
    -------
        Callable[[Batch], int]: A callable that counts the number of tokens in a batch.
    """

    def get_num_samples_in_batch(batch: Batch) -> int:
        if not isinstance(batch, Mapping) or "input_ids" not in batch:
            raise ValueError(
                "get_tokens_per_batch_func() requires a batch with an input_ids key"
            )

        if not decoder_only and "decoder_input_ids" not in batch:
            raise ValueError(
                "get_tokens_per_batch_func() for encoder decoder requires a batch with"
                "a decoder_input_ids key"
            )

        # Count number of non padding tokens in batch
        input_ids_tokens = int(torch.sum(batch["input_ids"] != pad_token_id).item())

        # For encoder decoder models only
        decoder_input_ids_tokens = 0
        if not decoder_only:
            decoder_input_ids_tokens = int(
                torch.sum(batch["decoder_input_ids"] != pad_token_id).item()
            )

        return input_ids_tokens + decoder_input_ids_tokens

    return get_num_samples_in_batch


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Test for the StreamingTextDataset."""
    from llmfoundry.utils.builders import build_tokenizer

    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    # Elaborating data path
    if _llm_config.data_remote is not None:
        log(
            INFO,
            f"Reading {_llm_config.eval_loader.dataset.split} split"
            # f"Reading {_llm_config.train_loader.dataset.split} split"
            f" from {_llm_config.data_local} <-"
            f" streamed from <- {_llm_config.data_remote}",
        )
    else:
        log(
            INFO,
            f"Reading {_llm_config.eval_loader.dataset.split} split"
            # f"Reading {_llm_config.train_loader.dataset.split} split"
            f" from {_llm_config.data_local}",
        )
    # Manually setting the batch size to 2
    device_batch_size = 2
    # Build tokenizer
    tokenizer_config: Dict[str, Any] = pop_config(
        _llm_config, "tokenizer", must_exist=True, convert=True
    )
    tokenizer_name = tokenizer_config["name"]
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)
    # Build dataloader and assertions
    _llm_config.eval_loader.dataset.keep_zip = True
    loader = build_text_dataloader(
        _llm_config.eval_loader, tokenizer, device_batch_size
    ).dataloader
    # _llm_config.train_loader.dataset.keep_zip = True
    # loader = build_text_dataloader(
    #     _llm_config.train_loader, tokenizer, device_batch_size
    # ).dataloader
    assert isinstance(loader, DataLoader)
    assert isinstance(loader.dataset, StreamingTextDataset)
    tokenizer = loader.dataset.tokenizer
    # Loop over the first 5 batches and print the samples of each batch
    for batch_ix, batch in enumerate(islice(loader, 5)):
        log(INFO, "\n")
        log(INFO, "%s %s %s", "#" * 20, f"Batch {batch_ix}", "#" * 20)
        for k, v in batch.items():
            log(INFO, "%s %s %s", k, v.shape, v.dtype)
        for sample_ix, token_sample in enumerate(batch["input_ids"]):
            log(
                INFO,
                "\n%s %s %s\n%s",
                "-" * 20,
                f" Sample {sample_ix} ",
                "-" * 20,
                tokenizer.decode(token_sample),
            )


if __name__ == "__main__":
    main()
