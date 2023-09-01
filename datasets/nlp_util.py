# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import gc
import os
import pickle
import time
from logging import ERROR, INFO
from multiprocessing import Pool
from typing import List, Tuple

import psutil
import torch
from flwr.common.logger import log
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def chunks_idx(l, n):
    d, r = divmod(len(l), n)
    for i in range(n):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def get_collate_fn(tokenizer: PreTrainedTokenizer):
    def collate_fn(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    return collate_fn


def feature_creation_worker(
    indices: List[int],
    files: List[str],
    tokenizer: PreTrainedTokenizer,
    block_size: int,
    worker_idx: int,
    file_path: str,
    model: str,
):
    start_time = time.time()
    for i, (idx, file) in enumerate(zip(indices, files)):
        try:
            # Read text file, one file per `user_id``, apparently
            with open(file, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # Tokenize the text to tokens
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            examples = []
            # Truncate in blocks of length `block_size``
            for j in range(0, len(tokenized_text) - block_size + 1, block_size):
                # Append the block of tokens
                examples.append(
                    tokenizer.build_inputs_with_special_tokens(
                        tokenized_text[j : j + block_size]
                    )
                )
            if len(examples) > 0:
                _cached_features_file = os.path.join(
                    file_path, model + "_cached_lm_" + str(block_size) + "_" + str(idx)
                )
                with open(_cached_features_file, "wb") as f:
                    pickle.dump(examples, f)
        except Exception as e:
            log(ERROR, f"Worker {worker_idx}: fail due to {e}")
        if i % 10000 == 0:
            log(
                INFO,
                f"Worker {worker_idx}: {len(files)-i} files left, {i} files complete, remaining time {(time.time()-start_time)/(i+1)*(len(files)-i)}",
            )
            gc.collect()


class TextDataset(Dataset):
    def __init__(
        self,
        model: str,
        tokenizer: PreTrainedTokenizer,
        file_path: str,
        examples: List[List[int]],
        n_jobs: int = 1,
        overwrite_cache: bool = False,
        block_size: int = 64,
        client_id: int = 0,
    ):
        # Correct the block size for building sequences of tokens
        block_size = block_size - (
            tokenizer.model_max_length - tokenizer.max_len_single_sentence
        )
        # Create the cached features file
        self.cached_features_file = os.path.join(
            file_path, model + "_cached_lm_" + str(block_size) + "_" + str(client_id)
        )
        # Set the number of jobs
        try:
            cpus = len(psutil.Process().cpu_affinity())
        except AttributeError:
            cpus = psutil.cpu_count()
        if n_jobs > cpus:
            n_jobs = cpus
        # Get the features
        if examples is not None:
            # If the features are passed as parameter, use them
            self.examples = examples
        elif os.path.exists(self.cached_features_file) and not overwrite_cache:
            # If the features are stored, load them
            # log(INFO, "Loading features from cached file %s", self.cached_features_file)
            gc.disable()
            with open(self.cached_features_file, "rb") as f:
                self.examples = pickle.load(f)
            gc.enable()
        else:
            # Otherwise, create them
            log(INFO, "Requested features file doesn't exist")
            ## Tokenisation
            # Get the list of files containing raw data (excluding the cached files)
            files = [
                entry.name
                for entry in os.scandir(file_path)
                if "_cached_lm_" not in entry.name
            ]
            # Make sure files are ordered
            files = [os.path.join(file_path, x) for x in sorted(files)]
            if client_id < 0:
                log(
                    INFO,
                    "Creating features from dataset file at %s for the entire dataset",
                    file_path,
                )
                # Parallelise entire dataset tokenisation
                pool_inputs = []
                pool = Pool(n_jobs)
                worker_cnt = 0
                for begin, end in chunks_idx(range(len(files)), n_jobs):
                    pool_inputs.append(
                        [
                            list(range(len(files)))[begin:end],
                            files[begin:end],
                            tokenizer,
                            block_size,
                            worker_cnt,
                            file_path,
                            model,
                        ]
                    )
                    worker_cnt += 1
                pool.starmap(feature_creation_worker, pool_inputs)
                pool.close()
                pool.join()
            elif client_id >= len(files):
                raise ValueError(f"Client id {client_id} is out of range")
            else:
                log(
                    INFO,
                    "Creating features from dataset file at %s for the client %s",
                    file_path,
                    client_id,
                )
                # Single client tokenisation
                feature_creation_worker(
                    [client_id],
                    [files[client_id]],
                    tokenizer,
                    block_size,
                    0,
                    file_path,
                    model,
                )

            gc.disable()
            with open(self.cached_features_file, "rb") as f:
                self.examples = pickle.load(f)
            gc.enable()

        self.targets = [0] * len(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(
    model: str,
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    n_jobs: int = 1,
    block_size: int = 64,
    evaluate: bool = False,
):
    """Perform the tokenisation and caching of the entire dataset."""
    file_path = (
        os.path.join(data_dir, "test") if evaluate else os.path.join(data_dir, "train")
    )

    return TextDataset(
        model,
        tokenizer,
        examples=None,
        n_jobs=n_jobs,
        file_path=file_path,
        overwrite_cache=False,
        block_size=block_size,
        client_id=-1,
    )


def mask_tokens(
    inputs: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    mlm_probability: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    labels = inputs.clone().to(device=device)
    # We sample a few tokens in each sequence for masked-LM training (with probability mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability, device=device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(
        torch.tensor(special_tokens_mask, dtype=torch.bool, device=device), value=0.0
    )
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = (
        torch.tensor(torch.bernoulli(probability_matrix), dtype=torch.bool)
        .detach()
        .to(device=device)
    )
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # FIXME: the following warning is printed in the logs
    # `$FEDSCALE_HOME/fedscale/dataloaders/nlp.py:195: UserWarning: To copy construct from a tensor,
    # it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True),
    # rather than torch.tensor(sourceTensor).labels.shape, 0.8)), dtype=torch.bool, device=device) & masked_indices`
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.tensor(
            torch.bernoulli(torch.full(labels.shape, 0.8)),
            dtype=torch.bool,
            device=device,
        )
        & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.tensor(
            torch.bernoulli(torch.full(labels.shape, 0.5)),
            dtype=torch.bool,
            device=device,
        )
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    bool_indices_random = indices_random
    inputs[bool_indices_random] = random_words[bool_indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


if __name__ == "__main__":
    import time

    import pandas as pd
    from transformers import AlbertTokenizer

    model = "albert-base-v2"
    features_files = [
        entry.name
        for entry in os.scandir("/datasets/FedScale/reddit/reddit/train")
        if "_cached_lm_62" in entry.name
    ]
    raw_files = [
        entry.name
        for entry in os.scandir("/datasets/FedScale/reddit/reddit/train")
        if "_cached_lm_62" not in entry.name
    ]
    log(
        INFO,
        f"Found {len(features_files)} features files and {len(raw_files)} raw files",
    )

    # Set the number of jobs
    n_jobs = 100
    tokenizer = AlbertTokenizer.from_pretrained(model, do_lower_case=True)
    file_path = "/datasets/FedScale/reddit/reddit/train"
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus

    def dump_info(model, tokenizer, file_path, client_ids, worker_idx):
        clients = []
        start_time = time.time()
        for i, client_id in enumerate(client_ids):
            cached_features_file = os.path.join(
                file_path, model + "_cached_lm_" + str(62) + "_" + str(client_id)
            )
            if os.path.exists(cached_features_file):
                ds = TextDataset(
                    model=model,
                    tokenizer=tokenizer,
                    file_path=file_path,
                    client_id=client_id,
                    examples=None,
                )
                clients.append((client_id, ds.cached_features_file, len(ds)))
            if i % 1000 == 0:
                log(
                    INFO,
                    f"Worker {worker_idx}: {len(client_ids)-i} client_ids left, {i} client_ids complete, remaining time {(time.time()-start_time)/(i+1)*(len(client_ids)-i)}",
                )
        return clients

    # Parallelise the tokenisation
    pool_inputs = []
    pool = Pool(n_jobs)
    client_ids = list(range(len(raw_files)))
    cnt = 0
    for begin, end in chunks_idx(range(len(raw_files)), n_jobs):
        pool_inputs.append([model, tokenizer, file_path, client_ids[begin:end], cnt])
        cnt += 1
    pool_outputs = pool.starmap(dump_info, pool_inputs)
    pool.close()
    pool.join()
    log(INFO, f"Pool outputs length: {len(pool_outputs)}")
    clients = []
    [clients.extend(out) for out in pool_outputs]
    log(INFO, f"Pool outputs concat length: {len(clients)}")

    df = pd.DataFrame(clients, columns=["client_id", "sample_path", "samples"])
    log(INFO, f"Dataframe: {df.head()}")
    df.to_parquet(
        "/datasets/FedScale/reddit/reddit/client_data_mapping/clients_dict.parquet"
    )
    s_t = time.time()
    df = pd.read_parquet(
        "/datasets/FedScale/reddit/reddit/client_data_mapping/clients_dict.parquet"
    )
    log(INFO, f"Dataframe: {df.head()}")
    log(INFO, f"Read parquet file in {time.time()-s_t} seconds")
    s_t = time.time()
    samples = []
    for i in client_ids:
        samples.append(int(df[df["client_id"] == i]["samples"]))
    log(INFO, f"Getting samples of clients from 0 to 100: {samples}")
    log(INFO, f"Getting samples took {time.time()-s_t} seconds")
