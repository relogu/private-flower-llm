"""Utility functions for Pollen.

It contains both task-independent utility functions and task-specific ones for the
Pollen paper.
"""
# TODO: split the codebase into task-specific and task-independent units.
from argparse import ArgumentTypeError
from functools import reduce
from logging import DEBUG, INFO
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pandas as pd
import psutil
import pyarrow as pa
import torch
from flwr.common.logger import log
from flwr.common.typing import NDArrays
from flwr.server.strategy.aggregate import aggregate
from torch import device as device_type
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, Dataset
from transformers import AlbertTokenizer

from pollen_worker.datasets.google_speech import SPEECH
from pollen_worker.datasets.nlp_util import TextDataset
from pollen_worker.datasets.openimage import OpenImage
from pollen_worker.datasets.shakespeare import SHAKESPEARE, SHAKESPEARE_LOADED
from pollen_worker.utils import chunks_idx


def get_device() -> device_type:
    """Determine which device to use for PyTorch.

    Returns
    -------
        str: device for PyTorch
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        torch.backends.mps.is_available()  # type: ignore
        and torch.backends.mps.is_built()  # type: ignore
    ):
        device = "mps"
    return cast(device_type, device)


def get_pyarrow_buffer_from_table(table: pa.Table) -> pa.Buffer:
    """Cast a PyArrow Table into a Buffer."""
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_file(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def get_table_from_pyarrow_buffer(buffer: pa.Buffer) -> pa.Table:
    """Cast a Buffer into a PyArrow Table ."""
    ret_table = None
    with pa.ipc.open_file(buffer) as reader:
        ret_table = reader.read_all()
    return ret_table


def aggregate_pytorch_tensor(
    results: List[Tuple[List[torch.Tensor], int]]
) -> List[torch.Tensor]:
    """Compute weighted average using PyTorch Tensors."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    # NOTE: error happening when a worker controls more than one GPU
    # RuntimeError: Expected all tensors to be on the same device,
    # but found at least two devices, cuda:0 and cuda:1!
    # We have to deal with the worker controlling multiple GPUs
    weights_prime: List[torch.Tensor] = [
        reduce(torch.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def partial_aggregation_pytorch_tensor(
    old_result: Tuple[List[torch.Tensor], int],
    new_result: Tuple[List[torch.Tensor], int],
) -> Tuple[List[torch.Tensor], int]:
    """Compute partial aggregatated FL results through weighted average."""
    # Calculate the total number of examples used during training
    if old_result[0]:  # Not empty
        total_samples = old_result[1] + new_result[1]
        # NOTE: we may want to just add the new result to the old one and
        # then averaging at the server
        temp = aggregate_pytorch_tensor([old_result, new_result])
        old_result = (temp, total_samples)
    else:
        old_result = new_result
    return old_result


def partial_aggregation_NDArrays(
    old_result: Tuple[NDArrays, int],
    new_result: Tuple[NDArrays, int],
) -> Tuple[NDArrays, int]:
    """Compute partial aggregatated FL results through weighted average."""
    # Calculate the total number of examples used during training
    if old_result[0]:  # Not empty
        total_samples = old_result[1] + new_result[1]
        # NOTE: we may want to just add the new result to the old one and
        # then averaging at the server
        temp = aggregate([old_result, new_result])
        old_result = (temp, total_samples)
    else:
        old_result = new_result
    return old_result


def valid_folder(path_str: str) -> Path:
    """Test if a path is a valid FL partition folder.

    Args:
        path_str (str): Path to directory containing train and test folder.

    Returns
    -------
        bool: result of checks
    """
    tmp_path = Path(path_str)
    test = True
    for sub_folder in ["train"]:
        test = test and (tmp_path / sub_folder).exists()
    if not test:
        raise ArgumentTypeError
    return tmp_path


def get_model(name: str) -> Module:
    """Return the model given the task's name."""
    # NOTE: we may want to load this once and then deepcopying it when needed
    if name == "shakespeare":
        from pollen_worker.models.shakespeare_leaf_model import ShakespeareLeafNet

        return ShakespeareLeafNet()
    if name == "shakespeare_memory":
        from pollen_worker.models.shakespeare_leaf_model import ShakespeareLeafNet

        return ShakespeareLeafNet()
    if name == "reddit":
        from transformers import AlbertForMaskedLM

        return AlbertForMaskedLM.from_pretrained("albert-base-v2")  # type: ignore
    if name == "google_speech":
        from pollen_worker.models.resnet_util import resnet34

        return resnet34(num_classes=35, in_channels=1)
    if name == "openimage":
        from torchvision import models

        return models.__dict__["shufflenet_v2_x2_0"](num_classes=596)

    raise ValueError("No model for the requested dataset")


def get_client_ds(
    cid: int,
    dataset_root: Path = Path("/datasets/FedScale/"),
    name: str = "openimage",
    dataset: str = "train",
) -> Tuple[Dataset, Optional[AlbertTokenizer]]:
    """Return the dataset object given the task's name."""
    if name == "shakespeare":
        ds = SHAKESPEARE(
            dataset_root / "leaf_shakespeare", client_id=cid, dataset=dataset
        )
        return ds, None

    if name == "shakespeare_memory":
        ds = SHAKESPEARE_LOADED(
            dataset_root / "leaf_shakespeare", client_id=cid, dataset=dataset
        )
        return ds, None

    if name == "reddit":
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
        ds = TextDataset(
            model="albert-base-v2",
            root_dir=dataset_root / "reddit" / "reddit",  # type: ignore
            tokenizer=tokenizer,
            examples=None,
            n_jobs=100,
            overwrite_cache=False,
            block_size=64,
            client_id=cid,
            dataset=dataset,
        )
        return ds, tokenizer

    if name == "google_speech":
        ds = SPEECH(
            root=dataset_root / "google_speech" / "google_speech",
            client_id=cid,
            dataset=dataset,
        )
        return ds, None

    if name == "openimage":
        ds = OpenImage(root=dataset_root / "openImg", client_id=cid, dataset=dataset)
        return ds, None

    raise ValueError("No dataset for the requested dataset name")


def get_client_ds_fn(
    dataset_root: Path = Path("/datasets/FedScale/"),
    name: str = "openimage",
    dataset: str = "train",
) -> Callable[[int], Tuple[Dataset, Optional[AlbertTokenizer]]]:
    """Return the function that returns the dataset given the task's name."""

    def get_ds_fn(client_id: int) -> Tuple[Dataset[Any], Optional[AlbertTokenizer]]:
        return get_client_ds(
            cid=client_id,
            dataset_root=dataset_root,
            name=name,
            dataset=dataset,
        )

    return get_ds_fn


def get_optimizer(name: str, model: Module) -> Optimizer:
    """Return the optimiser object given the task's name."""
    optimizer = None
    lr = 0.05
    if name == "shakespeare" or name == "shakespeare_memory":
        lr = 0.8
    elif name == "reddit":
        lr = 4e-5

    if name == "reddit":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        # Bert pre-training setup
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=lr, weight_decay=1e-2
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
    return optimizer


def get_clients_population_dict(
    name: str,
    seed: int,
    dataset: str = "train",
    batch_size: int = 20,
) -> Dict[Union[str, int], int]:
    """Return the client-samples mapping given the task's name."""
    dataframe = pd.read_parquet(
        _get_dataset_root(name)
        / "client_data_mapping"
        / f"{dataset}_clients_dict.parquet"
    )
    dataframe = dataframe.set_index("client_id")
    dataframe.samples = dataframe.samples.astype(int)
    dataframe = dataframe.sort_values(by=["samples"], ascending=False)
    log(DEBUG, f"Length of cids list before filtering {len(dataframe)}")
    dataframe = dataframe[dataframe["samples"] >= batch_size]
    log(DEBUG, f"Length of cids list after filtering {len(dataframe)}")
    dataframe = dataframe.sample(frac=1, random_state=seed)
    log(DEBUG, f"Randomly sampled clients with state {seed}")
    return {row.Index: row.samples for row in dataframe.itertuples()}


def _get_dataset_root(name: str) -> Path:
    if name == "reddit":
        return Path("/datasets/FedScale/reddit/reddit")
    if name == "shakespeare" or name == "shakespeare_memory":
        return Path("/datasets/FedScale/leaf_shakespeare")
    if name == "google_speech":
        return Path("/datasets/FedScale/google_speech/google_speech")
    if name == "openimage":
        return Path("/datasets/FedScale/openImg")

    raise ValueError("No dataset for the requested dataset name")


def _get_list_of_clients_ds(
    name: str, cids: List[int], dataset: str
) -> tuple[list[Any], Union[AlbertTokenizer, None]]:
    clients_test_sets = []
    tokenizer = None
    for cid in cids:
        ds, tokenizer = get_client_ds(name=name, cid=cid, dataset=dataset)
        clients_test_sets.append(ds)
    return clients_test_sets, tokenizer


def get_centralised_eval_set(
    name: str,
    seed: int,
    n_clients: Union[int, float],
) -> Tuple[Dataset, Optional[AlbertTokenizer]]:
    """Return the centralised evaluation dataset given the task's name."""
    # Get the list of cids
    cid_samples_dict = get_clients_population_dict(
        name=name,
        batch_size=1,
        dataset="test",
        seed=seed,
    )
    # Set up the parallelisation
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus
    clients_test_sets = []
    pool_inputs = []
    pool = Pool(n_jobs)
    if isinstance(n_clients, int):
        if n_clients > 0:
            client_ids = list(cid_samples_dict.keys())[:n_clients]
        else:
            client_ids = list(cid_samples_dict.keys())
    # If we have a percentage of clients
    # choose the closest integer number of clients
    elif isinstance(n_clients, float):
        if n_clients > 0:
            client_ids = list(cid_samples_dict.keys())[
                : int(n_clients * len(cid_samples_dict))
            ]
        else:
            raise ValueError("n_clients percentage must be greater than 0")
    else:
        raise ValueError("n_clients must be either an int or a float")

    # Split the clients in chunks
    for begin, end in chunks_idx(range(len(client_ids)), n_jobs):
        pool_inputs.append([name, client_ids[begin:end], "test"])
    pool_outputs = pool.starmap(_get_list_of_clients_ds, pool_inputs)
    pool.close()
    pool.join()
    # Retrieve the results fro the pool
    for x in pool_outputs:
        for a in x[0]:
            clients_test_sets.append(a)
    tokenizer = pool_outputs[0][1]
    # Concatenate the clients test sets
    testset = ConcatDataset(clients_test_sets)
    log(INFO, f"Test set size: {len(testset)}")
    return testset, tokenizer
