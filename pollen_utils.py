import pickle
from argparse import ArgumentTypeError
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from flwr.common.typing import Metrics, NDArrays, Scalar
from flwr.server.strategy.aggregate import aggregate
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset
from transformers import AlbertTokenizer

from datasets.google_speech import GOOGLE_SPEECH_DTYPES, SPEECH
from datasets.nlp_util import TextDataset
from datasets.openimage import OPENIMAGE_DTYPES, OpenImage
from datasets.shakespeare import SHAKESPEARE, SHAKESPEARE_DTYPES, SHAKESPEARE_LOADED


def get_device() -> str:
    """Determine which device to use for PyTorch.

    Returns:
        str: device for PyTorch
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


def invert_many_to_one_dictionary(
    input: Dict,
) -> Dict:
    output: Dict = defaultdict(list)
    for k, v in input.items():
        output[v] = output.get(v, []) + [k]
    return output


def invert_one_to_many_dictionary(
    input: Dict,
) -> Dict:
    output: Dict = {}
    for k, v in input.items():
        for w in v:
            output[w] = k
    return output


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


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
    """Tests if a path is a valid FL partition folder

    Args:
                path_str (str): Path to directory containing train and test folder.

    Returns:
                bool: result of checks
    """
    tmp_path = Path(path_str)
    test = True
    for sub_folder in ["train"]:
        test = test and (tmp_path / sub_folder).exists()
    if not test:
        raise ArgumentTypeError
    return tmp_path


def read_all_pickle(path: Path) -> List[List[Any]]:
    out = []
    with open(path, "rb") as f:
        try:
            while True:
                out.append(pickle.load(f))
        except EOFError:
            pass
    return out


def get_ctt_dataframe_from_pickle(path_to_pickle: Path) -> pd.DataFrame:
    """Reads a pickle file and returns a dataframe with the CCT data."""
    functions = {
        0: "init",
        1: "fit",
        2: "end fit-begin aggregate",
        3: "end aggregate",
        4: "end fit",
    }
    worker_out = read_all_pickle(path_to_pickle)
    df = pd.DataFrame(
        np.concatenate(worker_out),
        columns=["fn", "timestamp", "round", "cid", "n_batches"],
        dtype=np.float64,
    )
    df.fn = df.fn.apply(lambda x: functions[x])
    df["timedelta"] = df.timestamp.map(lambda a: a - df.timestamp[0])
    df = (
        df.groupby(["cid", "round", "n_batches"])
        .timestamp.agg(["min", "max"])
        .reset_index()
    )
    df["delta"] = df["max"] - df["min"]
    return df[df.cid != -1]


def get_clients_dataframe_from_dict(
    clients_dict: Dict[int, int], batch_size: int = 20
) -> pd.DataFrame:
    clients_df = pd.DataFrame(clients_dict.items(), columns=["cid", "n_samples"])
    clients_df["n_batches"] = clients_df.n_samples // batch_size
    return clients_df


def merge_ctt_size(
    ctt_df: pd.DataFrame, clients_df: pd.DataFrame
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    dff = ctt_df.merge(clients_df, how="inner", on="cid")
    dff.drop_duplicates(subset=["cid"], inplace=True)
    dff.reset_index(drop=True)
    return dff, dff.n_batches.to_numpy().reshape((-1, 1)), dff.delta.to_numpy()


def get_model(name: str) -> Module:
    # NOTE: we may want to load this once and then deepcopying it when needed
    model = None
    if name == "shakespeare":
        from pollen.models.shakespeare_leaf_model import ShakespeareLeafNet

        model = ShakespeareLeafNet()
    elif name == "shakespeare_memory":
        from pollen.models.shakespeare_leaf_model import ShakespeareLeafNet

        model = ShakespeareLeafNet()
    elif name == "reddit":
        from transformers import AutoConfig, AutoModelWithLMHead

        config = AutoConfig.from_pretrained(
            Path("/datasets/FedScale/reddit/redddit/albert-base-v2-config.json")
        )
        model = AutoModelWithLMHead.from_config(config)
    elif name == "google_speech":
        from pollen.models.resnet_util import resnet34

        model = resnet34(num_classes=35, in_channels=1)
    elif name == "openimage":
        from torchvision import models

        model = models.__dict__["shufflenet_v2_x2_0"](num_classes=596)
    return model


def get_client_ds(
    cid: int,
    dataset_root: Path = Path("/datasets/FedScale/"),
    name: str = "openimage",
    dataset: str = "train",
) -> Tuple[Dataset, Optional[AlbertTokenizer]]:
    ds = []
    tokenizer = None
    if name == "shakespeare":
        # from shakespeare_ds import SHAKESPEARE
        ds = SHAKESPEARE(
            dataset_root / "leaf_shakespeare", client_id=cid, dataset=dataset
        )
    elif name == "shakespeare_memory":
        # NOTE: we may want to load this once because it's loaded in RAM
        # from shakespeare_ds import SHAKESPEARE_LOADED
        ds = SHAKESPEARE_LOADED(
            dataset_root / "leaf_shakespeare", client_id=cid, dataset=dataset
        )
    elif name == "reddit":
        # NOTE: we may want to load this once because it's loaded in RAM,
        # look at the `examples` parameter
        # from transformers import AlbertTokenizer
        # from nlp_util import TextDataset
        tokenizer = AlbertTokenizer.from_pretrained(
            "albert-base-v2", do_lower_case=True
        )
        ds = TextDataset(
            model="albert-base-v2",
            file_path=dataset_root / "reddit" / "reddit" / dataset,
            tokenizer=tokenizer,
            examples=None,
        )
    elif name == "google_speech":
        # from google_speech import SPEECH
        ds = SPEECH(
            root=dataset_root / "google_speech" / "google_speech",
            client_id=cid,
            dataset=dataset,
        )
    elif name == "openimage":
        # from openimage import OpenImage
        ds = OpenImage(root=dataset_root / "openImg", client_id=cid, dataset=dataset)
    return ds, tokenizer


def get_client_ds_fn(
    dataset_root: Path = Path("/datasets/FedScale/"),
    name: str = "openimage",
    dataset: str = "train",
) -> Tuple[Callable[[int], Dataset], Optional[AlbertTokenizer]]:
    ds_fn = []
    tokenizer = None
    if name == "shakespeare":
        from pollen.datasets.shakespeare_ds import SHAKESPEARE

        def ds_fn(client_id):
            return SHAKESPEARE(
                dataset_root / "leaf_shakespeare", client_id=client_id, dataset=dataset
            )

    elif name == "shakespeare_memory":
        # NOTE: we may want to load this once because it's loaded in RAM
        from pollen.datasets.shakespeare_ds import SHAKESPEARE_LOADED

        def ds_fn(client_id):
            return SHAKESPEARE_LOADED(
                dataset_root / "leaf_shakespeare", client_id=client_id, dataset=dataset
            )

    elif name == "reddit":
        # NOTE: we may want to load this once because it's loaded in RAM,
        # look at the `examples` parameter
        from pollen.datasets.nlp_util import TextDataset
        from transformers import AlbertTokenizer

        tokenizer = AlbertTokenizer.from_pretrained(
            "albert-base-v2", do_lower_case=True
        )

        def ds_fn(client_id):
            return TextDataset(
                model="albert-base-v2",
                client_id=client_id,
                file_path=dataset_root / "reddit" / "reddit" / dataset,
                tokenizer=tokenizer,
                examples=None,
            )

    elif name == "google_speech":
        from pollen.datasets.google_speech import SPEECH

        def ds_fn(client_id):
            return SPEECH(
                root=dataset_root / "google_speech" / "google_speech",
                client_id=client_id,
                dataset=dataset,
            )

    elif name == "openimage":
        from pollen.datasets.openimage import OpenImage

        def ds_fn(client_id):
            return OpenImage(
                root=dataset_root / "openImg", client_id=client_id, dataset=dataset
            )

    return ds_fn, tokenizer


def get_optimizer(name: str, model: Module) -> Optimizer:
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
    dataset: str = "train",
    batch_size: int = 20,
) -> Dict[str, int]:
    dataframe = pd.read_csv(
        _get_dataset_root(name) / "client_data_mapping" / (dataset + ".csv"),
        # engine="pyarrow", # NOSONAR
        dtype=_get_name_dtypes(name),
        names=list(_get_name_dtypes(name).keys()),
        sep=",",
        header=0,
    )
    clients = {}
    for client_id in pd.unique(dataframe["client_id"]):
        tmp = dataframe[dataframe["client_id"] == client_id]
        clients[client_id] = len(tmp)
    print(f"Length of cids list before filtering {len(list(clients.keys()))}")
    if batch_size > 0:
        for client_id in pd.unique(dataframe["client_id"]):
            if clients[client_id] <= batch_size:
                del clients[client_id]
    print(f"Length of cids list after filtering {len(list(clients.keys()))}")
    return dict(sorted(clients.items(), key=lambda item: item[1], reverse=True))


def openimg_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config: Dict[str, Scalar] = {
        "batch_size": 20,
        "epochs": 1,
        "current_round": server_round,
        "n_workers": 0,
        "workers_policy": "split",
    }
    return config


def google_speech_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config: Dict[str, Scalar] = {
        "batch_size": 20,
        "epochs": 1,
        "current_round": server_round,
        "n_workers": 0,
        "workers_policy": "split",
    }
    return config


def shakespeare_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config: Dict[str, Scalar] = {
        "batch_size": 4,
        "epochs": 1,
        "current_round": server_round,
        "n_workers": 0,
        "workers_policy": "zero",
    }
    return config


def reddit_fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return training configuration dict for each round."""
    config: Dict[str, Scalar] = {
        "batch_size": 20,
        "epochs": 1,
        "current_round": server_round,
        "n_workers": 0,
        "workers_policy": "zero",
    }
    return config


def _get_name_dtypes(name: str) -> Dict[str, Any]:
    if name == "reddit":
        return {}
    elif name == "shakespeare" or name == "shakespeare_memory":
        return SHAKESPEARE_DTYPES
    elif name == "google_speech":
        return GOOGLE_SPEECH_DTYPES
    elif name == "openimage":
        return OPENIMAGE_DTYPES


def _get_dataset_root(name: str) -> Path:
    if name == "reddit":
        return Path("/datasets/FedScale/reddit/reddit")
    elif name == "shakespeare" or name == "shakespeare_memory":
        return Path("/datasets/FedScale/leaf_shakespeare")
    elif name == "google_speech":
        return Path("/datasets/FedScale/google_speech/google_speech")
    elif name == "openimage":
        return Path("/datasets/FedScale/openImg")


def _get_on_fit_config_fn(name: str) -> Callable[[int], Dict[str, Scalar]]:
    if name == "reddit":
        return reddit_fit_config
    elif name == "shakespeare" or name == "shakespeare_memory":
        return shakespeare_fit_config
    elif name == "google_speech":
        return google_speech_fit_config
    elif name == "openimage":
        return openimg_fit_config
