import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pandas as pd
import ray
import torch
import wandb
from flwr.common import Metrics, NDArrays, Scalar
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from datasets.shakespeare import SHAKESPEARE_DTYPES
from datasets.shakespeare import SHAKESPEARE_LOADED as ShakespeareDataset


#### Server ####
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def partially_aggregate(
    current_agg: Tuple[NDArrays, int], new_results: Tuple[NDArrays, int]
) -> Tuple[NDArrays, int]:
    """Partially aggregate parameters."""
    updated_agg = None
    if (current_agg[0] is None) or (current_agg[1] == 0):  # first time
        updated_agg = new_results[0]
        total_num_examples = new_results[1]
    else:
        updated_agg = aggregate([current_agg, new_results])
        total_num_examples = current_agg[1] + new_results[1]
    return updated_agg, total_num_examples


def partially_aggregate_with_metrics(
    current_agg: Tuple[NDArrays, int, float, float],
    new_results: Tuple[NDArrays, int, float, float],
) -> Tuple[NDArrays, int, float, float]:
    """Partially aggregate parameters."""
    updated_agg = None
    if (current_agg[0] is None) or (current_agg[1] == 0):  # first time
        updated_agg = new_results[0]
        total_num_examples = new_results[1]
        train_loss = new_results[2]
        train_accuracy = new_results[3]
    else:
        updated_agg = aggregate(
            [(current_agg[0], current_agg[1]), (new_results[0], new_results[1])]
        )
        total_num_examples = current_agg[1] + new_results[1]
        train_loss = weighted_loss_avg(
            [(current_agg[1], current_agg[2]), (new_results[1], new_results[2])]
        )
        train_accuracy = weighted_loss_avg(
            [(current_agg[1], current_agg[3]), (new_results[1], new_results[3])]
        )
    return updated_agg, total_num_examples, train_loss, train_accuracy


#### Client ####
## General
def get_parameters(net: torch.nn.Module) -> NDArrays:
    net.eval()
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(
    net: torch.nn.Module, parameters: NDArrays, device: str = "cpu"
) -> None:
    net.eval()
    keys = [k for k in net.state_dict().keys() if "bn" not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, device=device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict=state_dict, strict=False)


## Shakespeare
def shakespeare_gen_num_total_virtual_clients(
    data_root: str,
    dataset: str = "train",
    min_samples_per_client=20,
) -> List[str]:
    dataframe = pd.read_csv(
        Path(data_root) / "client_data_mapping" / f"{dataset}.csv",
        engine="pyarrow",  # NOSONAR
        dtype=SHAKESPEARE_DTYPES,
        names=list(SHAKESPEARE_DTYPES.keys()),
        sep=",",
        header=0,
    )
    clients = {}
    for client_id in pd.unique(dataframe["client_id"]):
        tmp = dataframe[dataframe["client_id"] == client_id]
        clients[client_id] = len(tmp)
    print(f"Length of cids list before filtering {len(list(clients.keys()))}")
    if min_samples_per_client > 0:
        for client_id in pd.unique(dataframe["client_id"]):
            if clients[client_id] <= min_samples_per_client:
                del clients[client_id]
    print(f"Length of cids list after filtering {len(list(clients.keys()))}")
    # return dict(sorted(clients.items(), key=lambda item: item[1], reverse=True))
    return list(clients.keys())


def gen_shakespeare_dataset_train_fn(data_root: Path, dataset_type: str = "train"):
    def shakespeare_gen_local_dataset_fn(client_id: str):
        return ShakespeareDataset(
            root=data_root, client_id=client_id, dataset=dataset_type
        )

    return shakespeare_gen_local_dataset_fn


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


def gen_on_fit_config_fn(
    batch_size,
    local_epochs,
    learning_rate,
    momentum,
    weight_decay,
    is_fake,
) -> Callable[[int], Dict[str, Scalar]]:
    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        """Return `Config` for fit/evaluate rounds."""
        return {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "server_round": server_round,
            "is_fake": is_fake,
            # TODO: Brainstorm how to set this hyperparameter
            "n_workers": 0,
        }

    return on_fit_config_fn


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""


def wandb_init(wandb_enabled: bool, *args, **kwargs):
    """Initialize wandb if enabled."""
    if wandb_enabled:
        return wandb.init(*args, **kwargs)

    return NoOpContextManager()


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self):
        """Initialize the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""

        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path()  # type: ignore
            )
            ray.shutdown()
            directory_size = shutil.disk_usage(temp_dir).used
            shutil.rmtree(temp_dir)
            print(
                f"Cleaned up ray temp session: {temp_dir} with size: {directory_size}"
            )
