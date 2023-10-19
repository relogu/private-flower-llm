"""Utility functions for FL and experiment management.

They assure compatibility with the Flower and wandb APIs.
"""

import shutil
from collections import OrderedDict, defaultdict
from functools import reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import ray
import torch
import wandb
from flwr.common import FitRes, Metrics, NDArrays, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg


#### Server ####
def weighted_average(metrics: List[Tuple[int, Dict]]) -> Metrics:
    """Implement weighted average."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def partially_aggregate(
    current_agg: Tuple[NDArrays, int], new_results: Tuple[NDArrays, int]
) -> Tuple[NDArrays, int]:
    """Aggregate partially parameters."""
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
    """Aggregate partially parameters with metrics."""
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
    """Implement generic `get_parameters` for Flower Client."""
    net.eval()
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(
    net: torch.nn.Module, parameters: NDArrays, device: str = "cpu"
) -> None:
    """Implement generic `set_parameters` for Flower Client."""
    net.eval()
    keys = [k for k in net.state_dict().keys() if "bn" not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, device=device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict=state_dict, strict=False)


def invert_many_to_one_dictionary(
    input: Dict,
) -> Dict:
    """Invert the mapping given by a dictionary when it is many-to-one."""
    output: Dict = defaultdict(list)
    for k, v in input.items():
        output[v] = output.get(v, []) + [k]
    return output


def invert_one_to_many_dictionary(
    input: Dict,
) -> Dict:
    """Invert the mapping given by a dictionary when it is one-to-many."""
    output: Dict = {}
    for k, v in input.items():
        for w in v:
            output[w] = k
    return output


def gen_on_fit_config_fn(
    batch_size: int = 10,
    local_epochs: int = 1,
    learning_rate: float = 0.1,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    is_fake: bool = False,
    n_workers: int = 0,
) -> Callable[[int], Dict[str, Scalar]]:
    """Return generic `on_fit_config_fn` for Flower Client."""

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
            "n_workers": n_workers,
        }

    return on_fit_config_fn


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""


def wandb_init(
    wandb_enabled: bool, *args, **kwargs
) -> Optional[Union[NoOpContextManager, Any]]:
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


def chunks_idx(list: Sequence, n_chunks: int) -> Generator[tuple[int, int], Any, None]:
    """Split a list in n_chunks of equal length."""
    d, r = divmod(len(list), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def aggregate_inplace(
    results: List[Tuple[ClientProxy, FitRes]]
) -> Tuple[NDArrays, int]:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum([fit_res.num_examples for _, fit_res in results])

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / num_examples_total for _, fit_res in results
    ]

    # Let's do in-place aggregation
    # get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params, num_examples_total
