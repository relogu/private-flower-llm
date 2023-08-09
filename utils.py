import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import multiprocess as mp
import pandas as pd
import torch
from flwr.common import Metrics, NDArrays, Scalar
from flwr.server.strategy.aggregate import aggregate
from multiprocess import Queue
from torch.utils.data import DataLoader

from datasets import SHAKESPEARE_DTYPES, ShakespeareDataset
from models import ShakespeareLeafNet


#### Server ####
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def partially_aggregate(
    current_agg: Tuple[NDArrays, int, float], new_results: Tuple[NDArrays, int, float]
) -> Tuple[NDArrays, int, float]:
    """Partially aggregate parameters."""
    # a = time.time_ns()
    updated_agg = None
    if current_agg[0] is None:  # first time
        updated_agg = new_results[0]
        total_num_examples = new_results[1]
        weighted_accuracy = new_results[2]
    else:
        updated_agg = aggregate([current_agg[:2], new_results[:2]])
        total_num_examples = current_agg[1] + new_results[1]
        weighted_accuracy = (
            current_agg[1] * current_agg[2] + new_results[1] * new_results[2]
        ) / total_num_examples
    # print(f"Partially aggregated in {(time.time_ns() - a) / 1e9} seconds")
    return updated_agg, total_num_examples, weighted_accuracy


#### Client ####
## General
def set_parameters(net: torch.nn.Module, parameters, device):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    return net


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


def gen_shakespeare_dataset_train_fn(data_root: str):
    return ShakespeareDataset(root=data_root, client_id=3, dataset="train")


def shakespeare_gen_client_fit_fn(
    data_root: str,
    batch_size: int,
    num_local_epochs_per_round: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    # ) -> Callable[[str, NDArrays, int, Dict[str, Scalar], Queue], None]:
) -> Callable[[str, NDArrays, int, Dict[str, Scalar]], None]:
    def client_fit_fn(
        cid: str,
        parameters: NDArrays,
        device: str,
        # results_queue: Queue,
    ):
        """Train the model on the training set."""
        net = ShakespeareLeafNet()
        net = set_parameters(net, parameters, device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        trainset = ShakespeareDataset(root=data_root, client_id=cid, dataset="train")
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        for _ in range(num_local_epochs_per_round):
            num_samples = 0
            num_correct = 0
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                num_samples += len(labels)
                optimizer.zero_grad()
                predicitons = net(inputs)
                num_correct += (
                    (torch.max(predicitons.data, 1)[1] == labels).sum().item()
                )
                criterion(predicitons, labels.to(device)).backward()
                optimizer.step()
        these_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        accuracy = num_correct / num_samples
        # results_queue.put((these_weights, num_samples, accuracy))
        return (these_weights, num_samples, accuracy)

    return client_fit_fn


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
