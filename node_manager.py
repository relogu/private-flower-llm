import warnings
from collections import OrderedDict
from itertools import repeat
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import nvsmi
import psutil
import torch
import torch.multiprocessing as mp
from omegaconf import DictConfig

mp.set_start_method("spawn", force=True)

import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Config, NDArrays, Scalar
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import DataLoader
from hydra.utils import call, get_original_cwd, instantiate, to_absolute_path

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)

from models import ShakespeareLeafNet as Net
from datasets import SHAKESPEARE_LOADED as ShakespeareDataset


def partially_aggregate(
    current_agg: Tuple[NDArrays, int], new_results: Tuple[NDArrays, int]
) -> Tuple[NDArrays, int]:
    """Partially aggregate parameters."""
    updated_agg = None
    if current_agg[0] is None:  # first time
        updated_agg = new_results[0]
        total_num_examples = new_results[1]
    else:
        updated_agg = aggregate([current_agg, new_results])
        total_num_examples = current_agg[1] + new_results[1]
    return updated_agg, total_num_examples


def gen_client_fit_fn(
    data_root: str,
) -> Callable[[str, NDArrays, int, Dict[str, Scalar], mp.Queue], None]:
    def client_fit_fn(
        cid,
        parameters,
        batch_size,
        learning_rate,
        momentum,
        weight_decay,
        gpu_id,
        results_queue,
    ):
        """Train the model on the training set."""
        print(f"Task {cid}")
        device = f"cuda:{gpu_id}"
        net = set_parameters(parameters, device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        trainset = ShakespeareDataset(root=data_root, client_id=cid, dataset="train")
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        for _ in range(1):
            num_samples = 0
            for images, labels in trainloader:
                num_samples += len(labels)
                optimizer.zero_grad()
                criterion(net(images.to(device)), labels.to(device)).backward()
                optimizer.step()
        these_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        results_queue.put((these_weights, num_samples))

    return client_fit_fn


def test(parameters, device):
    """Validate the model on the test set."""
    net = set_parameters(parameters, device)
    criterion = torch.nn.CrossEntropyLoss()
    testset = ShakespeareDataset(root="data", client_id=0, dataset="test")
    testloader = DataLoader(testset, batch_size=16, shuffle=True)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def set_parameters(parameters, device):
    net = Net()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    return net


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self, train_fn) -> None:
        super().__init__()
        self.all_gpus = nvsmi.get_gpus()
        self.pool: Optional[
            Dict[str, mp.Pool]
        ] = {  # Maybe cudatype_cudaid example: a40_0
            gpu.id: mp.Pool(10) for gpu in self.all_gpus
        }
        self.train_fn = train_fn

    def get_node_prop(self) -> Dict[str, float]:
        node_prop = {
            "cpus": mp.cpu_count(),
            "ram_total": psutil.virtual_memory().total,
            "ram_available": psutil.virtual_memory().available,
        }
        return node_prop

    def get_gpus_prop(self) -> Dict[str, float]:
        gpus_prop = {}
        for gpu in nvsmi.get_gpus():
            if gpu["memory.free"] > 0:
                gpus_prop[gpu.uuid] = gpu.mem_total
        return gpus_prop

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        node_prop = self.get_node_prop()
        gpus_prop = self.get_gpus_prop()
        # Can check if config contains updates pool
        return dict(node_prop, **gpus_prop)

    def get_parameters(sef, config):
        net = Net()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        print("Fit")
        total_virtual_clients = 10  # This will come from the config
        with mp.Manager() as manager:
            results_queue = manager.Queue()
            for gpu_id, p in self.pool.items():
                list_ids_for_this_gpu = [
                    i for i in range(total_virtual_clients)
                ]  # Gets from config
                tasks = list(
                    zip(
                        list_ids_for_this_gpu,
                        repeat(parameters),
                        repeat(gpu_id),
                        repeat(results_queue),
                    )
                )
                print(self.train_fn)
                p.starmap_async(train, tasks)
            # Partial aggregation
            part_agg_weights = (None, 0)
            for _ in range(total_virtual_clients):  # Length of results will vary!!!
                part_agg_weights = partially_aggregate(
                    part_agg_weights, results_queue.get()
                )

        return part_agg_weights[0], part_agg_weights[1], {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.pool is not None:
            for p in self.pool.values():
                p.close()


@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load train and evaluate functions
    # client_train_fn = call(cfg.gen_train_fn)
    client_fit_fn = gen_client_fit_fn(
        "/",
    )

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=NodeManager(client_fit_fn=train_fn),
    )


if __name__ == "__main__":
    main()
