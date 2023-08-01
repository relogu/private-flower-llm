import warnings
from collections import OrderedDict
from itertools import repeat
from typing import Dict, List, Optional, Tuple


import flwr as fl
import nvsmi
import psutil
import torch
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)

import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Config, Scalar, NDArrays
from flwr.server.strategy.aggregate import aggregate
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import CIFAR10

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# results = mp.Queue()


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


def train(cid, parameters, gpu_id, results):
    """Train the model on the training set."""
    print(f"Train {cid}")
    device = f"cuda:{gpu_id}"
    net = set_parameters(parameters, device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(1):
        num_samples = 0
        for images, labels in trainloader:
            num_samples += len(labels)
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    these_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    results.put((these_weights, num_samples))


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)


def set_parameters(parameters, device):
    net = Net()
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    return net


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
trainloader, testloader = load_data()


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self.queue: mp.Queue = mp.Queue()
        self.all_gpus = nvsmi.get_gpus()
        self.pool: Optional[Dict[str, mp.Pool]] = {
            gpu.id: mp.Pool(4) for gpu in self.all_gpus
        }

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
        total_virtual_clients = 10
        with mp.Manager() as manager:
            results = manager.Queue()
            for gpu_id, p in self.pool.items():
                list_ids = [i for i in range(total_virtual_clients)]  # Gets from config
                tasks = list(
                    zip(list_ids, repeat(parameters), repeat(gpu_id), repeat(results))
                )
                p.starmap(train, tasks)
            # Partial aggregation
            part_agg_weights = (None, 0)
            for _ in range(total_virtual_clients):  # Length of results will vary!!!
                print("hello")
                part_agg_weights = partially_aggregate(part_agg_weights, results.get())
            print(part_agg_weights[0])
            print(part_agg_weights[1])

        return part_agg_weights[0], part_agg_weights[1], {}

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.pool is not None:
            for p in self.pool.values():
                p.close()


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=NodeManager(),
    )
