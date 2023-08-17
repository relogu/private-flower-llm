from collections import OrderedDict
import os
import pickle
import time
import cloudpickle

cloudpickle.DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
from flwr.client import NumPyClient

pickle.Pickler = cloudpickle.Pickler
import warnings
from itertools import repeat
from typing import Callable, Dict, List, Optional, Tuple
from nvsmi import GPU

import flwr as fl
import hydra
import nvsmi
import psutil
import torch
from copy import deepcopy

import multiprocess as mp
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor, as_completed

# mp.set_start_method("spawn", force=True)

from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from torch.utils.data import DataLoader

from utils import partially_aggregate, set_parameters

warnings.filterwarnings("ignore", category=UserWarning)

from datasets.shakespeare import SHAKESPEARE_LOADED as ShakespeareDataset
from models.shakespeare_leaf_model import ShakespeareLeafNet as Net

from multiprocess import shared_memory
import numpy as np


def create_sm(parameters: NDArrays, name="pollen_sm"):
    # Buffer
    buff = [x.tobytes() for x in parameters]
    # Copy the data into shared memory
    shl = shared_memory.ShareableList(buff, name=name)

    return shl


def update_sm(parameters: NDArrays, name="pollen_sm"):
    shl = shared_memory.ShareableList(name=name)
    shl.shm = [x.tobytes() for x in parameters]
    return shl


def read_from_sm(
    sm_list: shared_memory.ShareableList, parameters: NDArrays
) -> NDArrays:
    new_parameters = [
        # np.frombuffer(x, dtype=np.float32).reshape(y.shape)
        np.ndarray(y.shape, dtype=np.float32, buffer=x)
        for x, y in zip(sm_list, parameters)
    ]

    return new_parameters


def test(parameters, device):
    """Validate the model on the test set."""
    net = Net()
    net = set_parameters(net, parameters, device)
    criterion = torch.nn.CrossEntropyLoss()
    testset = ShakespeareDataset(
        root="/datasets/FedScale/leaf_shakespeare", client_id=0, dataset="test"
    )
    testloader = DataLoader(testset, batch_size=4, shuffle=True)
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


class Worker(mp.Process):
    def __init__(
        self,
        client_fn,
        device,
        task_queue: mp.Queue,
        models_queue: mp.Queue,
        result_queue: mp.Queue,
    ):
        super(Worker, self).__init__()
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue: mp.Queue = task_queue
        self.models_queue: mp.Queue = models_queue
        self.result_queue: mp.Queue = result_queue
        self.local_parameters = []
        self.partial_agg_model = (None, 0, 0.0)
        self.current_round = 0

    def process_task(self, task):
        # REMOVE NEED FOR NEW WEIGHTS
        config = task
        if config["server_round"] != self.current_round:
            self.current_round = config["server_round"]
            self.local_parameters = self.models_queue.get()
        config["device"] = self.device
        temp_client = self.client_fn(config["client_id"])
        trained_weights, num_samples, metrics = temp_client.fit(
            self.local_parameters, config
        )

        new_results = (trained_weights, num_samples, metrics["accuracy"])
        self.result_queue.put(new_results)

    def run(self):
        # Check if still work to do
        for task in iter(self.task_queue.get, None):
            self.process_task(task)


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self, client_fn) -> None:
        super().__init__()
        self.all_gpus: List[GPU] = nvsmi.get_gpus()

        # One task_queue per GPU
        self.task_queues = {f"cuda:{gpu.id}": mp.Queue() for gpu in self.all_gpus}

        # One model per worker
        self.models_queue = mp.Queue()

        # Results from worker
        self.result_queue = mp.Queue()

        self.workers = {
            "cuda:0": [
                Worker(
                    client_fn,
                    "cuda:0",
                    self.task_queues[f"cuda:{0}"],
                    self.models_queue,
                    self.result_queue,
                )
                for _ in range(10)
            ]
        }

    def get_cpu_prop(self) -> Dict[str, float]:
        node_prop = {
            "cpu_num": mp.cpu_count(),
            "cpu_ram_total": psutil.virtual_memory().total,
            "cpu_ram_available": psutil.virtual_memory().available,
        }
        return node_prop

    def get_gpus_prop(self) -> Dict[str, float]:
        gpus_prop = {}
        for gpu in nvsmi.get_gpus():
            if gpu["memory.free"] > 0:
                gpus_prop[f"cuda:{gpu.id}_ram_total"] = gpu.mem_total
        return gpus_prop

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        cpu_prop = self.get_cpu_prop()
        gpus_prop = self.get_gpus_prop()
        return dict(cpu_prop, **gpus_prop)

    def get_parameters(sef, config):
        net = Net()
        net.eval()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def start_workers(self, config):
        for worker_list in self.workers.values():
            for worker in worker_list:
                worker.start()

    def fit(self, parameters, config):
        node_part_agg = (None, 0, 0.0)
        num_total_virtual_clients = 0

        # remove need to send parameters same worker, same parameters
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            num_total_virtual_clients += len(list_ids_for_this_gpu)

            # One set of parameters per worker
            for _ in range(len(self.workers[device])):
                self.models_queue.put(parameters)

            for cid in list_ids_for_this_gpu:
                this_config = deepcopy(config)
                this_config["client_id"] = cid
                self.task_queues[device].put((this_config))

            # Start workers move this outside
            if config["server_round"] == 1:
                self.start_workers(config)

        for _ in range(num_total_virtual_clients):
            new_results = self.result_queue.get()
            node_part_agg = partially_aggregate(node_part_agg, new_results)

        return (
            node_part_agg[0],
            node_part_agg[1],
            {"accuracy": node_part_agg[2]},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.workers is not None:
            for device, list_of_workers in self.workers.items():
                [
                    self.task_queues[device].put(None)
                    for _ in range(len(list_of_workers))
                ]


# global initialization
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Define Node Manager
    node_manager = NodeManager(client_fn=call(cfg.gen_client_fn))

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=cfg.flwr_address,
        client=node_manager,
    )


if __name__ == "__main__":
    main()
