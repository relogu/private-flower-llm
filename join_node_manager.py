import os
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import cloudpickle

pickle.Pickler = cloudpickle.Pickler
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import repeat
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import multiprocess as mp
import nvsmi
import psutil
import torch
from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils import partially_aggregate, set_parameters

# mp.set_start_method("spawn", force=True)


warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from multiprocess import shared_memory

from datasets import ShakespeareDataset
from models import ShakespeareLeafNet as Net


def test(parameters, device):
    """Validate the model on the test set."""
    net = Net()
    net = set_parameters(net, parameters, device)
    criterion = torch.nn.CrossEntropyLoss()
    testset = ShakespeareDataset(root="data", client_id=0, dataset="test")
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
        net,
        dataset_fn,
        config,
        device,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
    ):
        super(Worker, self).__init__()
        self.node_net = net
        self.dataset_fn = dataset_fn
        self.config = config
        self.device = device
        self.task_queue: mp.Queue = task_queue
        self.result_queue: mp.Queue = result_queue
        self.partial_agg_model = (None, 0, 0.0)
        self.current_round = 0

    def process_task(self, task):
        # REMOVE NEED FOR NEW WEIGHTS
        cid, net = task
        net.to(self.device)

        net.train()
        # trainset = self.dataset_fn(client_id=cid)
        train_dataset = ShakespeareDataset(
            root=Path("/datasets/FedScale/shakespeare/data/"),
            clint_id=cid,
            dataset="train",
        )
        trainloader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=self.config["learning_rate"],
            momentum=self.config["momentum"],
            weight_decay=self.config["weight_decay"],
        )
        for _ in range(self.config["epochs"]):
            num_samples = 0
            num_correct = 0
            for data in trainloader:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                num_samples += len(labels)
                optimizer.zero_grad()
                predicitons = net(inputs)
                num_correct += (
                    (torch.max(predicitons.data, 1)[1] == labels).sum().item()
                )
                criterion(predicitons, labels.to(self.device)).backward()
                optimizer.step()
        net.eval()
        these_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        new_results = (these_weights, num_samples, num_correct / num_samples)
        # self.partial_agg_model = partially_aggregate(
        #    self.partial_agg_model, new_results
        # )
        self.result_queue.put(new_results)

    def run(self):
        # Check if still work to do
        for task in iter(self.task_queue.get, None):
            self.process_task(task)


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(
        self,
        net,
        dataset_fn,
        exp_config,
    ) -> None:
        super().__init__()
        self.all_gpus = nvsmi.get_gpus()

        # One task_queue per GPU
        self.task_queues = {f"cuda:{gpu.id}": mp.Queue() for gpu in self.all_gpus}

        # One result_queue to rule them all
        self.result_queue = mp.Queue()

        # Inital parameters
        self.node_parameters = [
            val.cpu().numpy() for _, val in net.state_dict().items()
        ]
        self.workers = {
            "cuda:0": [
                Worker(
                    net,
                    dataset_fn,
                    exp_config,
                    f"cuda:{0}",
                    self.task_queues[f"cuda:{0}"],
                    self.result_queue,
                )
                for _ in range(10)
            ],
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

        # Save model
        net = Net()
        net = set_parameters(net, parameters, "cpu")
        net.eval()

        # remove need to send parameters same worker, same parameters
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            num_total_virtual_clients += len(list_ids_for_this_gpu)
            for cid in list_ids_for_this_gpu:
                self.task_queues[device].put((cid, net))

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
                time.sleep(1)
                [worker.join() for worker in list_of_workers]


# global initialization
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load train and evaluate functions
    # client_fit_fn = call(cfg.gen_client_fit_fn)
    exp_config = {
        "learning_rate": 0.1,
        "batch_size": 4,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "epochs": 1,
    }
    node_manager = NodeManager(
        net=Net(),
        dataset_fn=call(cfg.gen_dataset_fn),
        exp_config=exp_config,
    )
    # Start Flower client
    fl.client.start_numpy_client(
        # server_address="127.0.0.1:8080", client=NodeManager(client_fit_fn=client_fit_fn)
        server_address="127.0.0.1:8080",
        client=node_manager,
    )


if __name__ == "__main__":
    main()
