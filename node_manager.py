from collections import OrderedDict
import pickle
import cloudpickle

pickle.Pickler = cloudpickle.Pickler
import warnings
from itertools import repeat
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import nvsmi
import psutil
import torch

# import multiprocess as mp
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process

# mp.set_start_method("spawn", force=True)

from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from torch.utils.data import DataLoader

from utils import partially_aggregate, set_parameters

warnings.filterwarnings("ignore", category=UserWarning)

from datasets import ShakespeareDataset
from models import ShakespeareLeafNet as Net


def test(parameters, device):
    """Validate the model on the test set."""
    net = Net()
    net = set_parameters(net, parameters, device)
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


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self, client_fit_fn) -> None:
        super().__init__()
        self.all_gpus = nvsmi.get_gpus()
        self.executors = {
            f"cuda:{gpu.id}": ProcessPoolExecutor(max_workers=10)
            for gpu in self.all_gpus
        }
        self.client_fit_fn = client_fit_fn

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
        # Can check if config contains updates pool
        return dict(cpu_prop, **gpus_prop)

    def get_parameters(sef, config):
        net = Net()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        total_num_virtual_clients = 0
        futures = []
        for device, executor in self.executors.items():
            if device not in config:
                continue
            list_ids_for_this_gpu = config[device].split(",")
            total_num_virtual_clients += len(list_ids_for_this_gpu)
            tasks = list(
                zip(
                    list_ids_for_this_gpu,
                    repeat(parameters),
                    repeat(device),
                    # repeat(self.results_queue),
                )
            )
            futures = futures + [
                executor.submit(self.client_fit_fn, *task) for task in tasks
            ]

        # Partial aggregation
        part_agg = (None, 0, 0.0)
        for future in as_completed(futures):
            part_agg = partially_aggregate(part_agg, future.result())

        return (
            part_agg[0],
            part_agg[1],
            {"accuracy": part_agg[2]},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.executors is not None:
            for ex in self.executors.values():
                ex.shutdown()


# global initialization
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Load train and evaluate functions
    client_fit_fn = call(cfg.gen_client_fit_fn)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", client=NodeManager(client_fit_fn=client_fit_fn)
    )


if __name__ == "__main__":
    main()
