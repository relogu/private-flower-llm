import warnings
from itertools import repeat
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import nvsmi
import psutil
import torch
import torch.multiprocessing as mp
import multiprocess as mp
from omegaconf import DictConfig

mp.set_start_method("spawn", force=True)

from flwr.common import Config, NDArrays, Scalar
from torch.utils.data import DataLoader
from hydra.utils import call
from utils import set_parameters
from utils import partially_aggregate

warnings.filterwarnings("ignore", category=UserWarning)

from models import ShakespeareLeafNet as Net
from datasets import ShakespeareDataset


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
        self.pool: Optional[
            Dict[str, mp.Pool]
        ] = {  # Maybe cudatype_cudaid example: a40_0
            f"cuda:{gpu.id}": mp.Pool(10) for gpu in self.all_gpus
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
        with mp.Manager() as manager:
            results_queue = manager.Queue()
            for device, p in self.pool.items():
                if device not in config:
                    continue
                list_ids_for_this_gpu = config[device].split(",")
                total_num_virtual_clients += len(list_ids_for_this_gpu)
                tasks = list(
                    zip(
                        list_ids_for_this_gpu,
                        repeat(parameters),
                        repeat(device),
                        repeat(results_queue),
                    )
                )
                print(f"len(tasks): {len(tasks)}")
                p.starmap_async(self.client_fit_fn, tasks)

            # Partial aggregation
            part_agg_weights = (None, 0, 0.0)
            for _ in range(total_num_virtual_clients):  # Length of results will vary!!!
                part_agg_weights = partially_aggregate(
                    part_agg_weights, results_queue.get()
                )

        return (
            part_agg_weights[0],
            part_agg_weights[1],
            {"accuracy": part_agg_weights[2]},
        )

    def evaluate(self, parameters, config):
        return 0.0, 1, {}

    def __del__(self):
        if self.pool is not None:
            for p in self.pool.values():
                p.close()


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
