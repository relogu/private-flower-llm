from collections import OrderedDict
import pickle
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import cloudpickle
import torch
from flwr.client import NumPyClient

pickle.Pickler = cloudpickle.Pickler
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import hydra
import multiprocess as mp
import nvsmi
import psutil
from flwr.common import Config, NDArrays, Scalar
from hydra.utils import call
from nvsmi import GPU
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from clients import train
from utils import partially_aggregate, set_parameters

# mp.set_start_method("spawn", force=True)

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from multiprocess import shared_memory

from datasets.shakespeare import SHAKESPEARE_LOADED as ShakespeareDataset
from models.shakespeare_leaf_model import ShakespeareLeafNet as Net


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


class Worker(mp.Process):
    def __init__(
        self,
        client_fn,
        device,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        dataset_root: Path = Path("/datasets/FedScale/leaf_shakespeare"),
        net: Net = Net(),
    ):
        super(Worker, self).__init__()
        self.device = device
        self.client_fn: Callable[[int], NumPyClient] = client_fn
        self.task_queue: mp.Queue = task_queue
        self.result_queue: mp.Queue = result_queue
        self.partial_agg_model: Tuple[Optional[NDArrays], int, float] = (None, 0, 0.0)
        self.current_round: int = 0
        self.shm_config = SharedMemory(name="pollen_config_sm")
        self.shm_params = SharedMemory(name="pollen_param_sm")
        self.dataset_root = dataset_root
        self.net = net

    def process_task(self, client_id: int):
        # Get config, and new parameters via Ray and shared memory
        # Add namespace to avoid collision
        config = pickle.loads(self.shm_config.buf)  # Loads a dict

        # Load model from shared memory
        # net = pickle.loads(self.shm_params.buf)  # Loads net
        # parameters = pickle.loads(self.shm_params.buf)  # Loads net
        params_dict = zip(
            self.net.state_dict().keys(), pickle.loads(self.shm_params.buf)
        )
        state_dict = OrderedDict(
            {k: torch.tensor(v, device=self.device) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)
        # print(
        #    f"Worker on round {config['server_round']} processing task {client_id} got ZeroCopy"
        # )

        # Set parameters
        # temp_client = self.client_fn(config["client_id"])
        # trained_weights, num_samples, metrics = temp_client.fit(
        #    self.local_parameters, config
        # )
        trainset = ShakespeareDataset(self.dataset_root, client_id=client_id)
        trainloader = DataLoader(
            trainset, batch_size=config["batch_size"], shuffle=True
        )
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        # print(f"Training client {client_id}")
        trained_weights, num_samples, metrics = train(
            self.net, trainloader, config["local_epochs"], optimizer, self.device
        )
        # print(f"Trained client {client_id}")

        new_results = (trained_weights, num_samples, metrics["accuracy"])
        self.result_queue.put(new_results)
        # print("current memory queue size: ", self.result_queue.qsize())

    def run(self):
        # Check if still work to do
        for task in iter(self.task_queue.get, None):
            self.process_task(task)


# Define Flower client
class NodeManager(fl.client.NumPyClient):
    def __init__(self, client_fn) -> None:
        super().__init__()
        self.all_gpus: List[GPU] = nvsmi.get_gpus()

        # One task_queue per GPU make this ctypes array
        self.task_queues = {f"cuda:{gpu.id}": mp.Queue() for gpu in self.all_gpus}

        # One model per worker
        self.shm_config = SharedMemory(name="pollen_config_sm", create=True, size=1000)
        self.shm_params = SharedMemory(
            name="pollen_param_sm", create=True, size=100_000_000
        )

        # Results from worker
        self.result_queue = mp.Queue()

        self.workers = {
            "cuda:0": [
                Worker(
                    client_fn,
                    "cuda:0",
                    self.task_queues[f"cuda:{0}"],
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
        # Send config to shared memory
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        self.shm_config.buf[: len(config_bytes)] = config_bytes

        # Send parameters to shared memory
        # temp_net = Net()
        # set_parameters(temp_net, parameters)
        param_ray_obj_ref_bytes = pickle.dumps(
            parameters, protocol=pickle.HIGHEST_PROTOCOL
        )
        self.shm_params.buf[: len(param_ray_obj_ref_bytes)] = param_ray_obj_ref_bytes

        num_total_virtual_clients = 0
        for device in self.workers.keys():
            list_ids_for_this_gpu = config[device].split(",")
            # print(f"list of ids: {config[device]}")
            num_total_virtual_clients += len(list_ids_for_this_gpu)

            for cid in list_ids_for_this_gpu:
                # print(f"putting cid in queue {cid}")
                self.task_queues[device].put(cid)

            # Start workers in this GPU move this outside
            if config["server_round"] == 1:
                self.start_workers(config)

        # Aggregate results
        node_part_agg = (None, 0, 0.0)
        # print(f"Total num virtual clients: {num_total_virtual_clients}")
        for x in range(num_total_virtual_clients):
            new_results = self.result_queue.get()
            # nprint(f"Got one result{x}")
            node_part_agg = partially_aggregate(node_part_agg, new_results)
        # print("Done with aggregation")
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
        # Free shared memory
        self.shm_config.close()
        self.shm_params.close()
        self.shm_config.unlink()
        self.shm_params.unlink()


# global initialization
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # Define Node Manager
    # ray.init(address=cfg.ray_address, logging_level=logging.ERROR)
    node_manager = NodeManager(client_fn=call(cfg.gen_client_fn))

    # Start Flower client
    fl.client.start_numpy_client(
        server_address=cfg.flwr_address,
        client=node_manager,
    )


if __name__ == "__main__":
    main()
