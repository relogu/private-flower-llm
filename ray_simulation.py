import argparse
import json
import os
import warnings
from logging import INFO
from pathlib import Path

import flwr as fl
import hydra
import nvsmi
import torch
import transformers
from flwr.client import ClientLike
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import log
from hydra.utils import call
from omegaconf import DictConfig

from pollen_utils import get_clients_population_dict
from rs_fedavg import FedAvgReproducibleSampling
from utils import weighted_average
from virtual_client import VirtualClient

transformers.logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--name", type=str, default="openimage")


def get_n_worker_gpu_type(name: str = "openimage"):
    # NOTE: These numbers are compatible with the last version of Pollen
    if name == "reddit":
        return {
            "NVIDIA A40": 14,
            "NVIDIA GeForce RTX 2080 Ti": 3,
        }
    elif name == "shakespeare" or name == "shakespeare_memory":
        return {
            "NVIDIA A40": 36,
            "NVIDIA GeForce RTX 2080 Ti": 11,
        }
    elif name == "google_speech":
        return {
            "NVIDIA A40": 22,
            "NVIDIA GeForce RTX 2080 Ti": 7,
        }
    elif name == "openimage":
        return {
            "NVIDIA A40": 15,
            "NVIDIA GeForce RTX 2080 Ti": 4,
        }


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    log(
        INFO,
        f"Task is: {cfg.task.name} with fake={cfg.task.is_fake} with run unique id: {cfg.run_uuid}",
    )

    # number of dataset partions (= number of total clients)
    pool_size = 10 if "shakespeare" in cfg.task.name else 100
    num_available_gpus = torch.cuda.device_count()
    gpu_name = [gpu.name for gpu in nvsmi.get_gpus()]
    n_workers = min(
        pool_size,
        num_available_gpus * get_n_worker_gpu_type(cfg.task.name)[gpu_name[0]],
    )
    print(f"n_workers: {n_workers}")

    client_resources = {
        "num_gpus": num_available_gpus / n_workers,
    }

    # Get the list of cids
    cid_samples_dict = get_clients_population_dict(
        name=cfg.task.name,
        batch_size=cfg.task.batch_size,
    )
    n_total_clients = len(cid_samples_dict)
    n_clients_per_round = cfg.task.n_clients_per_round

    def get_client_fn(
        cid: int,
    ) -> ClientLike:
        return VirtualClient(
            name=cfg.task.name,
            cid=cid,
        )

    on_fit_config_fn = call(cfg.gen_on_fit_config_fn)
    # configure the strategy
    strategy = FedAvgReproducibleSampling(
        total_clients=n_total_clients,
        min_fit_clients=2,
        fraction_evaluate=0.0,
        fraction_fit=n_clients_per_round / n_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=ndarrays_to_parameters(
            get_client_fn(cid=0).get_parameters(config={}, net=None)
        ),
        fit_metrics_aggregation_fn=weighted_average,
    )
    log(INFO, f"Fraction fit is: {strategy.fraction_fit}")

    # (optional) specify Ray config
    print(f"Using {int(n_workers)} CPUs overall")
    print(f"Having affinity {os.sched_getaffinity(0)}")
    ray_init_args = {
        # "address": cfg.ray_address,
        "include_dashboard": False,
        "num_cpus": len(os.sched_getaffinity(0)),
        "logging_level": INFO,
        "log_to_driver": True,
        "_system_config": {
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {"directory_path": "/hdd1/ray/ray_spilled_objects/"},
                },
            )
        },
    }

    # start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn,
        clients_ids=list(cid_samples_dict.keys()),
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=cfg.task.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )


if __name__ == "__main__":
    main()
