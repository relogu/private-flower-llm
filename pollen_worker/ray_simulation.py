"""Run a Ray-based Flower simulation.

Serves as a baseline for the Pollen paper. It supports using wandb for logging and hydra
for exeperiment configuration.
"""
import json
import os
import warnings
from logging import INFO
from pathlib import Path
from typing import Iterable, cast

import flwr as fl
import hydra
import nvsmi
import torch
import transformers
from flwr.client import ClientLike
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

import wandb
from pollen_worker.pollen_utils import get_clients_population_dict
from pollen_worker.utils import RayContextManager, wandb_init, weighted_average
from pollen_worker.virtual_client import VirtualClient
from pollen_worker.wandb_history import WandbHistory
from pollen_worker.wandb_server import WandbServer

transformers.logging.set_verbosity_error()

warnings.filterwarnings("ignore", category=UserWarning)

os.environ["RAY_USAGE_STATS"] = "0"
os.environ["RAY_USAGE_STATS_ENABLED"] = "0"


def get_n_worker_gpu_type(name: str = "openimage") -> dict[str, int]:
    """Return the mapping between GPU resources and number of Ray clients."""
    # NOTE: These numbers are compatible with the last version of Pollen
    if name == "reddit":
        return {
            "NVIDIA A40": 14,
            "NVIDIA GeForce RTX 2080 Ti": 3,
        }
    if name == "shakespeare" or name == "shakespeare_memory":
        return {
            "NVIDIA A40": 36,
            "NVIDIA GeForce RTX 2080 Ti": 10,
        }
    if name == "google_speech":
        return {
            "NVIDIA A40": 22,
            "NVIDIA GeForce RTX 2080 Ti": 7,
        }
    if name == "openimage":
        return {
            "NVIDIA A40": 15,
            "NVIDIA GeForce RTX 2080 Ti": 4,
        }
    raise ValueError(f"Unknown dataset name: {name}")


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Implement main function to lauch a Ray-based simulation."""
    log(
        INFO,
        "Task is: %s with fake=%s with run unique id: %s",
        cfg.task.name,
        cfg.task.is_fake,
        cfg.run_uuid,
    )

    # Setting up the expected number of Ray workers
    pool_size = cfg.task.n_clients_per_round
    num_available_gpus = torch.cuda.device_count()
    gpu_name = [gpu.name for gpu in nvsmi.get_gpus()]
    if cfg.num_nodes > 1:
        # NOTE: We need to account for the minimum number of workers that the GPUs offer
        n_w_expected = min([v for _, v in get_n_worker_gpu_type(cfg.task.name).items()])
    else:
        n_w_expected = get_n_worker_gpu_type(cfg.task.name)[gpu_name[0]]
    n_workers = min(
        pool_size,
        num_available_gpus * n_w_expected,
    )
    log(INFO, "This Ray-based simulation will use %s workers", n_workers)

    client_resources = {
        "num_gpus": num_available_gpus / n_workers,
        # NOTE: Ray supports fractional CPU resources
        "num_cpus": len(os.sched_getaffinity(0)) / n_workers,
    }
    log(INFO, "Client resources are: %s", client_resources)

    # Get the list of cids
    cid_samples_dict = get_clients_population_dict(
        name=cfg.task.name,
        batch_size=cfg.task.batch_size,
        seed=cfg.seed,
    )
    n_total_clients = len(cid_samples_dict)
    n_clients_per_round = cfg.task.n_clients_per_round

    def get_client_fn(
        cid: str,
    ) -> ClientLike:
        return VirtualClient(
            name=cfg.task.name,
            cid=int(cid),
        )

    on_fit_config_fn = call(cfg.gen_on_fit_config_fn)

    # Configure the strategy
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()  # type: ignore
    saving_path = Path(hydra_cfg["runtime"]["output_dir"])  # type: ignore[index]
    strategy = instantiate(
        cfg.task.strategy,
        saving_path=saving_path,
        min_fit_clients=2,
        fraction_evaluate=0.0,
        fraction_fit=n_clients_per_round / n_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=ndarrays_to_parameters(
            get_client_fn(cid=0).get_parameters(config={}, net=None)  # type: ignore
        ),
        fit_metrics_aggregation_fn=weighted_average,
        freq=cfg.save_freq,
        seed=cfg.seed,
    )
    log(INFO, f"Fraction fit is: {strategy.fraction_fit}")

    # (Optional) Specify Ray configuration
    log(INFO, f"This simulation has affinity: {os.sched_getaffinity(0)}")
    ray_init_args = {
        "address": cfg.ray_address,
        "_redis_password": cfg.ray_redis_password,
        "_node_ip_address": cfg.ray_node_ip_address,
    }

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Start simulation
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as _:
        wandb_history = WandbHistory(use_wandb=cfg.use_wandb)
        server = WandbServer(
            client_manager=SimpleClientManager(),
            history=wandb_history,
            strategy=strategy,
        )
        with RayContextManager() as _:
            hist = fl.simulation.start_simulation(
                client_fn=get_client_fn,
                clients_ids=list(cast(Iterable, cid_samples_dict.keys())),
                client_resources=client_resources,
                server=server,
                config=fl.server.ServerConfig(num_rounds=cfg.task.num_rounds),
                ray_init_args=ray_init_args,
            )
            with open(saving_path / "history.json", "w") as f:
                json.dump(hist.__dict__, f)


if __name__ == "__main__":
    main()
