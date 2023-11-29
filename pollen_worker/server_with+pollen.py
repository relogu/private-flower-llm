"""Flower simulation using a pollen server.

Starts a Flower server which awaits connections from Pollen node managers. It supports
using wandb for logging and hydra for exeperiment configuration.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Union

import flwr as fl
import hydra
import transformers
from omegaconf import DictConfig, OmegaConf

import wandb
from pollen_worker.clients.virtual_llm_client import gen_client_fn
from pollen_worker.pollen_client_manager import PollenClientManager
from pollen_worker.pollen_server import PollenServer
from pollen_worker.rs_fedavg import FedAvgReproducibleSampling
from pollen_worker.utils import wandb_init
from pollen_worker.wandb_history import WandbHistory

transformers.logging.set_verbosity_error()


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Implement main function to launch a Pollen's Server."""
    # TODO: Get the list of cids
    cid_samples_dict: Dict[Union[str, int], int] = {
        str(k): 1 for k in range(cfg.fl.n_total_clients)
    }
    # TODO: Instantiate the strategy
    strategy = FedAvgReproducibleSampling(
        fraction_fit=sys.float_info.min,
        fraction_evaluate=sys.float_info.min,
        min_fit_clients=cfg.fl.n_clients_per_round,
        min_available_clients=cfg.fl.n_clients_per_round,
        min_evaluate_clients=cfg.fl.n_clients_per_round,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures=False,
        initial_parameters=None,
        evaluate_metrics_aggregation_fn=None,
        seed=cfg.seed,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Wrap with wandb context manager
    with wandb_init(
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),
        config=wandb_config,  # type: ignore
    ) as _:
        wandb_history = WandbHistory(use_wandb=cfg.use_wandb)
        # Start Flower server
        hist = fl.server.start_server(
            server_address=cfg.flwr_address,
            server=PollenServer(
                cids=cid_samples_dict,
                client_fn=gen_client_fn(),
                strategy=strategy,
                client_manager=PollenClientManager(),
                placement_policy=cfg.pollen.placement_policy,
                saving_path=Path(cfg.pollen.saving_path),
                history=wandb_history,
                num_nodes=cfg.pollen.num_nodes,
            ),
            config=fl.server.ServerConfig(num_rounds=cfg.fl.num_rounds),
        )
        with open(Path(cfg.pollen.saving_path) / "history.json", "w") as f:
            json.dump(hist.__dict__, f)


if __name__ == "__main__":
    main()
