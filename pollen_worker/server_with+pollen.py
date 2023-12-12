"""Flower simulation using a pollen server.

Starts a Flower server which awaits connections from Pollen node managers. It supports
using wandb for logging and hydra for exeperiment configuration.
"""
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Union

import flwr as fl
import hydra
import transformers
from flwr.common import ndarrays_to_parameters
from omegaconf import DictConfig, OmegaConf

import wandb
from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from pollen_worker.clients.virtual_llm_client import gen_client_fn
from pollen_worker.pollen_client_manager import PollenClientManager
from pollen_worker.pollen_server import PollenServer
from pollen_worker.strategy.rs_nesterov import FedNesterov
from pollen_worker.utils import wandb_init
from pollen_worker.wandb_history import WandbHistory

transformers.logging.set_verbosity_error()


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Implement main function to launch a Pollen's Server."""
    # TODO: Get the list of cids
    cid_samples_dict: Dict[Union[str, int], int] = {
        k: 1 for k in range(cfg.fl.n_total_clients)
    }
    # Get initial model parameters
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    initial_parameters = ndarrays_to_parameters(
        get_raw_model_parameters(copy.deepcopy(_llm_config))
    )
    # TODO: Instantiate the strategy
    # No evaluation here, all lazy
    strategy = FedNesterov(
        fraction_fit=sys.float_info.min,
        fraction_evaluate=0,
        min_fit_clients=cfg.fl.n_clients_per_round,
        min_available_clients=cfg.fl.n_clients_per_round,
        min_evaluate_clients=0,
        evaluate_fn=None,
        on_fit_config_fn=lambda x: {"server_round": x, "batch_size": 32},
        on_evaluate_config_fn=lambda x: {"server_round": x, "batch_size": 32},
        accept_failures=False,
        initial_parameters=initial_parameters,
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
            server_address=cfg.pollen.server_address,
            server=PollenServer(
                cids=cid_samples_dict,
                client_fn=gen_client_fn(),
                strategy=strategy,
                client_manager=PollenClientManager(),
                placement_policy=cfg.pollen.placement_policy,
                saving_path=Path(cfg.pollen.saving_path),
                history=wandb_history,
                num_nodes=cfg.pollen.n_nodes,
            ),
            config=fl.server.ServerConfig(num_rounds=cfg.fl.n_rounds),
            grpc_max_message_length=int(1_000_000_000),
        )
        Path(cfg.pollen.saving_path).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg.pollen.saving_path) / "history.json", "x") as f:
            json.dump(hist.__dict__, f)


if __name__ == "__main__":
    main()
