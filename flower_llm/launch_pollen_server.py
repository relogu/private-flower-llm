"""Flower simulation using a pollen server.

Starts a Flower server which awaits connections from Pollen node managers. It supports
using wandb for logging and hydra for experiment configuration.
"""

import copy
from logging import DEBUG
import os
import sys
from pathlib import Path
from typing import cast
import warnings

import flwr as fl
import transformers
from flwr.common import ndarrays_to_parameters, log, NDArrays
from flwr.common.logger import update_console_handler
from omegaconf import OmegaConf

from flower_llm.conf.base_schema import BaseConfig
from flower_llm.clients.empty_virtual_client import gen_client_fn
from flower_llm.clients.llm_client_functions import get_raw_model_parameters
from flower_llm.pollen_client_manager import PollenClientManager
from flower_llm.pollen_server import PollenServer
from flower_llm.strategy.rs_nesterov import FedNesterov
from flower_llm.strategy.aggregation import weighted_average
from flower_llm.utils import (
    load_model_parameters_from_file,
)
from flower_llm.wandb_history import WandbHistory
from flower_llm.wandb_server_app import WandbServerApp

transformers.logging.set_verbosity_error()


# Define strategy
def main() -> WandbServerApp:
    """Implement main function to launch a Pollen's Server."""
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("POLLEN_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
    # Load the configuration from the config file
    cfg = cast(BaseConfig, OmegaConf.load(save_path + "/config.yaml"))

    # Get a fake list of cids
    cid_samples_dict: dict[str | int, int] = dict.fromkeys(
        range(cfg.fl.n_total_clients), 1
    )
    num_train_client_streams = len(cfg.dataset.train.streams)
    num_eval_client_streams = len(cfg.dataset.val.streams)

    if num_train_client_streams < cfg.fl.n_total_clients:
        raise ValueError(
            """When statically specifying client streams for training, the number of
            entries must be greater or equal to the total number of clients.
            Number of train streams: %d, total number of clients: %d""",
            num_train_client_streams,
            cfg.fl.n_total_clients,
        )

    if num_eval_client_streams < 1:
        raise ValueError(
            """When statically specifying client streams for eval, the number of entries
            must be equal or greater than 1. Number of eval streams: %d""",
            num_eval_client_streams,
        )

    # Get initial model parameters
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    if cfg.pretrained_model_path:
        log(
            DEBUG,
            "FL server is loading pretrained model from %s",
            cfg.pretrained_model_path,
        )
        initial_parameters = ndarrays_to_parameters(
            load_model_parameters_from_file(Path(cfg.pretrained_model_path))
        )
    else:
        log(
            DEBUG,
            "FL server initializes model with random parameters.",
        )
        initial_parameters_ndarrays: NDArrays
        names: list[str]
        (initial_parameters_ndarrays, names) = cast(
            tuple[NDArrays, list[str]],
            get_raw_model_parameters(copy.deepcopy(_llm_config), True, True),
        )
        for i, (param, name) in enumerate(
            zip(initial_parameters_ndarrays, names, strict=True)
        ):
            log(
                DEBUG,
                "Initial parameter, component %s, name %s, shape %s",
                i,
                name,
                param.shape,
            )
        initial_parameters = ndarrays_to_parameters(initial_parameters_ndarrays)
    # Instantiate the strategy
    strategy = FedNesterov(
        server_learning_rate=cfg.fl.server_learning_rate,
        server_momentum=cfg.fl.server_momentum,
        fraction_fit=sys.float_info.min,
        fraction_evaluate=sys.float_info.min,
        min_fit_clients=cfg.fl.n_clients_per_round,
        min_available_clients=cfg.fl.n_clients_per_round,
        min_evaluate_clients=1,  # We are using this for "centralized" evaluation
        evaluate_fn=None,
        on_fit_config_fn=lambda x: {
            "server_round": x,
            "batch_size": cfg.llm_config.global_train_batch_size,
            "n_local_steps": cfg.fl.n_local_steps,
            "n_local_epochs": cfg.fl.n_local_epochs,
            "collaborative": cfg.pollen.fit_collaborative,
            "reset_optimizer": cfg.fl.reset_optimizer,
        },
        on_evaluate_config_fn=lambda x: {
            "server_round": x,
            "batch_size": cfg.llm_config.device_eval_batch_size,
            "collaborative": cfg.pollen.eval_collaborative,
        },
        accept_failures=False,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        seed=cfg.seed,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    assert (
        type(wandb_config) is dict
    ), f"wandb_config must be a dictionary, not a {type(wandb_config)}."
    wandb_history = WandbHistory(use_wandb=cfg.use_wandb)
    restore_run_uuid_and_steps_per_round: tuple[str, int] | None = None
    # Restore from a previous run
    if (
        cfg.use_s3_comm or cfg.pollen.checkpoint
    ) and cfg.pollen.restore_run_uuid is not None:
        restore_run_uuid_and_steps_per_round = (
            cfg.pollen.restore_run_uuid,
            int(cfg.llm_config.local_steps.replace("ba", "")),
        )

    # Start Flower Next ServerApp
    return WandbServerApp(
        config=fl.server.ServerConfig(num_rounds=cfg.fl.n_rounds),
        server=PollenServer(
            run_uuid=cfg.run_uuid,
            cids=cid_samples_dict,
            client_fn=gen_client_fn(),
            strategy=strategy,
            client_manager=PollenClientManager(),
            placement_policy=cfg.pollen.placement_policy,
            history=wandb_history,
            num_nodes=cfg.pollen.n_nodes,
            use_s3_comm=cfg.use_s3_comm,
            s3_comm_config=cfg.s3_comm_config,
            checkpoint=cfg.pollen.checkpoint,
            resume_round=cfg.pollen.resume_round,
            restore_run_uuid_and_steps_per_round=restore_run_uuid_and_steps_per_round,
        ),
        wandb_kwargs=cfg.wandb.setup if cfg.use_wandb else None,
        wandb_config=wandb_config,
    )


# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)
# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# TODO: These don't work -- not sure why
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)
server_app = main()

if __name__ == "__main__":
    main()
