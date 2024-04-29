"""Flower simulation using a pollen server.

Starts a Flower server which awaits connections from Pollen node managers. It supports
using wandb for logging and hydra for experiment configuration.
"""

import copy
from logging import INFO
import pickle
import sys
from pathlib import Path
from typing import cast

import flwr as fl
import hydra
import transformers
from flwr.common import ndarrays_to_parameters, log, NDArrays
from omegaconf import DictConfig, OmegaConf

import wandb
from pollen_worker.clients.empty_virtual_client import gen_client_fn
from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from pollen_worker.pollen_client_manager import PollenClientManager
from pollen_worker.pollen_server import PollenServer
from pollen_worker.strategy.rs_nesterov import FedNesterov
from pollen_worker.utils import (
    POLLEN_LLM_MAX_MESSAGE_LENGTH,
    wandb_init,
    weighted_average,
)
from pollen_worker.wandb_history import WandbHistory

transformers.logging.set_verbosity_error()


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Implement main function to launch a Pollen's Server."""
    # Get a fake list of cids
    cid_samples_dict: dict[str | int, int] = dict.fromkeys(
        range(cfg.fl.n_total_clients), 1
    )
    num_train_client_streams = len(cfg.dataset.train.streams)
    num_eval_client_streams = len(cfg.dataset.val.streams)

    if num_train_client_streams < cfg.fl.n_total_clients:
        raise ValueError(
            """When statically specifying client streams for training, the number of
            entries must be greater or equal to the total number of clients."""
        )

    if num_eval_client_streams < 1:
        raise ValueError(
            """When statically specifying client streams for eval, the number of entries
            must be equal to 1."""
        )

    # Get initial model parameters
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    if cfg.pretrained_model_path:
        log(
            INFO,
            "FL server is loading pretrained model from %s",
            cfg.pretrained_model_path,
        )
        with open(Path(cfg.pretrained_model_path), "rb") as f:
            initial_parameters = ndarrays_to_parameters(pickle.load(f))
    else:
        log(
            INFO,
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
                INFO,
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
        min_evaluate_clients=1,
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
        rescale_global_model=cfg.fl.rescale_global_model,
        rescale_momentum_vector=cfg.fl.rescale_momentum_vector,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Wrap with wandb context manager
    with wandb_init(  # type: ignore[union-attr]
        cfg.use_wandb,
        **cfg.wandb.setup,
        settings=wandb.Settings(start_method="thread"),  # type: ignore[arg-type]
        config=wandb_config,  # type: ignore[arg-type]
    ) as _:
        wandb_history = WandbHistory(use_wandb=cfg.use_wandb)
        restore_run_uuid_round_and_step: tuple[str, int, int] | None = None
        # Restore from a previous run
        if (
            (cfg.use_s3_comm or cfg.pollen.checkpoint)
            and cfg.pollen.restore_run_uuid is not None
            and cfg.pollen.resume_round >= 0
        ):
            restore_run_uuid_round_and_step = (
                cfg.pollen.restore_run_uuid,
                int(cfg.pollen.resume_round),
                int(
                    int(cfg.llm_config.local_steps.replace("ba", ""))
                    * cfg.pollen.resume_round
                ),
            )

        # Start Flower server
        fl.server.start_server(
            server_address=cfg.pollen.server_address,
            server=PollenServer(
                run_uuid=cfg.run_uuid,
                cids=cid_samples_dict,
                client_fn=gen_client_fn(),
                strategy=strategy,
                client_manager=PollenClientManager(),
                placement_policy=cfg.pollen.placement_policy,
                saving_path=Path(cfg.pollen.saving_path),
                history=wandb_history,
                num_nodes=cfg.pollen.n_nodes,
                use_s3_comm=cfg.use_s3_comm,
                s3_comm_config=cfg.s3_comm_config,
                checkpoint=cfg.pollen.checkpoint,
                resume_round=cfg.pollen.resume_round,
                restore_run_uuid_round_and_step=restore_run_uuid_round_and_step,
            ),
            config=fl.server.ServerConfig(num_rounds=cfg.fl.n_rounds),
            grpc_max_message_length=POLLEN_LLM_MAX_MESSAGE_LENGTH,
        )


if __name__ == "__main__":
    main()
