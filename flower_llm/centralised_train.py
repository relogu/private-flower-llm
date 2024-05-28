"""Centralised training script for LLMFoundry models.

Slightly adapted from the original https://github.com/mosaicml/llm-foundry/blob/25599294c942cfed2c6f8329e14791e4a2f91539/scripts/train/train.py
Copyright 2022 MosaicML LLM Foundry authors
SPDX-License-Identifier: Apache-2.0
"""

import gc
from logging import INFO

import hydra
import numpy as np
import torch
from composer import Trainer
from flwr.common.logger import log
from llmfoundry.utils.config_utils import (
    log_config,
)
from omegaconf import DictConfig, OmegaConf

from flower_llm.clients.llm_client_functions import (
    _get_trainer_object,
    get_parameters_from_state,
)
from flower_llm.clients.llm_config_functions import validate_config


@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(_cfg: DictConfig) -> Trainer:
    """Implement the main training loop for LLMFoundry models."""
    log(
        INFO,
        "The centralized training script received the following config:\n%s",
        OmegaConf.to_yaml(_cfg, resolve=True),
    )
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(_cfg)
    OmegaConf.set_struct(_cfg, False)
    cfg = _cfg.llm_config
    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    trainer, eval_first, logged_cfg = _get_trainer_object(_cfg=cfg, cid=0)

    log(INFO, "Logging config")
    log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    # Eval first if requested
    if eval_first and trainer.state.timestamp.batch.value == 0:
        trainer.eval()

    # Dump model parameters to file
    if _cfg.store_init_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        with open(f"{_cfg.run_uuid}-{n_steps}-checkpoint.npz", "wb") as f:
            np.savez_compressed(f, *model_parameters)

    log(INFO, "Starting training...")
    trainer.fit()

    # Dump model parameters to file
    if _cfg.store_final_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        with open(f"{_cfg.run_uuid}-{n_steps}-checkpoint.npz", "wb") as f:
            np.savez_compressed(f, *model_parameters)

    log(INFO, "Done.")
    return trainer


if __name__ == "__main__":
    main()
