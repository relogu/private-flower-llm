"""Centralised training script for LLMFoundry models.

Slightly adapted from the original https://github.com/mosaicml/llm-foundry/blob/25599294c942cfed2c6f8329e14791e4a2f91539/scripts/train/train.py
Copyright 2022 MosaicML LLM Foundry authors
SPDX-License-Identifier: Apache-2.0
"""

import gc
from logging import INFO
import os
from typing import cast

import numpy as np
import torch
from composer import Trainer
from flwr.common import log, NDArray
from omegaconf import OmegaConf
from llmfoundry.callbacks import EvalGauntlet

from flower_llm.conf.base_schema import BaseConfig
from flower_llm.clients.llm_client_functions import (
    _get_trainer_object,
    get_parameters_from_state,
)
from flower_llm.clients.llm_config_functions import validate_config
from flower_llm.server.s3_utils import load_pretrained_model_from_path
from flower_llm.utils import (
    get_wte_parameters_from_trainer,
    set_wte_parameters_to_trainer,
)


def main() -> Trainer:
    """Implement the main training loop for LLMFoundry models."""
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("POLLEN_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
    # Load the configuration from the config file
    _cfg = cast(BaseConfig, OmegaConf.load(save_path + "/config.yaml"))
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(_cfg)
    OmegaConf.set_struct(_cfg, False)
    cfg = _cfg.llm_config
    # Check for incompatibilities between the model and data loaders
    validate_config(cfg)
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    # NOTE: The cid passed her is use to appoint the position for the stream used for
    # creating the streaming dataset object
    log(
        INFO,
        "Creating trainer object using stream_id: %s...",
        _cfg.centralized.stream_id,
    )
    trainer, eval_first, *_ = _get_trainer_object(
        _cfg=cfg,
        cid=_cfg.centralized.stream_id,
        log_name="_centralised",
        use_unigram_metrics=cfg.fl.use_unigram_metrics,
        s3_comm_config=_cfg.s3_comm_config,
    )
    torch.cuda.empty_cache()
    gc.collect()

    wte_parameters: NDArray | None = None
    if _cfg.wte_parameters_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=_cfg.wte_parameters_path,
            run_uuid=_cfg.run_uuid,
            s3_comm_config=_cfg.s3_comm_config,
        )
        wte_parameters = get_wte_parameters_from_trainer(trainer)

    if _cfg.pretrained_model_path:
        load_pretrained_model_from_path(
            trainer=trainer,
            pretrained_model_path=_cfg.pretrained_model_path,
            run_uuid=_cfg.run_uuid,
            s3_comm_config=_cfg.s3_comm_config,
        )

    if wte_parameters is not None:
        set_wte_parameters_to_trainer(trainer, wte_parameters)

    # Eval first if requested
    if eval_first:
        trainer.eval()
        eval_gauntlet_callback: EvalGauntlet | None = None
        for callback in trainer.state.callbacks:
            if isinstance(callback, EvalGauntlet):
                eval_gauntlet_callback = callback
        if eval_gauntlet_callback is not None:
            assert isinstance(eval_gauntlet_callback, EvalGauntlet)
            composite_scores = eval_gauntlet_callback.eval_after_all(
                trainer.state,
                trainer.logger,
            )
            log(
                INFO,
                "Evaluated model with the Gauntlet before training: %s",
                composite_scores,
            )

    # Dump model parameters to file
    if _cfg.centralized.store_init_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        with open(f"{_cfg.run_uuid}-{n_steps}-checkpoint.npz", "wb") as f:
            np.savez(f, *model_parameters)

    if not _cfg.centralized.eval_only:
        log(INFO, "Starting training...")
        trainer.fit()

    # Dump model parameters to file
    if _cfg.centralized.store_final_model:
        # Get model parameters from trainer object
        model_parameters = get_parameters_from_state({}, trainer)
        # Get number of steps executed
        n_steps = trainer.state.timestamp.batch.value
        # Dump the compressed model parameters to file
        with open(f"{_cfg.run_uuid}-{n_steps}-checkpoint.npz", "wb") as f:
            np.savez(f, *model_parameters)

    log(INFO, "Done.")
    return trainer


if __name__ == "__main__":
    main()
