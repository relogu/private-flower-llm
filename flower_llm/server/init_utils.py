"""Utility functions for initialization task on main server loop in flwr next."""

import ast
from collections.abc import Callable, Generator
from copy import deepcopy
import copy
from dataclasses import asdict
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any, cast
import time

from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
    get_raw_model_parameters,
)
from flower_llm.pollen_server import TooManyFailuresError
from flower_llm.server.s3_utils import download_server_checkpoint, replace_clients_updates_with_remote, upload_server_checkpoint
from flower_llm.server.server_util import interpret_resume_round
from flower_llm.utils import (
    ClientState,
    download_file_from_s3,
    dump_model_parameters_to_file,
    load_model_parameters_from_file,
    obtain_sorted_runs,
    upload_file_to_s3,
)
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Parameters,
    log,
    Message,
    MessageType,
    DEFAULT_TTL,
    ConfigsRecord,
    RecordSet,
    Scalar,
    FitIns,
    FitRes,
    EvaluateRes,
    Status,
    Code,
)

from flwr.common.recordset_compat import parameters_to_parametersrecord
from flwr.common.recordset_compat import (
    fitins_to_recordset,
    recordset_to_fitres,
    recordset_to_evaluateres,
)
from flwr.server import Driver, History
from flwr.server.strategy import FedAvg
from omegaconf import OmegaConf
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path


from flower_llm.conf.base_schema import BaseConfig




def get_initial_parameters(cfg: BaseConfig) -> Parameters:
    """Retrieve the initial parameters for the federated learning server model.

    This function returns the initial parameters of the model using the configuration.
    If a pretrained model path is specified in the configuration (`cfg`), it loads its
    parameters from the specified file. Otherwise, it returns random parameters
    based on the provided large language model (LLM) configuration. Also, it logs
    the shapes and names of the initial parameters for debugging purposes.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing the pretrained model path and LLM config.

    Returns
    -------
    'Parameters'
        The initial parameters of the model, either loaded from a pretrained model or
        initialized randomly based on the LLM configuration.
    """
    if cfg.pretrained_model_path:
        log(
            DEBUG,
            "FL server is loading pretrained model from %s",
            cfg.pretrained_model_path,
        )
        return ndarrays_to_parameters(
            load_model_parameters_from_file(Path(cfg.pretrained_model_path))
        )
    else:
        log(
            DEBUG,
            "FL server initializes model with random parameters.",
        )
        _llm_config = cfg.llm_config
        OmegaConf.resolve(_llm_config)
        OmegaConf.set_struct(_llm_config, False)
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
        return ndarrays_to_parameters(initial_parameters_ndarrays)


def initialize_round(
    cfg: BaseConfig, remote_up_down: RemoteUploaderDownloader | None
) -> tuple[
    Parameters, History, int, float, int, dict[str | int, ClientState], NDArrays | None
]:
    """Initialize the state for a new round of federated learning.

    This function sets up the initial state for a new round of federated learning,
    including initializing bookkeeping variables, client states, global parameters, and
    optionally saving the initial checkpoint to an S3 Object Store if configured. It
    prepares the server for the federated learning process by setting the starting
    round, time offset, cumulative server steps, and initializing the momentum vector
    for optimization algorithms.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing settings for federated learning and system
        behavior.
    remote_up_down : RemoteUploaderDownloader | None
        An optional uploader/downloader object for interacting with remote storage,
        required if checkpointing or S3 communication is enabled.

    Returns
    -------
    tuple
        A tuple containing the initialized global parameters, history object, starting
        round number, time offset, cumulative server steps, client state dictionary, and
        the initial momentum vector (or None if not applicable).

    Raises
    ------
    AssertionError
        If checkpointing or S3 communication is enabled but no RemoteUploaderDownloader
        object is provided.
    """
    # Initialize the bookkeeping variables
    start_round: int = 0
    time_offset: float = 0.0
    server_steps_cumulative: int = 0
    history = History()
    # Initialize client_state_dict
    client_state: dict[str | int, ClientState] = {
        cid: ClientState(0) for cid in range(cfg.fl.n_total_clients)
    }
    # Initialize parameters
    log(INFO, "Initializing global parameters")
    parameters = get_initial_parameters(cfg)
    momentum_vector = deepcopy(parameters_to_ndarrays(parameters))
    # Save the checkpoint to S3 Object Store (w/ model parameters)
    if cfg.pollen.checkpoint or cfg.use_s3_comm:
        assert (
            remote_up_down is not None
        ), "Cannot checkpoint without a RemoteUploaderDownloader object"
        upload_server_checkpoint(
            parameters=parameters,
            history=history,
            current_round=start_round,
            current_time_elapsed=time_offset,
            server_steps_cumulative=server_steps_cumulative,
            momentum_vector=momentum_vector,
            client_state=client_state,
            remote_up_down=remote_up_down,
        )
    return (
        parameters,
        history,
        start_round,
        time_offset,
        server_steps_cumulative,
        client_state,
        momentum_vector,
    )


def resume_from_round(
    cfg: BaseConfig, remote_up_down: RemoteUploaderDownloader
) -> tuple[
    Parameters, History, int, float, int, dict[str | int, ClientState], NDArrays | None
]:
    """Resume from a previous round.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.
    remote_up_down : RemoteUploaderDownloader
        The object to upload/download files from/to the S3 Object Store.

    Returns
    -------
    tuple[
        Parameters,
        History,
        int,
        float,
        int,
        dict[str | int, ClientState],
        NDArrays | None
    ]
        The model parameters, the history, the round number, the time offset, the
        cumulative number of steps, the client state, and the momentum vector.
    """
    cfg.pollen.resume_round = interpret_resume_round(
        resume_round=cfg.pollen.resume_round,
        server_path=(
            f"s3://{cfg.s3_comm_config.bucket_name}/" f"{cfg.run_uuid}/server/"
        ),
        # NOTE: Check whether we can relax this condition
        raise_error=cfg.pollen.resume_round != -1,
    )
    assert (
        cfg.pollen.resume_round is not None
    ), "Cannot resume run if `cfg.pollen.resume_round` is None"
    # NOTE: Check whether we can relax this condition
    if cfg.pollen.resume_round == -1:
        log(INFO, "No checkpoint found for resuming. Starting from scratch.")
        return initialize_round(cfg, remote_up_down)
    log(DEBUG, "Resume round %s", cfg.pollen.resume_round)

    log(INFO, "Resuming from checkpoint")
    return download_server_checkpoint(cfg, remote_up_down)
