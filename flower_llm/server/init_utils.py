"""Utility functions for initialization task on main server loop in flwr next."""

import copy
from logging import DEBUG, INFO
import operator
import re
import numpy as np

from flower_llm.clients.llm_client_functions import (
    _get_trainer_object,
    get_initial_parameters,
)
from flower_llm.server.s3_utils import (
    download_server_checkpoint,
    interpret_resume_round,
    upload_server_checkpoint,
)
from flower_llm.utils import (
    ClientState,
    get_parameters_from_state,
)
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Parameters,
    log,
)


import os
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import list_remote_objects


from flower_llm.conf.base_schema import BaseConfig
from flower_llm.wandb_history import WandbHistory


def get_centralized_run_parameters(dummy_config: BaseConfig) -> Parameters:
    """Retrieve the parameters from a centralized run.

    Args
    ----------
    dummy_config : BaseConfig
        The configuration object containing the settings for the federated learning
        server.
    remote_up_down : RemoteUploaderDownloader
        The object to upload/download files from/to the S3 Object Store.

    Returns
    -------
    Params
        The parameters from the centralized run.
    """
    dummy_config = copy.deepcopy(dummy_config)
    desired_steps = dummy_config.pollen.restore_cent_run_batches
    folder = f"s3://checkpoints/{dummy_config.pollen.restore_cent_run_uuid}"
    remote_objects = list_remote_objects(folder)
    log(INFO, f"Restoring from centralized run, found {remote_objects}")
    sorted_pairs = sorted(
        [
            (
                int(reg.group(1)),  # epoch number
                int(reg.group(2)),  # number of batches
            )
            for path in remote_objects
            if (reg := re.search(r"/ep(\d+)-ba(\d+)", path)) is not None
        ],
        key=operator.itemgetter(1),
    )
    path_to_check = next(
        (
            (epoch, batches)
            for epoch, batches in sorted_pairs
            if batches == desired_steps
        ),
        None,
    )
    if path_to_check is None:
        raise ValueError(f"Could not find a checkpoint with {desired_steps} batches")
    epoch, batches = path_to_check

    dummy_config_llm = dummy_config.llm_config
    dummy_config_llm.load_path = folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
    dummy_config_llm.load_ignore_keys = [
        "*scheduler*",
        "*optim*",
        "*dataset_state*",
    ]
    os.environ["APPOINTED_CUDA_DEVICE"] = str(None)
    dummy_config_llm.save_folder = None
    dummy_config_llm.device_train_microbatch_size = 1
    trainer, *_ = _get_trainer_object(dummy_config_llm, cid=None, no_data_loading=True)
    return ndarrays_to_parameters(
        get_parameters_from_state(
            {},
            trainer,
        )
    )


def initialize_round(
    cfg: BaseConfig,
    remote_up_down: RemoteUploaderDownloader | None,
    parameters: Parameters | None = None,
) -> tuple[
    Parameters,
    WandbHistory,
    int,
    float,
    int,
    dict[str | int, ClientState],
    NDArrays | None,
    NDArrays | None,
]:
    """Initialize the state for a new round of federated learning.

    This function sets up the initial state for a new round of federated learning,
    including initializing bookkeeping variables, client states, global parameters, and
    optionally saving the initial checkpoint to an S3 Object Store if configured. It
    prepares the server for the federated learning process by setting the starting
    round, time offset, cumulative server steps, and initializing the momentum vector
    for optimization algorithms.

    Args
    ----------
    cfg : BaseConfig
        The configuration object containing settings for federated learning and system
        behavior.
    remote_up_down : RemoteUploaderDownloader | None
        An optional uploader/downloader object for interacting with remote storage,
        required if checkpointing or S3 communication is enabled.
    parameters: Parameters | None
        Optional initial parameters to use, e.g from a centralized run.

    Returns
    -------
    tuple
        A tuple containing the initialized global parameters, history object, starting
        round number, time offset, cumulative server steps, client state dictionary,
        the initial momentum vector (or None if not applicable), the initial second
        momentum vector (or None if not applicable).

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
    history = WandbHistory(use_wandb=cfg.use_wandb)
    # Initialize client_state_dict
    client_state: dict[str | int, ClientState] = {
        cid: ClientState(0) for cid in range(cfg.fl.n_total_clients)
    }
    if parameters is None:
        # Initialize parameters only if not provided
        log(INFO, "Initializing global parameters")
        parameters = get_initial_parameters(cfg)

    momentum_vector: NDArrays = [
        np.zeros_like(x) for x in parameters_to_ndarrays(parameters)
    ]
    second_momentum_vector: NDArrays = [
        np.zeros_like(x) for x in parameters_to_ndarrays(parameters)
    ]
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
            second_momentum_vector=second_momentum_vector,
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
        second_momentum_vector,
    )


def resume_from_round(
    cfg: BaseConfig, remote_up_down: RemoteUploaderDownloader
) -> tuple[
    Parameters,
    WandbHistory,
    int,
    float,
    int,
    dict[str | int, ClientState],
    NDArrays | None,
    NDArrays | None,
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
        WandbHistory,
        int,
        float,
        int,
        dict[str | int, ClientState],
        NDArrays | None,
        NDArrays | None
    ]
        The model parameters, the history, the round number, the time offset, the
        cumulative number of steps, the client state, and the momentum vectors.
    """
    cfg.pollen.resume_round = interpret_resume_round(
        resume_round=cfg.pollen.resume_round,
        run_uuid_path=(f"s3://{cfg.s3_comm_config.bucket_name}/{cfg.run_uuid}/"),
        # NOTE: Check whether we can relax this condition
        raise_error=cfg.pollen.resume_round != -1,
        state_keys=(
            "state.bin",
            "current_server_parameters",
            "current_momentum_vector",
        ),
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
