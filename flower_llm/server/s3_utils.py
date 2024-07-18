"""Utility functions for running S3-related tasks on main server loop in flwr next."""

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


def import_checkpoints(
    remote_up_down: RemoteUploaderDownloader,
    cfg: BaseConfig,
) -> None:
    """Import checkpoints from a previous run for restoration based on configuration.

    This function validates the  configuration parameters for checkpoints import,
    constructs the server path for the checkpoints, and calculates the resume round.
    It then calls `copy_old_checkpoints_to_new_run` to copy the checkpoints from
    the previous run to the current run's server path. The function ensures that all
    configuration parameters are not None and logs the restoration process.

    Parameters
    ----------
    remote_up_down : RemoteUploaderDownloader
        The uploader and downloader instance for interacting with remote storage.
    cfg : BaseConfig
        The configuration object containing run UUIDs, checkpoint information, and S3
        communication configuration.

    Raises
    ------
    AssertionError
        If any of the required configuration parameters (`run_uuid`, `restore_run_uuid`,
        `checkpoint`, `use_s3_comm`, `bucket_name`, `resume_round`) are None.
    """
    assert (
        cfg.run_uuid is not None
    ), "Cannot import chkpt for restoration if `cfg.run_uuid` is None"
    assert (
        cfg.pollen.restore_run_uuid is not None
    ), "Cannot import chkpt for restoration if `cfg.pollen.restore_run_uuid` is None"
    assert (
        cfg.pollen.checkpoint is not None
    ), "Cannot import chkpt for restoration if `cfg.pollen.checkpoint` is None"
    assert (
        cfg.use_s3_comm is not None
    ), "Cannot import chkpt for restoration if `cfg.use_s3_comm` is None"
    assert (
        cfg.s3_comm_config.bucket_name is not None
    ), "Cannot import chkpt for restoration if `cfg.s3_comm_config.bucket_name` is None"
    log(DEBUG, "Importing checkpoints from a previous run for restoration.")
    server_path = (
        f"s3://{cfg.s3_comm_config.bucket_name}/"
        f"{cfg.pollen.restore_run_uuid}/server/"
    )
    cfg.pollen.resume_round = interpret_resume_round(
        cfg.pollen.resume_round, server_path
    )
    assert (
        cfg.pollen.resume_round is not None
    ), "Cannot import checkpoint for restoration if `cfg.pollen.resume_round` is None"
    log(DEBUG, "Restore round %s", cfg.pollen.resume_round)

    copy_old_checkpoints_to_new_run(
        remote_up_down=remote_up_down,
        bucket_uri=f"s3://{cfg.s3_comm_config.bucket_name}",
        run_uuid=cfg.run_uuid,
        restore_run_uuid=cfg.pollen.restore_run_uuid,
        restore_run_round=cfg.pollen.resume_round,
        restore_run_step=cfg.pollen.resume_round
        * int(cfg.llm_config.local_steps.replace("ba", "")),
        n_total_clients=cfg.fl.n_total_clients,
    )


def _upload_server_state(
    history: History,
    current_round: int,
    current_time_elapsed: float,
    server_steps_cumulative: int,
    client_state: dict[str | int, ClientState],
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the current server state to the S3 Object Store.

    This function serializes and uploads the current server state, including the
    federated learning round number, history, time elapsed, cumulative server steps,
    and client states, to the S3 Object Store. It creates a temporary directory to
    store the serialized state before uploading.

    Parameters
    ----------
    history : History
        The history object containing the training history.
    current_round : int
        The current federated learning round.
    current_time_elapsed : float
        The current time elapsed since the start of federated learning.
    server_steps_cumulative : int
        The cumulative number of server steps taken.
    client_state : dict[str | int, ClientState]
        A dictionary mapping client identifiers to their states.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    Returns
    -------
    None
    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    # Create the server state dictionary containing light stuff
    current_server_state = {
        "server_round": current_round,
        "history": history,
        "time_offset": current_time_elapsed,
        "client_state": str({k: asdict(v) for k, v in client_state.items()}),
        "server_steps_cumulative": server_steps_cumulative,
    }
    # Dump the server state to disk and upload to the S3 Object Store
    with open(Path(temp_dir.name) / "current_server_state.bin", "wb") as f:
        pickle.dump(current_server_state, f)
    log(DEBUG, "Push server state to S3")
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/state.bin",
        Path(temp_dir.name) / "current_server_state.bin",
    )


def _upload_momentum_vector(
    current_round: int,
    momentum_vector: NDArrays,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the current momentum vector to the S3 Object Store.

    This function serializes and uploads the current momentum vector, used in
    optimization algorithms, to the S3 Object Store for the given federated learning
    round. It creates a temporary directory to store the serialized momentum vector
    before uploading and logs the time taken to dump the momentum
    vector to disk.

    Parameters
    ----------
    current_round : int
        The current federated learning round.
    momentum_vector : NDArrays
        The momentum vector to be uploaded.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    Returns
    -------
    None
    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    log(DEBUG, "Dump momentum vector to disk")
    dump_mom_vec_time = time.time()
    dump_model_parameters_to_file(
        Path(temp_dir.name) / "current_momentum_vector.npz",
        momentum_vector,
    )
    log(
        DEBUG,
        "Push momentum vector to S3 Object Store. " "Time to dump to disk: %s",
        time.time() - dump_mom_vec_time,
    )
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/current_momentum_vector.npz",
        Path(temp_dir.name) / "current_momentum_vector.npz",
    )


def _upload_model_parameters(
    parameters: Parameters,
    current_round: int,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the model parameters to the S3 Object Store for the current round.

    This function serializes and uploads the model parameters to the S3 Object Store,
    organizing them by the current federated learning round. It first dumps them
    to a temporary file and then uploads this file.

    Parameters
    ----------
    parameters : Parameters
        The model parameters to be uploaded.
    current_round : int
        The current federated learning round.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    Returns
    -------
    None
    """
    # Create a temporary directory
    temp_dir = TemporaryDirectory()
    dump_model_time = time.time()
    dump_model_parameters_to_file(
        Path(temp_dir.name) / "current_server_parameters.npz",
        parameters_to_ndarrays(parameters),
    )
    log(
        DEBUG,
        "Push parameters to S3 Object Store. Time to dump to disk: %s",
        time.time() - dump_model_time,
    )
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/current_server_parameters.npz",
        Path(temp_dir.name) / "current_server_parameters.npz",
    )


def upload_server_checkpoint(
    parameters: Parameters | None,
    history: History | None,
    current_round: int,
    current_time_elapsed: float | None,
    server_steps_cumulative: int | None,
    momentum_vector: NDArrays | None,
    client_state: dict[str | int, ClientState] | None,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the server checkpoint to the S3 Object Store.

    This function uploads various components of the server's state as part of the
    checkpointing process. It includes the model parameters, training history, current
    round, time elapsed, cumulative server steps, momentum vector, and client states.
    Each component is uploaded separately, and only if it is not None.

    Parameters
    ----------
    parameters : Parameters | None
        The model parameters to be uploaded, if any.
    history : History | None
        The training history to be uploaded, if any.
    current_round : int
        The current federated learning round.
    current_time_elapsed : float | None
        The current time elapsed since the start of federated learning, if applicable.
    server_steps_cumulative : int | None
        The cumulative number of server steps taken, if applicable.
    momentum_vector : NDArrays | None
        The momentum vector to be uploaded, if any.
    client_state : dict[str | int, ClientState] | None
        The client states to be uploaded, if any.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.

    Returns
    -------
    None
    """
    # Uploading the server state
    if (
        history
        and current_time_elapsed is not None
        and server_steps_cumulative is not None
        and client_state
    ):
        _upload_server_state(
            history=history,
            current_round=current_round,
            current_time_elapsed=current_time_elapsed,
            server_steps_cumulative=server_steps_cumulative,
            client_state=client_state,
            remote_up_down=remote_up_down,
        )
    # Dump and upload momentum vector if present
    if momentum_vector is not None:
        _upload_momentum_vector(
            current_round=current_round,
            momentum_vector=momentum_vector,
            remote_up_down=remote_up_down,
        )
    # Dump and upload model parameters
    if parameters is not None:
        _upload_model_parameters(
            parameters=parameters,
            current_round=current_round,
            remote_up_down=remote_up_down,
        )


def download_server_checkpoint(
    cfg: BaseConfig,
    remote_up_down: RemoteUploaderDownloader,
    timeout: float = 0.5,
) -> tuple[
    Parameters, History, int, float, int, dict[str | int, ClientState], NDArrays | None
]:
    """Download the server checkpoint from the S3 Object Store.

    This function downloads the server checkpoint, including model parameters and
    potentially other state information, from the S3 Object Store. It checks for the
    existence of the server parameters file, waits until it is found (with a delay
    between checks defined by `timeout`), and then downloads the file. The downloaded
    parameters are then loaded from the file into memory. It also downloads the server
    state file, which contains the history, time offset, client states, and cumulative
    server steps. Finally, if a momentum vector is present, it is downloaded as well.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object containing S3 configurations and run UUID.
    remote_up_down : RemoteUploaderDownloader
        The uploader/downloader object for interacting with the S3 Object Store.
    timeout : float, optional
        The timeout in seconds to wait between checks for the server parameters file,
        by default 0.5.

    Returns
    -------
    tuple
        A tuple containing the loaded server checkpoint components. The exact components
        include model parameters, training history, current round, time elapsed,
        cumulative server steps, client states, and (potentially) the momentum vector.
    """
    # Set the path to server checkpoints
    server_path = f"s3://{cfg.s3_comm_config.bucket_name}/" f"{cfg.run_uuid}/server/"
    # Create a temporary directory
    temp_dir = TemporaryDirectory()

    # Model parameters
    # Check whether the server parameters exist
    file_found = False
    remote_file_name_no_ext = (
        server_path + f"{cfg.pollen.resume_round}/current_server_parameters"
    )
    while not file_found:
        file_found = validate_given_remote_path(
            remote_file_name_no_ext + ".bin"
        ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        time.sleep(timeout)
    # Set the server parameters file names depending on the extension found
    remote_file_name = (
        f"{cfg.pollen.resume_round}/current_server_parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else f"{cfg.pollen.resume_round}/current_server_parameters.npz"
    )
    local_file_name = (
        Path(temp_dir.name) / "current_server_parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else Path(temp_dir.name) / "current_server_parameters.npz"
    )
    log(DEBUG, "Pull server parameters from S3 Object Store")
    # Download the parameters
    download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
    log(DEBUG, "Read server parameters from disk")
    checkpoint_parameters = load_model_parameters_from_file(local_file_name)
    parameters = ndarrays_to_parameters(checkpoint_parameters)

    # Server state (history, time_offset, client_state, server_steps_cumulative)
    log(DEBUG, "Pull server state from S3 Object Store")
    # Download the server state from S3 Object Store
    download_file_from_s3(
        remote_up_down,
        f"{cfg.pollen.resume_round}/state.bin",
        str(Path(temp_dir.name) / "current_server_state.bin"),
    )
    log(DEBUG, "Read server state from disk")
    with open(Path(temp_dir.name) / "current_server_state.bin", "rb") as f:
        server_state = pickle.load(f)
    start_round = server_state["server_round"]
    assert (
        start_round == cfg.pollen.resume_round
    ), "Server round mismatch with checkpoint"
    history: History = server_state["history"]
    if "client_state" in server_state:
        saved_client_state: dict[str | int, dict[str, Any]] = ast.literal_eval(
            server_state["client_state"]
        )
    else:
        # TODO: Retro-compatibility with previous versions
        saved_client_state = {
            # cid: {"local_steps_cumulative": int(500 * cfg.pollen.resume_round)}
            cid: {"local_steps_cumulative": 0}
            for cid in range(cfg.fl.n_total_clients)
        }
    client_state = {k: ClientState(**v) for k, v in saved_client_state.items()}
    time_offset = 0.0
    if "time_offset" in server_state:
        time_offset = server_state["time_offset"]
    if "server_steps_cumulative" in server_state:
        server_steps_cumulative = server_state["server_steps_cumulative"]
    else:
        # Make it back compatible with the previous versions
        server_steps_cumulative = max(
            *[
                _client_state.local_steps_cumulative
                for _client_state in client_state.values()
            ],
            0,
        )

    # Momentum vector
    momentum_vector: NDArrays | None = None
    remote_file_name_momentum = (
        server_path + f"{cfg.pollen.resume_round}/current_momentum_vector.npz"
    )
    if "momentum" in server_state:
        log(DEBUG, "Get momentum vector from server state")
        momentum_vector = server_state["momentum"]
    elif validate_given_remote_path(remote_file_name_momentum):
        log(DEBUG, "Pull momentum from S3 Object Store")
        # Set the file names depending on the extension found
        remote_file_name = f"{cfg.pollen.resume_round}/current_momentum_vector.npz"
        local_file_name = Path(temp_dir.name) / "current_momentum_vector.npz"
        # Download the parameters
        download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
        momentum_vector = load_model_parameters_from_file(local_file_name)
    log(INFO, "Checkpoint loaded")
    return (
        parameters,
        history,
        start_round,
        time_offset,
        server_steps_cumulative,
        client_state,
        momentum_vector,
    )


def replace_clients_updates_with_remote(
    remote_uploader_downloader: RemoteUploaderDownloader,
    current_round: int,
    fit_res: FitRes,
) -> FitRes:
    """Replace the parameters in the FitRes with the ones from S3 Object Store."""
    pollen_temp_dir: TemporaryDirectory = TemporaryDirectory()

    endpoint_id: Any
    if "endpoint_id" in fit_res.metrics:
        endpoint_id = fit_res.metrics["endpoint_id"]
        del fit_res.metrics["endpoint_id"]
        if not isinstance(endpoint_id, str):
            raise TypeError("endpoint_id is not a string")
    else:
        raise ValueError("endpoint_id is not present in fit_res")
    # Check whether the server has uploaded the parameters
    file_found = False
    remote_file_name_no_ext = (
        f"s3://{remote_uploader_downloader.remote_bucket_name}/"  # type: ignore[union-attr]
        f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
        f"{current_round}/{endpoint_id}/parameters"
    )

    log(
        DEBUG,
        "Wait for NodeManager %s parameters to be in S3 Object Store at %s",
        endpoint_id,
        remote_file_name_no_ext,
    )

    while not file_found:
        file_found = validate_given_remote_path(
            remote_file_name_no_ext + ".bin"
        ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        time.sleep(0.5)

    # Set the file names depending on the extension found
    remote_file_name = (
        f"{current_round}/{endpoint_id}/parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else f"{current_round}/{endpoint_id}/parameters.npz"
    )
    local_file_name = (
        Path(pollen_temp_dir.name) / f"{endpoint_id}_current_server_parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else Path(pollen_temp_dir.name) / f"{endpoint_id}_current_server_parameters.npz"
    )
    log(
        DEBUG,
        "Pull Node %s parameters from S3 Object Store: %s -> %s",
        endpoint_id,
        remote_file_name,
        local_file_name,
    )
    download_file_from_s3(remote_uploader_downloader, remote_file_name, local_file_name)
    log(DEBUG, "Read server parameters from disk")
    fit_res.parameters = ndarrays_to_parameters(
        load_model_parameters_from_file(local_file_name)
    )

    log(
        DEBUG,
        "Node %s parameters have been read from disk and assigned to fit_res",
        endpoint_id,
    )

    return fit_res