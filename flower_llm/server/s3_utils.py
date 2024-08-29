"""Utility functions for running S3-related tasks on main server loop in flwr next."""

import ast
from dataclasses import asdict
from logging import DEBUG, INFO, WARNING
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any
import time

from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
)
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
    ConfigsRecord,
    Message,
    Code,
)
from flwr.common.recordset_compat import (
    _extract_status_from_recordset,  # noqa: PLC2701
    parameters_to_parametersrecord,
    parametersrecord_to_parameters,
)
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path


from flower_llm.conf.base_schema import BaseConfig
from flower_llm.wandb_history import WandbHistory


class NoCheckpointsFoundError(Exception):
    """Exception raised when there are no checkpoints in the path looked up."""


def extract_s3_comm_config_from_configrecord(
    s3_comm_config: ConfigsRecord,
) -> tuple[str, str, str]:
    """Extract S3 communication configuration details from a ConfigsRecord object.

    This function parses a ConfigsRecord object containing S3 communication
    configuration and extracts essential information required for S3 operations.
    Specifically, it retrieves the `endpoint_id`, `file_name`, and `current_round`
    from the ConfigsRecord. These values are crucial for identifying the correct S3
    bucket and path, and for versioning or round-specific operations.

    Parameters
    ----------
    s3_comm_config : ConfigsRecord
        A ConfigsRecord object containing the S3 communication configuration. Expected
        to have keys for `endpoint_id`, `file_name`, and `current_round`.

    Returns
    -------
    tuple[str, str, str]
        A tuple containing `endpoint_id`, `file_name`, and `current_round` as strings.

    Raises
    ------
    ValueError
        If any of the required keys (`endpoint_id`, `file_name`, or `current_round`) are
        missing from the ConfigsRecord.

    Notes
    -----
    The function ensures that all returned values are strings, even if they are provided
    as different types in the ConfigsRecord. This standardization facilitates their use
    in S3 operations without further type checking or conversion.
    """
    # Extract endpoint id from the content of the message
    endpoint_id: Any
    if "endpoint_id" in s3_comm_config:
        endpoint_id = str(s3_comm_config["endpoint_id"])
    else:
        raise ValueError("endpoint_id is not present in the message")
    file_name: Any
    if "file_name" in s3_comm_config:
        file_name = str(s3_comm_config["file_name"])
    else:
        raise ValueError("file_name is not present in the message")
    folder_name: Any
    if "folder_name" in s3_comm_config:
        folder_name = str(s3_comm_config["folder_name"])
    else:
        raise ValueError("folder_name is not present in the message")
    return endpoint_id, file_name, folder_name


def interpret_resume_round(
    resume_round: int | None,
    server_path: str,
    state_keys: tuple[str, ...],
    raise_error: bool = True,
) -> int | None:
    """Interpret the resume round parameter for server checkpoint resumption.

    This function interprets the `resume_round` parameter, which specifies the round
    to resume server operations from. If `resume_round` is negative, it is treated as
    an index into the list of sorted rounds obtained from the server's path, allowing
    for reverse indexing. If `resume_round` is None, the function returns None,
    indicating no specific round to resume from. An error is raised if no checkpoints
    are found when `raise_error` is True and `resume_round` is negative but no rounds
    are available.

    Parameters
    ----------
    resume_round : int | None
        The round number to resume from. If negative, treated as a reverse index. If
        None, indicates no resumption is required.
    server_path : str
        The path to the server's checkpoint directory.
    raise_error : bool, optional
        Whether to raise an error if no checkpoints are found and `resume_round` is
        negative. Default is True.

    Returns
    -------
    int | None
        The interpreted round number to resume from, or None if no resumption.

    Raises
    ------
    NoCheckpointsFoundError
        If `raise_error` is True, no checkpoints are found, and `resume_round` < 0.
    """
    log(
        DEBUG,
        "The parameter `resume_round=%s` will be interpret as an index "
        "for the list of rounds for the server_path=%s",
        resume_round,
        server_path,
    )
    if resume_round is None:
        return None
    if resume_round < 0:
        server_round_indices = obtain_sorted_runs(server_path, state_keys)
        log(DEBUG, "Found server round indices %s", server_round_indices)
        if not server_round_indices and raise_error:
            raise NoCheckpointsFoundError
        if server_round_indices:
            resume_round = server_round_indices[resume_round]
    return resume_round


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
        cfg.pollen.resume_round,
        server_path,
        state_keys=(
            "state.bin",
            "current_server_parameters",
            "current_momentum_vector",
        ),
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
    history: WandbHistory,
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
    history : WandbHistory
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
    is_second_momentum: bool = False,
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
    filename_no_ext = (
        "current_momentum_vector"
        if not is_second_momentum
        else "current_second_momentum_vector"
    )
    dump_mom_vec_time = time.time()
    dump_model_parameters_to_file(
        Path(temp_dir.name) / f"{filename_no_ext}.npz",
        momentum_vector,
    )
    log(
        DEBUG,
        "Push momentum vector to S3 Object Store. " "Time to dump to disk: %s",
        time.time() - dump_mom_vec_time,
    )
    upload_file_to_s3(
        remote_up_down,
        f"{current_round}/{filename_no_ext}.npz",
        Path(temp_dir.name) / f"{filename_no_ext}.npz",
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
    history: WandbHistory | None,
    current_round: int,
    current_time_elapsed: float | None,
    server_steps_cumulative: int | None,
    momentum_vector: NDArrays | None,
    second_momentum_vector: NDArrays | None,
    client_state: dict[str | int, ClientState] | None,
    remote_up_down: RemoteUploaderDownloader,
) -> None:
    """Upload the server checkpoint to the S3 Object Store.

    This function uploads various components of the server's state as part of the
    checkpointing process. It includes the model parameters, training history, current
    round, time elapsed, cumulative server steps, momentum vectors, and client states.
    Each component is uploaded separately, and only if it is not None.

    Parameters
    ----------
    parameters : Parameters | None
        The model parameters to be uploaded, if any.
    history : WandbHistory | None
        The training history to be uploaded, if any.
    current_round : int
        The current federated learning round.
    current_time_elapsed : float | None
        The current time elapsed since the start of federated learning, if applicable.
    server_steps_cumulative : int | None
        The cumulative number of server steps taken, if applicable.
    momentum_vector : NDArrays | None
        The momentum vector to be uploaded, if any.
    second_momentum_vector : NDArrays | None
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
    # Dump and upload second momentum vector if present
    if second_momentum_vector is not None:
        _upload_momentum_vector(
            current_round=current_round,
            momentum_vector=second_momentum_vector,
            remote_up_down=remote_up_down,
            is_second_momentum=True,
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
    Parameters,
    WandbHistory,
    int,
    float,
    int,
    dict[str | int, ClientState],
    NDArrays | None,
    NDArrays | None,
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
        cumulative server steps, client states, and (potentially) the momentum vectors.
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
    history: WandbHistory = server_state["history"]
    if "client_state" in server_state:
        saved_client_state: dict[str | int, dict[str, Any]] = ast.literal_eval(
            server_state["client_state"]
        )
    else:
        # NOTE: This `local_steps_cumulative` is just used for backlogging and not by
        # any logic during training so we put a zero for now.
        log(
            WARNING,
            "No client state found in the checkpoint."
            "We will put dummy values of zero.",
        )
        saved_client_state = {
            cid: {"local_steps_cumulative": 0} for cid in range(cfg.fl.n_total_clients)
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

    # Momentum vector
    second_momentum_vector: NDArrays | None = None
    remote_file_name_momentum = (
        server_path + f"{cfg.pollen.resume_round}/current_second_momentum_vector.npz"
    )
    if "momentum" in server_state:
        log(DEBUG, "Get momentum vector from server state")
        second_momentum_vector = server_state["momentum"]
    elif validate_given_remote_path(remote_file_name_momentum):
        log(DEBUG, "Pull momentum from S3 Object Store")
        # Set the file names depending on the extension found
        remote_file_name = (
            f"{cfg.pollen.resume_round}/current_second_momentum_vector.npz"
        )
        local_file_name = Path(temp_dir.name) / "current_second_momentum_vector.npz"
        # Download the parameters
        download_file_from_s3(remote_up_down, remote_file_name, local_file_name)
        second_momentum_vector = load_model_parameters_from_file(local_file_name)
    log(INFO, "Checkpoint loaded")
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


def replace_remote_with_parameters_in_recordset(
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    outgoing_message: Message,
    use_s3_comm: bool,
    msg_str: str,
) -> Message:
    """Replace parameters in the recordset of a message with ref to S3 location.

    This function modifies the `outgoing_message` by uploading its parameters to an S3
    bucket and replacing the parameters in the message with references to their
    locations in S3. This is only done if S3 communication is used (`use_s3_comm` is
    True) and a `remote_uploader_downloader` is provided. It handles the creation of a
    temporary directory for storing parameters locally before uploading, constructs the
    S3 file name based on message content, uploads the file, and then updates the
    message to reference the S3 location. If S3 communication is not used, the original
    message is returned without modification.

    Parameters
    ----------
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The uploader/downloader instance for interacting with S3. Required if
        `use_s3_comm` is True.
    outgoing_message : Message
        The message whose parameters are to be uploaded to S3. The message is modified
        in-place.
    use_s3_comm : bool
        Flag indicating whether to use S3 for communication. If False, the function
        returns the message unchanged.
    msg_str : str, optional
        A string identifier used to prefix keys in the message's content, by default
        "fitres".

    Returns
    -------
    Message
        The modified message with parameters replaced by S3 references, or the original
        message if S3 communication is not used.

    Raises
    ------
    ValueError
        If required keys (`endpoint_id`, `file_name`, or `current_round`) are missing
        from the message's content.
    TypeError
        If the `endpoint_id` in the message's content is not a string.

    Notes
    -----
    The function assumes the existence of `dump_model_parameters_to_file`,
    `parameters_to_ndarrays`, `parametersrecord_to_parameters`,
    `parameters_to_parametersrecord`, `upload_file_to_s3`, `validate_given_remote_path`,
    and `log` functions, as well as the `DEBUG` constant for logging purposes. It also
    relies on the structure of the `Message` object and the `RemoteUploaderDownloader`
    interface for S3 interactions.
    """
    # Extract the content of the incoming message
    recordset = outgoing_message.content
    # Check if it's necessary to download from S3
    if use_s3_comm and remote_uploader_downloader is not None:
        # Create a temporary directory for storing the downloaded parameters
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        s3_comm_config = recordset.configs_records[f"{msg_str}.s3_comm_config"]
        parameters = recordset.parameters_records[f"{msg_str}.parameters"]
        # Extract endpoint id from the content of the message
        endpoint_id, file_name, folder_name = extract_s3_comm_config_from_configrecord(
            s3_comm_config
        )
        # Set the file names
        remote_file_name = f"{folder_name}/{endpoint_id}/{file_name}.npz"
        local_file_name = Path(temp_dir.name) / f"{endpoint_id}_{file_name}.npz"
        dump_model_parameters_to_file(
            local_file_name,
            parameters_to_ndarrays(
                parametersrecord_to_parameters(record=parameters, keep_input=False)
            ),
        )
        # Upload the parameters to S3 Object Store
        upload_file_to_s3(remote_uploader_downloader, remote_file_name, local_file_name)
        # Empty the recordset parameters
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(
                Parameters(tensors=[], tensor_type="empty"),
                False,
            )
        )
        # Update the content of the message
        outgoing_message.content = recordset
        # NOTE: See if this is still necessary!
        # Check whether the server has uploaded the parameters
        file_found = False
        remote_file_name_no_ext = (
            f"s3://{remote_uploader_downloader.remote_bucket_name}/"
            f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
            f"{folder_name}/{endpoint_id}/{file_name}"
        )
        while not file_found:
            file_found = validate_given_remote_path(
                remote_file_name_no_ext + ".bin"
            ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
            time.sleep(0.5)
        log(
            DEBUG,
            "Node %s parameters have been pushed to the S3",
            endpoint_id,
        )
        return outgoing_message
    else:
        # No translation performed as we assume the task failed
        return outgoing_message


def replace_parameters_in_recordset_with_remote(
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    incoming_message: Message,
    use_s3_comm: bool,
    msg_str: str,
) -> Message:
    """Replace parameters in the recordset of an incoming message with those from S3.

    This function checks the status of the task associated with the incoming message.
    If the task was successful and S3 communication is enabled, it downloads the
    parameters from S3 and updates the incoming message's recordset with these
    parameters. The function supports downloading parameters in either binary or NumPy
    compressed formats. It ensures that the parameters are only downloaded if the task
    was successful and S3 communication is being used. If the task failed or S3
    communication is not enabled, the original message is returned without modification.

    Parameters
    ----------
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The uploader/downloader instance for interacting with S3. Required if
        `use_s3_comm` is True.
    incoming_message : Message
        The message whose parameters are to be replaced with those downloaded from S3.
    use_s3_comm : bool
        Flag indicating whether to use S3 for communication. If False, the function
        returns the message unchanged.
    msg_str : str
        A string identifier used to prefix keys in the message's content and to locate
        the specific parameters
        within the recordset.

    Returns
    -------
    Message
        The modified message with parameters replaced by those downloaded from S3, or
        the original message if S3 communication is not used or the task associated with
        the message failed.

    Raises
    ------
    ValueError
        If the required S3 communication configuration (`endpoint_id`, `file_name`, or
        `current_round`) is missing from the message's content.

    Notes
    -----
    The function assumes the existence of `extract_s3_comm_config_from_configrecord`,
    `validate_given_remote_path`, `download_file_from_s3`, `ndarrays_to_parameters`,
    `load_model_parameters_from_file`, `parameters_to_parametersrecord`, and `log`
    functions, as well as the `DEBUG` constant for logging purposes. It also relies on
    the structure of the `Message` object and the `RemoteUploaderDownloader` interface
    for S3 interactions.
    """
    # Extract the content of the incoming message
    recordset = incoming_message.content
    status = _extract_status_from_recordset(msg_str, recordset)
    if status.code != Code.OK:
        # No translation performed as we assume the task failed
        return incoming_message
    # Check if it's necessary to download from S3
    if use_s3_comm and remote_uploader_downloader is not None:
        # Create a temporary directory for storing the downloaded parameters
        temp_dir: TemporaryDirectory = TemporaryDirectory()
        s3_comm_config = recordset.configs_records[f"{msg_str}.s3_comm_config"]
        # Extract endpoint id from the content of the message
        endpoint_id, file_name, folder_name = extract_s3_comm_config_from_configrecord(
            s3_comm_config
        )
        # Check whether the server has uploaded the parameters
        file_found = False
        remote_file_name_no_ext = (
            f"s3://{remote_uploader_downloader.remote_bucket_name}/"
            f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
            f"{folder_name}/{endpoint_id}/{file_name}"
        )
        while not file_found:
            file_found = validate_given_remote_path(
                remote_file_name_no_ext + ".bin"
            ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
            time.sleep(0.5)
        # Set the file names depending on the extension found
        remote_file_name = (
            f"{folder_name}/{endpoint_id}/{file_name}.bin"
            if validate_given_remote_path(remote_file_name_no_ext + ".bin")
            else f"{folder_name}/{endpoint_id}/{file_name}.npz"
        )
        local_file_name = (
            Path(temp_dir.name) / f"tmp-{endpoint_id}.bin"
            if validate_given_remote_path(remote_file_name_no_ext + ".bin")
            else Path(temp_dir.name) / f"tmp-{endpoint_id}.npz"
        )
        download_file_from_s3(
            remote_uploader_downloader, remote_file_name, local_file_name
        )
        parameters = ndarrays_to_parameters(
            load_model_parameters_from_file(local_file_name)
        )
        recordset.parameters_records[f"{msg_str}.parameters"] = (
            parameters_to_parametersrecord(parameters, False)
        )
        incoming_message.content = recordset
        log(
            DEBUG,
            "Node %s parameters have been read from disk and assigned to the Message",
            endpoint_id,
        )
        return incoming_message
    else:
        # No translation performed as we assume the task failed
        return incoming_message
