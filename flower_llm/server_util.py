"""TODO:"""

import ast
from copy import deepcopy
import copy
from dataclasses import asdict
from logging import DEBUG, INFO
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any, cast
import time

from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
    get_raw_model_parameters,
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
    Message,
    MessageType,
    DEFAULT_TTL,
    ConfigsRecord,
    RecordSet,
)
from flwr.common.recordset_compat import parameters_to_parametersrecord
from flwr.server import Driver, History
from omegaconf import OmegaConf
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path

from flower_llm.conf.base_schema import BaseConfig


class NoCheckpointsFoundError(Exception):
    """Exception raised when there are no checkpoints in the path looked up."""


def wait_for_nodes_to_connect(driver: Driver, n_nodes: int, timeout: float = 3) -> None:
    """Wait for a specified number of client nodes to connect.

    This function continuously checks the number of client nodes that have connected by
    calling the `get_node_ids` method on the provided driver object. It logs the current
    number of connected client nodes at each check. The function will exit once the
    number of connected client nodes meets or exceeds the specified `n_nodes`. If the
    required number of nodes are not found, the function will wait for the specified
    `timeout` period before checking again.

    Parameters
    ----------
    driver : Driver
        The driver object used to interact with the client nodes.
    n_nodes : int
        The minimum number of client nodes that must be connected before the function returns.
    timeout : float, optional
        The time in seconds to wait between checks. Default is 3 seconds.

    Returns
    -------
    None
    """
    # The Driver API might not immediately return enough client node IDs, so we
    # loop and wait until enough client nodes are available.
    while True:
        all_node_ids = driver.get_node_ids()
        log(DEBUG, f"Got {len(all_node_ids)} client nodes: {all_node_ids}")
        if len(all_node_ids) >= n_nodes:
            break
        time.sleep(timeout)


def broadcast_parameters_to_recordset(
    parameters: Parameters, keep_input: bool
) -> RecordSet:
    """Convert Parameters into RecordSet for broadcasting, optionally keeping the input.

    This function takes a set of parameters and converts them into a format suitable for
    broadcasting to a record set. It allows for keeping the original input parameters
    as part of the conversion process. The converted parameters and a configuration
    indicating the action type ("set_parameters") are added to the record set.

    Parameters
    ----------
    parameters : Parameters
        The parameters to be converted and broadcasted.
    keep_input : bool
        A flag indicating whether to keep the original input parameters in the conversion.

    Returns
    -------
    RecordSet
        The record set containing the converted parameters and configuration record.
    """
    recordset = RecordSet()
    parametersrecord = parameters_to_parametersrecord(parameters, keep_input)
    recordset.parameters_records["broadcastins.parameters"] = parametersrecord
    recordset.configs_records["query"] = ConfigsRecord({"type": "set_parameters"})
    return recordset


def broadcast_parameters_to_nodes(
    driver: Driver, parameters: Parameters, node_ids: list[int], current_round: int
) -> None:
    """Broadcast parameters to specified nodes for a given round.

    This function creates and sends a broadcast message with parameters to each node
    specified in `node_ids` for the current round. It waits for all nodes to reply
    and ensures that all broadcast operations were successful.

    Parameters
    ----------
    driver : Driver
        The driver object used for creating and pushing messages.
    parameters : Parameters
        The parameters to be broadcasted.
    node_ids : list[int]
        A list of node IDs to which the parameters will be broadcasted.
    current_round : int
        The current round of broadcasting, used for grouping messages.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If any of the broadcast operations are reported as failed by the nodes.
    """
    # Create one message per node from one single recordset
    messages = []
    recordset = broadcast_parameters_to_recordset(
        parameters=parameters, keep_input=True
    )
    for node_id in node_ids:
        message = driver.create_message(
            content=recordset,
            message_type=MessageType.QUERY,
            dst_node_id=node_id,
            group_id=str(current_round),
            ttl=DEFAULT_TTL,
        )
        messages.append(message)
    # Push all messages to all nodes
    message_ids = driver.push_messages(messages)
    log(DEBUG, f"Pushed {len(messages)} broadcast messages to {len(node_ids)} nodes")
    # Wait for results, ignore empty message_ids
    message_ids = [message_id for message_id in message_ids if message_id != ""]
    all_replies: list[Message] = []
    while True:
        replies = driver.pull_messages(message_ids=message_ids)
        for res in replies:
            log(DEBUG, f"Got 1 {'result' if res.has_content() else 'error'}")
        all_replies += replies
        if len(all_replies) == len(message_ids):
            break
        log(DEBUG, "Pulling messages...")
        time.sleep(3)
    # Filter correct results
    all_broadcastres = [msg.content for msg in all_replies if msg.has_content()]
    log(DEBUG, f"Received {len(all_broadcastres)} results")
    # Elaborate results
    for res in all_broadcastres:
        assert "broadcast" in res.configs_records, "Broadcast key not found"
        assert "status" in res.configs_records["broadcast"], "Status key not found"
        if res.configs_records["broadcast"]["status"] != "OK":
            raise ValueError("Broadcast failed")


def interpret_resume_round(
    resume_round: int | None, server_path: str, raise_error: bool = True
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
        server_round_indices = obtain_sorted_runs(server_path)
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
