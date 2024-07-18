"""Utility functions for running the main server loop in flwr next."""

import ast
from collections.abc import Callable, Generator
from copy import deepcopy
import copy
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from typing import Any, cast
import time

from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
    get_raw_model_parameters,
)
from flower_llm.pollen_server import TooManyFailuresError
from flower_llm.server.s3_utils import download_server_checkpoint, replace_clients_updates_with_remote, upload_server_checkpoint
from flower_llm.utils import (
    ClientState,
    load_model_parameters_from_file,
    obtain_sorted_runs,
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
        The minimum number of client nodes that must be connected.
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


def message_collaborative(
    driver: Driver,
    message_type: str,
    sampled_clients: list[int] | list[str],
    gen_instructions: Callable[[int, int | str], dict[str, Scalar]],
    all_node_ids: list[int],
    current_round: int,
) -> Generator[Message, None, None]:
    """Fit collaboratively.

    Parameters
    ----------
    driver : Driver
        The driver object used for creating and pushing messages.
    record_sets: list[RecordSet]
        A list of record sets to be sent to the nodes for fitting.
    all_node_ids : list[int]
        A list of all node IDs to which the messages will be sent.
    current_round : int
        The current round of fitting, used for grouping messages.

    Yields
    ------
    Generator[Message, None, None]
        A generator that yields messages from the driver.
    """
    record_sets: list[RecordSet] = [
        fitins_to_recordset(
            FitIns(
                parameters=Parameters(tensors=[], tensor_type="empty"),
                config={cid: gen_instructions(current_round, cid)},  # type: ignore[reportArgumentType,dict-item]
            ),
            keep_input=True,
        )
        for cid in sampled_clients
    ]

    messages_to_nodes: dict[str, int] = {}
    messages: list[Message] = []
    for node_id in all_node_ids:
        if not record_sets:
            break
        message = driver.create_message(
            content=record_sets.pop(),
            message_type=message_type,
            dst_node_id=node_id,
            group_id=str(current_round),
            ttl=DEFAULT_TTL,
        )
        messages.append(message)
    message_ids = driver.push_messages(messages)
    received = 0
    total = len(record_sets)
    log(DEBUG, f"Pushed messages{messages_to_nodes}")

    while received < total:
        replies = list(driver.pull_messages(message_ids=message_ids))
        messages = []
        for res in replies:
            log(DEBUG, f"Got 1 {'result' if res.has_content() else 'error'}")
            received += 1
            node_id = res.metadata.src_node_id
            if record_sets:
                message = driver.create_message(
                    content=record_sets.pop(),
                    message_type=message_type,
                    dst_node_id=node_id,
                    group_id=str(current_round),
                    ttl=DEFAULT_TTL,
                )
        message_ids = driver.push_messages(messages)
        log(DEBUG, f"Pushed messages{messages_to_nodes}")
        for res in replies:
            yield res


def message_independent(
    driver: Driver,
    message_type: str,
    gen_instructions: Callable[[int, int | str], dict[str, Scalar]],
    all_node_ids: list[int],
    current_round: int,
    assignment_function: Callable[[int], list[int] | list[str]],
) -> Generator[Message, None, None]:
    """Fit collaboratively.

    Parameters
    ----------
    driver : Driver
        The driver object used for creating and pushing messages.
    record_sets: list[RecordSet]
        A list of record sets to be sent to the nodes for fitting.
    all_node_ids : list[int]
        A list of all node IDs to which the messages will be sent.
    assignment_function : Callable[[int], list[int | str]]
        A function that assigns record sets to nodes.
    current_round : int
        The current round of fitting, used for grouping messages.

    Yields
    ------
    Generator[Message, None, None]
        A generator that yields messages from the driver.
    """
    messages = []
    for node_id in all_node_ids:
        cids_to_train = assignment_function(node_id)
        if cids_to_train:
            message = driver.create_message(
                content=create_merged_recordset(
                    cids_to_train, current_round, gen_instructions
                ),
                message_type=message_type,
                dst_node_id=node_id,
                group_id=str(current_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)
    message_ids = list(driver.push_messages(messages))
    log(DEBUG, f"Pushed messages{messages}")
    total_messages = len(message_ids)
    received_messages = 0
    while received_messages < total_messages:
        for res in driver.pull_messages(message_ids=message_ids):
            log(DEBUG, f"Got 1 {'result' if res.has_content() else 'error'}")
            received_messages += 1
            yield res


def get_rr_assignment_function(
    sampled_clients: list[int] | list[str],
    all_node_ids: list[int],
) -> Callable[[int], list[int] | list[str]]:
    """Create a round-robin assignment function for the given clients and nodes."""

    def assignment_function(node_id: int) -> list[int] | list[str]:
        return [
            client_id
            for i, client_id in enumerate(sampled_clients)
            if i % len(all_node_ids) == all_node_ids.index(node_id)
        ]  # type: ignore[reportReturnType,return-value]

    return assignment_function


def create_merged_recordset(
    sampled_clients: list[int] | list[str],
    current_round: int,
    fit_ins_function: Callable[[int, int | str], dict[str, Scalar]],
) -> RecordSet:
    """Create a merged record set for the given clients and fit_ins_function.

    Parameters
    ----------
    sampled_clients : list[int | str]
        A list of client identifiers to be used in the fit_ins_function.
    fit_ins_function : Callable[[int, int | str], dict[str, Scalar]]
        A function that generates the fit_ins configuration for a given client.

    Returns
    -------
    RecordSet
        The merged record set containing the fit_ins configurations for each client.
    """
    configs = {cid: fit_ins_function(current_round, cid) for cid in sampled_clients}
    record_set = fitins_to_recordset(
        FitIns(
            parameters=Parameters(tensors=[], tensor_type="empty"),
            config=configs,  # type: ignore[reportArgumentType,arg-type]
        ),
        keep_input=True,
    )
    return record_set


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
