"""Utility functions for broadcasting to clients on main server loop in flwr next."""

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
        A flag indicating whether to keep the original input parameters.

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
    message_ids = [message_id for message_id in message_ids if message_id]
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
    for message_res in all_broadcastres:
        assert "broadcast" in message_res.configs_records, "Broadcast key not found"
        assert (
            "status" in message_res.configs_records["broadcast"]
        ), "Status key not found"
        if message_res.configs_records["broadcast"]["status"] != "OK":
            raise ValueError("Broadcast failed")
