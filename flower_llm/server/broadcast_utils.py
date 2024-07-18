"""Utility functions for broadcasting to clients on main server loop in flwr next."""

from logging import DEBUG
import time

from flower_llm.server.s3_utils import replace_remote_with_parameters_in_recordset
from flwr.common import (
    Parameters,
    log,
    Message,
    MessageType,
    DEFAULT_TTL,
    ConfigsRecord,
    RecordSet,
    Code,
)
from flwr.common.recordset_compat import parameters_to_parametersrecord
from flwr.server import Driver
from composer.loggers import RemoteUploaderDownloader


def parameters_to_broadcast_recordset(
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
    recordset.configs_records["query"] = ConfigsRecord({"type": "broadcast_parameters"})
    return recordset


def broadcast_parameters_to_nodes(
    driver: Driver,
    parameters: Parameters,
    node_ids: list[int],
    current_round: int,
    remote_uploader_downloader: RemoteUploaderDownloader | None,
    use_s3_comm: bool,
) -> None:
    """
    Broadcasts parameters to specified nodes using either direct messaging or S3.

    This function takes a set of parameters and broadcasts them to a list of node IDs.
    It supports two modes of communication: direct messaging through the `driver` and
    indirect messaging via S3 when `use_s3_comm` is True. The function first creates a
    recordset from the parameters, adds status and S3 configuration information to the
    recordset, and then either uploads the parameters to S3 (if `use_s3_comm` is True)
    or prepares them for direct messaging. It then sends the messages to all specified
    nodes and waits for their acknowledgments, ensuring all nodes have successfully
    received the parameters.

    Parameters
    ----------
    driver : Driver
        The communication driver responsible for sending and receiving messages.
    parameters : Parameters
        The parameters to be broadcasted to the nodes.
    node_ids : list[int]
        A list of node IDs to which the parameters will be broadcasted.
    current_round : int
        The current round of the operation, used for tracking and logging.
    remote_uploader_downloader : RemoteUploaderDownloader | None
        The remote uploader/downloader instance for S3 communication. Required if
        `use_s3_comm` is True.
    use_s3_comm : bool
        Flag indicating whether to use S3 for communication instead of direct messaging.

    Raises
    ------
    ValueError
        If any node reports a failure in receiving or processing the broadcasted
        parameters.

    Notes
    -----
    The function assumes the existence of `parameters_to_broadcast_recordset`,
    `replace_remote_with_parameters_in_recordset`, `log`, and `time.sleep`
    functions/utilities, as well as `MessageType`, `ConfigsRecord`, `Code`, and `DEBUG`
    constants. It also relies on the `Driver` interface for message handling.
    """
    # Message name
    msg_str = "broadcastins"
    # Create one message per node from one single recordset
    messages = []
    recordset = parameters_to_broadcast_recordset(
        parameters=parameters, keep_input=True
    )
    # Add Status to the recordset
    recordset.configs_records[f"{msg_str}.status"] = ConfigsRecord(
        {"code": int(Code.OK.value), "message": "Broadcasting parameters"}
    )
    # Add S3 configuration to the recordset
    recordset.configs_records[f"{msg_str}.s3_comm_config"] = ConfigsRecord(
        {
            "endpoint_id": "server",
            "file_name": "parameters",
            "current_round": str(current_round),
        }
    )
    # Translating the message and uploading the parameters to S3 if asked to
    fake_message = replace_remote_with_parameters_in_recordset(
        remote_uploader_downloader=remote_uploader_downloader,
        # Create a fake message to upload to S3
        outgoing_message=driver.create_message(
            content=recordset,
            message_type=MessageType.QUERY,
            dst_node_id=node_ids[0],
            group_id=str(current_round),
            ttl=DEFAULT_TTL,
        ),
        use_s3_comm=use_s3_comm,
        msg_str=msg_str,
    )
    # Replacing recordset with the empty one
    recordset = fake_message.content
    # Send the message to all nodes
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
        all_replies += replies
        if len(all_replies) == len(message_ids):
            break
        time.sleep(3)
    # Filter correct results
    all_broadcastres = [msg.content for msg in all_replies if msg.has_content()]
    # Elaborate results
    for message_res in all_broadcastres:
        assert "broadcast" in message_res.configs_records, "Broadcast key not found"
        assert (
            "status" in message_res.configs_records["broadcast"]
        ), "Status key not found"
        if message_res.configs_records["broadcast"]["status"] != "OK":
            raise ValueError("Broadcast failed")
    log(DEBUG, f"Received {len(all_broadcastres)} results")
