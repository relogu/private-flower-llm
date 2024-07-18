"""Utility functions for running the main server loop in flwr next."""

from collections.abc import Callable, Generator
from logging import DEBUG
import time
from flwr.common import (
    Parameters,
    log,
    Message,
    DEFAULT_TTL,
    RecordSet,
    FitIns,
    ConfigsRecord,
)
from flwr.common.recordset_compat import (
    fitins_to_recordset,
)
from flwr.common.typing import ConfigsRecordValues
from flwr.server import Driver


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
    gen_ins_function: Callable[[int, int | str], dict[str, ConfigsRecordValues]],
    all_node_ids: list[int],
    current_round: int,
    msg_str: str,
) -> Generator[Message, None, None]:
    # Constructing separate record sets for each client
    record_sets: list[RecordSet] = []
    for cid in sampled_clients:
        record_set = fitins_to_recordset(
            FitIns(
                parameters=Parameters(tensors=[], tensor_type="empty"),
                config={},
            ),
            keep_input=True,
        )
        record_set.configs_records.update(
            {str(cid): ConfigsRecord(gen_ins_function(current_round, cid))}
        )
        record_set.configs_records.update(
            {f"{msg_str}.config": ConfigsRecord({"server_round": current_round})}
        )
        record_sets.append(record_set)

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
    log(DEBUG, "Pushed %s messages: %s", len(messages), message_ids)

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
        log(DEBUG, "Pushed another %s messages: %s", len(messages), message_ids)
        for res in replies:
            yield res


def message_independent(
    driver: Driver,
    message_type: str,
    gen_ins_function: Callable[[int, int | str], dict[str, ConfigsRecordValues]],
    all_node_ids: list[int],
    current_round: int,
    assignment_function: Callable[[int], list[int] | list[str]],
    msg_str: str,
) -> Generator[Message, None, None]:
    """Generate and send messages to nodes for independent processing and yield results.

    This function is designed to operate in a federated learning context where messages
    containing instructions or data are sent to various nodes (clients or servers) for
    processing. It first assigns clients to nodes based on the `assignment_function`,
    then generates messages for each node using the `gen_ins_function` to create the
    content. These messages are sent out via the `driver`, and the function then waits
    for and yields the results as they arrive.

    Parameters
    ----------
    driver : Driver
        The communication driver responsible for message creation, sending, and
        receiving.
    message_type : str
        The type of the message to be sent, defining its purpose or action to be taken
        by the receiver.
    gen_ins_function : Callable[[int, int | str], dict[str, ConfigsRecordValues]]
        A function that generates the instruction set for a message given the current
        round and a client identifier. It returns a dictionary of configuration record
        values.
    all_node_ids : list[int]
        A list of all node identifiers to which messages will be sent.
    current_round : int
        The current round of the federated learning process.
    assignment_function : Callable[[int], list[int] | list[str]]
        A function that assigns client identifiers to a node based on the node
        identifier. It returns a list of client identifiers assigned to the node.
    msg_str : str
        A string prefix used in message creation to identify or categorize the message.

    Yields
    ------
    Generator[Message, None, None]
        A generator that yields the results of the message processing as `Message`
        objects. Each `Message` object represents either a result or an error from the
        processing node.

    Notes
    -----
    The function assumes the existence of a `log` function for logging and a
    `DEFAULT_TTL` constant that defines the time-to-live for messages. It also relies
    on the `Driver` interface for message handling and the `create_merged_recordset`
    function for generating the content of each message.
    """
    messages = []
    for node_id in all_node_ids:
        cids_to_train = assignment_function(node_id)
        if cids_to_train:
            message = driver.create_message(
                content=create_merged_recordset(
                    sampled_clients=cids_to_train,
                    current_round=current_round,
                    gen_ins_function=gen_ins_function,
                    msg_str=msg_str,
                ),
                message_type=message_type,
                dst_node_id=node_id,
                group_id=str(current_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)
    message_ids = list(driver.push_messages(messages))
    log(DEBUG, "Pushed %s messages: %s", len(messages), message_ids)
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
    gen_ins_function: Callable[[int, int | str], dict[str, ConfigsRecordValues]],
    msg_str: str,
) -> RecordSet:
    """Create a merged record set for the sampled clients in a federated learning round.

    This function generates a `RecordSet` object that contains configuration records for
    each sampled client and a main configuration record that includes the server round
    information. It uses a generator function to create individual client configuration
    records based on the current round and client identifiers. These configurations are
    then merged into a single `RecordSet` object, which also includes an empty `FitIns`
    object with no parameters and an empty configuration, intended for initialization
    purposes.

    Parameters
    ----------
    sampled_clients : list[int] | list[str]
        A list of identifiers for the clients sampled in the current federated learning
        round. These identifiers can be either integers or strings.
    current_round : int
        The current federated learning round number.
    gen_ins_function : Callable[[int, int | str], dict[str, ConfigsRecordValues]]
        A generator function that takes the current round and a client identifier as
        inputs and returns a dictionary representing the client's configuration record
        values.
    msg_str : str
        A string used to prefix the main configuration record key, typically indicating
        the type of message or operation being performed.

    Returns
    -------
    RecordSet
        A `RecordSet` object containing the merged configuration records for all sampled
        clients and the main configuration record with the server round information.

    Notes
    -----
    The `RecordSet` object is a custom data structure used to aggregate and manage
    different types of records,such as configurations and inputs for federated
    learning operations. The `ConfigsRecord` and `FitIns` are also custom data
    structures representing configuration records and federated learning instructions,
    respectively.
    """
    configs = {
        str(cid): ConfigsRecord(gen_ins_function(current_round, cid))
        for cid in sampled_clients
    }
    record_set = fitins_to_recordset(
        FitIns(
            parameters=Parameters(tensors=[], tensor_type="empty"),
            config={},
        ),
        keep_input=True,
    )
    record_set.configs_records.update(configs)
    # NOTE: We always need to pass the server round to the main config record
    record_set.configs_records.update(
        {f"{msg_str}.config": ConfigsRecord({"server_round": current_round})}
    )
    return record_set
