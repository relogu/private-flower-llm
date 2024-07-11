from logging import DEBUG
import os
from typing import cast
import random
import time
import warnings

import flwr as fl
from flwr.common import (
    Context,
    FitIns,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Code,
    Message,
    MessageType,
    DEFAULT_TTL,
)
from flwr.common.logger import log, update_console_handler
from flwr.common.recordset_compat import fitins_to_recordset, recordset_to_fitres
from flwr.server import Driver, History
from flwr.server.strategy.aggregate import aggregate
from omegaconf import OmegaConf

from flower_llm.conf.base_schema import BaseConfig


# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)
# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# TODO: These don't work -- not sure why
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)

# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    """TODO."""
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("POLLEN_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
    # Load the configuration from the config file
    cfg = cast(BaseConfig, OmegaConf.load(save_path + "/config.yaml"))

    # TODO: Get FL setting parameters from the config file
    # TODO: Create ClientManagers
    # TODO: Strategy dispatcher
    # TODO: Wandb context manager
    # TODO: Init previous server attributes (from __init__)
    # TODO: Import checkpoints
    # TODO: Resume/initialize round
    # TODO: Start FL loop
    # TODO:

    log(DEBUG, "RUNNING!!!!!")

    num_client_nodes_per_round = 2
    sleep_time = 1
    num_rounds = 3
    parameters = ndarrays_to_parameters(get_weights(net=Net()))

    history = History()
    for server_round in range(num_rounds):
        log(DEBUG, f"Commencing server round {server_round + 1}")

        # List of sampled node IDs in this round
        sampled_nodes: list[int] = []

        # The Driver API might not immediately return enough client node IDs, so we
        # loop and wait until enough client nodes are available.
        while True:
            all_node_ids = driver.get_node_ids()

            log(DEBUG, f"Got {len(all_node_ids)} client nodes: {all_node_ids}")
            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_nodes = random.sample(all_node_ids, num_client_nodes_per_round)
                break
            time.sleep(3)

        # Log sampled node IDs
        log(DEBUG, f"Sampled {len(sampled_nodes)} node IDs: {sampled_nodes}")

        # Schedule a task for all sampled nodes
        fit_ins: FitIns = FitIns(parameters=parameters, config={})
        recordset = fitins_to_recordset(fitins=fit_ins, keep_input=True)

        messages = []
        for node_id in sampled_nodes:
            message = driver.create_message(
                content=recordset,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
                group_id=str(server_round),
                ttl=DEFAULT_TTL,
            )
            messages.append(message)

        message_ids = driver.push_messages(messages)
        log(DEBUG, f"Pushed {len(message_ids)} messages: {message_ids}")

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
        all_fitres = [
            recordset_to_fitres(msg.content, keep_input=True)
            for msg in all_replies
            if msg.has_content()
        ]
        log(DEBUG, f"Received {len(all_fitres)} results")

        weights_results: list[tuple[NDArrays, int]] = []
        metrics_results: list[tuple[int, dict]] = []
        for fitres in all_fitres:
            log(
                DEBUG,
                f"num_examples: {fitres.num_examples}, status: {fitres.status.code}",
            )

            # Aggregate only if the status is OK
            if fitres.status.code != Code.OK:
                continue
            weights_results.append(
                (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            )
            metrics_results.append((fitres.num_examples, fitres.metrics))

        if len(weights_results) > 0:
            # Aggregate parameters (FedAvg)
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
            parameters = parameters_aggregated

            # Aggregate metrics
            metrics_aggregated = weighted_average(metrics_results)
            history.add_metrics_distributed_fit(
                server_round=server_round, metrics=metrics_aggregated
            )
            log(DEBUG, "Round ", server_round, " metrics: ", metrics_aggregated)
        else:
            log(
                DEBUG,
                f"Round {server_round} got {len(weights_results)} results. Skipping aggregation...",
            )

        # Slow down the start of the next round
        time.sleep(sleep_time)

    log(DEBUG, "app_fit: losses_distributed %s", str(history.losses_distributed))
    log(
        DEBUG,
        "app_fit: metrics_distributed_fit %s",
        str(history.metrics_distributed_fit),
    )
    log(DEBUG, "app_fit: metrics_distributed %s", str(history.metrics_distributed))
    log(DEBUG, "app_fit: losses_centralized %s", str(history.losses_centralized))
    log(DEBUG, "app_fit: metrics_centralized %s", str(history.metrics_centralized))
