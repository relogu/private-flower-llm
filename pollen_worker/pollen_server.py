"""Pollen server."""

import concurrent.futures
import os
import sys
import timeit
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import pyarrow as pa
import pyarrow.parquet as pq
from flwr.client import ClientLike
from flwr.common import DisconnectRes, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.typing import GetPropertiesIns, Properties
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import (
    _handle_finished_future_after_fit,
    evaluate_clients,
    fit_client,
)
from flwr.server.strategy import FedAvg

from pollen_worker.placements import get_placement_fn, get_pollen_models
from pollen_worker.pollen_client_manager import PollenClientManager
from pollen_worker.pollen_utils import get_table_from_pyarrow_buffer
from pollen_worker.resources_manager import Node
from pollen_worker.virtual_client import VirtualClient

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, DisconnectRes]],
    List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]],
]

GetPropResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Node]],
    List[Union[Tuple[ClientProxy, Node], BaseException]],
]


class PollenServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: PollenClientManager,
        cids: Dict[Union[str, int], int],
        client_fn: Callable[[int], ClientLike],
        strategy: Optional[FedAvg] = None,
        placement_policy: str = "rr",
        saving_path: Optional[Path] = None,
        history: Optional[History] = None,
        num_nodes: int = 1,
    ) -> None:
        self.start_up_time = timeit.default_timer()
        self._client_manager: PollenClientManager = client_manager
        self.cids = cids
        self.client_fn = client_fn
        self.placement_policy = placement_policy
        self.placement_fn = get_placement_fn(self.placement_policy)
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: FedAvg = strategy if strategy is not None else FedAvg()
        _check_strategy_for_pollen(self.strategy)
        self.on_fit_config: Callable[[int], Dict[str, Scalar]] = (
            conf_fn
            if (conf_fn := self.strategy.on_fit_config_fn) is not None
            else lambda _: {}
        )
        self.max_workers: Optional[int] = None
        self.nodes_dict: Dict[str, Tuple[ClientProxy, Node]] = {}
        if saving_path is None:
            saving_path = Path(os.getcwd())
        self.saving_path = saving_path
        self.clients_training_stats: Optional[pa.Table] = None
        self.history = history
        self.num_nodes = num_nodes
        self.pollen_models: Optional[Dict[str, Any]] = None
        self.correction_tables: Optional[Dict[str, pa.Table]] = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(  # type: ignore[override]
        self,
        strategy: FedAvg,
    ) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> PollenClientManager:
        """Return PollenClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        log(INFO, "Initializing Pollen simulation")
        history = self.history if self.history is not None else History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # NOTE: Register VirtualClients to the PollenClientManager
        self._client_manager.clients = {
            str(i): cast(ClientProxy, VirtualClient(name="", cid=str(k)))
            for i, (k, _) in enumerate(self.cids.items())
        }
        # Waiting for at least one node to connect
        log(INFO, "Waiting for at least one node to connect")
        self._client_manager.wait_for_node_managers(self.num_nodes)
        # Collect nodes' properties
        # NOTE: Ideally, we want to get here the info about the concurrency
        # per hardware accelerator because everything from the server-side
        # has been launched and running, e.g. centralised evaluation (on GPU).
        log(
            DEBUG,
            "Asking for nodes properties to %s NodeManagers",
            self._client_manager.node_managers,
        )
        results, failures = get_nodes_properties(
            node_managers=self._client_manager.node_managers,
            max_workers=self.max_workers,
        )
        log(
            INFO,
            "Get nodes properties: there are %s results and %s failures",
            len(results),
            len(failures),
        )
        # This is a dictionary of the form {"node_id": Node}
        self.nodes_dict = {
            client_proxy.cid: (client_proxy, node) for client_proxy, node in results
        }
        log(
            INFO,
            "Connected node managers: %s",
            self.nodes_dict,
        )
        log(
            INFO,
            "Start-up time for the server is %s",
            timeit.default_timer() - self.start_up_time,
        )
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            # Check for changes in connected NodeManagers
            dropped, new = _check_connected_node_managers(
                old_connected_node_managers_cid=[k for k, _ in self.nodes_dict.items()],
                new_connected_node_managers_cid=[
                    k for k, _ in self._client_manager.node_managers.items()
                ],
            )
            if len(dropped) > 0:
                # Handle dropped NodeManagers
                [self.nodes_dict.pop(k) for k in dropped]
            if len(new) > 0:
                # Handle newly added NodeManagers
                results, failures = get_nodes_properties(
                    node_managers={
                        k: self._client_manager.node_managers[k] for k in new
                    },
                    max_workers=self.max_workers,
                )
                log(
                    INFO,
                    "Get nodes properties: there are %s results and %s failures",
                    len(results),
                    len(failures),
                )
                new_nodes_dict = {
                    client_proxy.cid: (client_proxy, node)
                    for client_proxy, node in results
                }
                self.nodes_dict.update(new_nodes_dict)

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Save the statistics to a parquet file
        if self.clients_training_stats is not None:
            pq.write_table(
                self.clients_training_stats,
                str(self.saving_path / "clients_training_stats.parquet"),
            )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Translate `client_instruction` to `node_instructions`
        node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = self.placement_fn(
            sampled_virtual_cids=[
                (int(client.cid), self.cids[int(client.cid)])
                for client, _ in client_instructions
            ],
            nodes_dict=self.nodes_dict,
            batch_size=self.on_fit_config(server_round)["batch_size"],
            cids=self.cids,
            pollen_models=self.pollen_models,
            clients_stats=self.clients_training_stats,
            correction_tables=self.correction_tables,
            verbose=False,
        )
        # log(
        #     DEBUG,
        #     "Node assignments for fit_round %s: %s",
        #     server_round,
        #     node_assignments,
        # )
        node_instructions = []
        for client_proxy, device_assignment in node_assignments:
            # Get the `fit_config` for the virtual clients
            node_fit_config = self.on_fit_config(server_round)

            # NOTE: This key is used only when the training policy of workers
            # is not `sequential`, and for setting the `num_workers` parameter
            # in the `DataLoader`
            if "server_round" not in node_fit_config:
                node_fit_config["server_round"] = server_round
            if "n_workers" not in node_fit_config:
                node_fit_config["n_workers"] = 1

            # Assign `cids` to NodeManagers' devices
            node_fit_config.update(device_assignment)

            # Append instruction
            node_instructions.append(
                (client_proxy, FitIns(self.parameters, node_fit_config))
            )

        # log(
        #     DEBUG,
        #     "Node instructions for fit_round %s: %s",
        #     server_round,
        #     node_instructions,
        # )

        log(
            DEBUG,
            "fit_round %s: sending instructions to %s NodeManagers",
            server_round,
            len(node_instructions),
        )

        # Collect `fit` results from all NodeManagers participating in this round
        (
            (results, failures),
            self.pollen_models,
            self.correction_tables,
        ) = pollen_fit_clients(
            client_instructions=node_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            clients_stats=self.clients_training_stats,
            batch_size=int(self.on_fit_config(server_round)["batch_size"]),
            cids=self.cids,
            placement_policy=self.placement_policy,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Collect statistics that Pollen uses from the FitRes of the NodeManagers
        received_clients_training_stats = []
        for _client, fit_res in results:
            tmp_clients_training_stats = fit_res.metrics.pop("stats")
            received_clients_training_stats.append(
                get_table_from_pyarrow_buffer(
                    cast(pa.Buffer, tmp_clients_training_stats)
                )
            )

        # Collect the new statistics and append to the global statistics
        if self.clients_training_stats is None:
            self.clients_training_stats = pa.concat_tables(
                received_clients_training_stats
            )
        else:
            self.clients_training_stats = pa.concat_tables(
                [self.clients_training_stats] + received_clients_training_stats
            )
            # self.clients_training_stats = pa.concat_tables(
            #     received_clients_training_stats
            # )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)


####################### NEW FUNCTIONS #######################


def pollen_fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    cids: Dict[Union[str, int], int],
    clients_stats: Optional[pa.Table] = None,
    batch_size: int = 1,
    placement_policy: str = "rr",
) -> Tuple[
    FitResultsAndFailures, Optional[Dict[str, Any]], Optional[Dict[str, pa.Table]]
]:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        pollen_models, correction_tables = get_pollen_models(
            placement_policy=placement_policy,
            clients_stats=clients_stats,
            batch_size=batch_size,
            cids=cids,
            server_round=int(client_instructions[0][1].config["server_round"]),
        )
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return (results, failures), pollen_models, correction_tables


def get_nodes_properties(
    node_managers: Dict[str, ClientProxy],
    max_workers: Optional[int],
    timeout: Optional[float] = None,
) -> GetPropResultsAndFailures:
    """Get the properties of all nodes in the cluster."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(get_properties_client, client_proxy, timeout)
            for node_id, client_proxy in node_managers.items()
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=timeout,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, Node]] = []
    failures: List[Union[Tuple[ClientProxy, Node], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_get_properties(
            future=future, results=results, failures=failures
        )
    return results, failures


def get_properties_client(
    client: ClientProxy, timeout: Optional[float]
) -> Tuple[ClientProxy, Node]:
    """Get properties froma a Node."""
    ins = GetPropertiesIns(config={})
    node_properties_res = client.get_properties(ins=ins, timeout=timeout)
    node_properties: Properties = node_properties_res.properties
    log(
        DEBUG,
        "node properties received from %s: %s",
        client,
        node_properties,
    )
    return client, Node.from_str(str(node_properties["node"]))


def _handle_finished_future_after_get_properties(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, Node]],
    failures: List[Union[Tuple[ClientProxy, Node], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, Node] = future.result()
    results.append(result)


def _check_strategy_for_pollen(
    strategy: FedAvg,
) -> bool:
    if strategy.on_fit_config_fn is None:
        log(
            ERROR,
            "The strategy, %s, passed to the `PollenServer` doesn't"
            " have a proper `on_fit_config_fn` attribute. The user"
            " must define such method as type `Callable[[int], Dict]`"
            "Currently, `on_fit_config_fn` is %s.",
            strategy,
            strategy.on_fit_config_fn,
        )
        sys.exit(0)
    if "batch_size" not in strategy.on_fit_config_fn(0) or not isinstance(
        strategy.on_fit_config_fn(0)["batch_size"], int
    ):
        log(
            ERROR,
            "The `on_fit_config_fn` function of the strategy passed"
            " to the `PollenServer` must have a proper `batch_size`"
            " key with an `int` value. The call"
            " `strategy.on_fit_config_fn(0)` returned %s instead",
            strategy.on_fit_config_fn(0),
        )
        sys.exit(0)
    return True


def _check_connected_node_managers(
    old_connected_node_managers_cid: List[str],
    new_connected_node_managers_cid: List[str],
) -> Tuple[List[str], List[str]]:
    dropped: List[str] = []
    new: List[str] = []
    for old_cid in old_connected_node_managers_cid:
        if old_cid not in set(new_connected_node_managers_cid):
            dropped.append(old_cid)
    for new_cid in new_connected_node_managers_cid:
        if new_cid not in set(old_connected_node_managers_cid):
            new.append(new_cid)
    return dropped, new
