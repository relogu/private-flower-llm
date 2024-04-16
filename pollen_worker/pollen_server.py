"""Pollen server."""

import concurrent.futures
from dataclasses import dataclass
import sys
import timeit
from collections.abc import Callable, Generator
from logging import DEBUG, ERROR, INFO
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from flwr.client import Client
from flwr.common import (
    DisconnectRes,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
    Code,
)
from flwr.common.logger import log
from flwr.common.typing import GetPropertiesIns, Properties
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import evaluate_clients, fit_client
from flwr.server.strategy import FedAvg

from pollen_worker.placements import (
    get_placement_fn,
    get_pollen_models,
    skim_clients_training_stats,
)
from pollen_worker.pollen_client_manager import PollenClientManager
from pollen_worker.pollen_utils import get_table_from_pyarrow_buffer
from pollen_worker.resources_manager import Node
from pollen_worker.virtual_client import VirtualClient

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[tuple[ClientProxy, FitRes] | BaseException],
]
EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[tuple[ClientProxy, EvaluateRes] | BaseException],
]
ReconnectResultsAndFailures = tuple[
    list[tuple[ClientProxy, DisconnectRes]],
    list[tuple[ClientProxy, DisconnectRes] | BaseException],
]

GetPropResultsAndFailures = tuple[
    list[tuple[ClientProxy, Node]],
    list[tuple[ClientProxy, Node] | BaseException],
]


class TooManyFailuresError(Exception):
    """Exception raised when a client is dropped out of the tree."""


class IntentionalClientDropoutError(Exception):
    """Exception raised when a client is dropped out of the tree."""


@dataclass
class AuxiliaryFitClientsResults:
    """Dataclass for the auxiliary results of the fit_clients function."""

    pollen_models: dict[str, Any] | None
    correction_tables: dict[str, pa.Table] | None
    fit_pollen_models_time: float


class PollenServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: PollenClientManager,
        cids: dict[str | int, int],
        client_fn: Callable[[int], Client],
        strategy: FedAvg | None = None,
        placement_policy: str = "rr",
        saving_path: Path | None = None,
        history: History | None = None,
        num_nodes: int = 1,
    ) -> None:
        self.start_up_time = timeit.default_timer()
        self._client_manager: PollenClientManager = client_manager  # type: ignore[reportIncompatibleVariableOverride]
        self.cids = cids
        self.client_fn = client_fn
        self.placement_policy = placement_policy
        self.placement_fn = get_placement_fn(self.placement_policy)
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: FedAvg = strategy if strategy is not None else FedAvg()
        _check_strategy_for_pollen(self.strategy)
        self.on_fit_config: Callable[[int], dict[str, Scalar]] = (
            conf_fn
            if (conf_fn := self.strategy.on_fit_config_fn) is not None
            else lambda _: {}
        )
        self.max_workers: int | None = None
        self.nodes_dict: dict[str, tuple[ClientProxy, Node]] = {}
        if saving_path is None:
            saving_path = Path.cwd()
        self.saving_path = saving_path
        self.clients_training_stats: pa.Table | None = None
        self.history = history
        self.num_nodes = num_nodes
        self.pollen_models: dict[str, Any] | None = None
        self.correction_tables: dict[str, pa.Table] | None = None
        self.ignore_failed_rounds = False
        self.accept_failures_cnt = 0
        self.ignore_failed_rounds = False
        self.print_failures = True
        self.print_intentional_failures = True

    def set_max_workers(self, max_workers: int | None) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        strategy: FedAvg,  # type: ignore[override]
    ) -> None:
        """Replace server strategy."""
        self.strategy = strategy  # type: ignore[reportIncompatibleVariableOverride]

    def client_manager(self) -> PollenClientManager:
        """Return PollenClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: float | None) -> History:
        """Run federated averaging for a number of rounds."""
        server_system_metrics: dict[str, Scalar] = {}
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
        server_system_metrics["server/start_up_time"] = (
            timeit.default_timer() - self.start_up_time
        )
        history.add_metrics_centralized(server_round=0, metrics=server_system_metrics)
        server_system_metrics = {}
        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()
        for current_round in range(1, num_rounds + 1):
            start_time_round = timeit.default_timer()
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
            check_nm_time_round = timeit.default_timer() - start_time_round
            server_system_metrics["server/check_nm_round_time"] = check_nm_time_round
            tmp_timer = timeit.default_timer()
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
            fit_round = timeit.default_timer() - tmp_timer
            server_system_metrics["server/fit_round_time"] = fit_round

            tmp_timer = timeit.default_timer()
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
            evaluate = timeit.default_timer() - tmp_timer
            server_system_metrics["server/evaluate_time"] = evaluate

            tmp_timer = timeit.default_timer()
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
            elapsed_time_round = timeit.default_timer() - start_time_round
            server_system_metrics["server/round_time"] = elapsed_time_round
            evaluate_round_time = timeit.default_timer() - tmp_timer
            server_system_metrics["server/evaluate_round_time"] = evaluate_round_time
            history.add_metrics_centralized(
                server_round=current_round, metrics=server_system_metrics
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
        timeout: float | None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
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
        aggregated_result: tuple[
            float | None,
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(  # type: ignore[override, reportIncompatibleMethodOverride]
        self,
        server_round: int,
        timeout: float | None,
    ) -> (
        None
        | tuple[
            Parameters | None,
            dict[str, Scalar],
            tuple[
                list[tuple[ClientProxy, dict[str, Scalar], Status, int]],
                list[tuple[ClientProxy, FitRes] | BaseException],
                list[BaseException],
            ],
        ]
    ):
        """Perform a single round of federated averaging."""
        start_time = timeit.default_timer()
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
        node_assignments: list[tuple[ClientProxy, dict[str, str]]] = self.placement_fn(
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
            node_instructions.append((
                client_proxy,
                FitIns(self.parameters, node_fit_config),
            ))
        fit_config_time = timeit.default_timer() - start_time

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

        # Using a generator limits us in failure/metrics accumulation
        # The output params are not used in the aggregation
        # They are merely populated by the processing of the generator
        failures: list[tuple[ClientProxy, FitRes] | BaseException] = []

        # Accumulate IntentionalClientDropoutError exceptions
        intentional_failures: list[BaseException] = []

        # Accumulate metrics from all NodeManagers
        metrics_accumulator: list[
            tuple[ClientProxy, dict[str, Scalar], Status, int]
        ] = []

        # Send fit instructions to all NodeManagers participating in this round
        start_time = timeit.default_timer()
        auxiliary_fit_results = AuxiliaryFitClientsResults(
            pollen_models=None,
            correction_tables=None,
            fit_pollen_models_time=0.0,
        )
        (results_futures) = pollen_fit_clients(
            node_instructions=node_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            clients_stats=self.clients_training_stats,
            batch_size=int(self.on_fit_config(server_round)["batch_size"]),
            cids=self.cids,
            placement_policy=self.placement_policy,
            auxiliary_fit_results=auxiliary_fit_results,
        )
        # Wraps the results to discriminate between success and failure
        results_and_failures = (
            _handle_finished_future_after_fit_async(future)
            for future in results_futures
        )
        # Return a closure to pre-process success and failure of the NodeManagers
        handle_success_and_failure = get_handle_success_and_failure(
            metrics_accumulator,
            failures,
            intentional_failures,
            accept_failures_cnt=self.accept_failures_cnt,
        )
        # Wraps pre-processed successes and failures
        results_and_failures = (
            handle_success_and_failure(result) for result in results_and_failures
        )
        # Filter out the successful results and cast them as a generator
        results = (result for success, result in results_and_failures if success)
        complete_results = cast(
            Generator[tuple[ClientProxy, FitRes], None, None], results
        )
        # Collect and aggregate training results asynchronously
        extraction_time = 0.0
        aggregation_time = 0.0
        try:
            # Aggregate training results
            # TODO: Measure aggregation time
            aggregated_result: tuple[
                Parameters | None,
                dict[str, Scalar],
            ] = self.strategy.aggregate_fit(
                server_round,
                cast(list[tuple[ClientProxy, FitRes]], complete_results),
                failures,
            )

            # Collect statistics that Pollen uses from the FitRes of the NodeManagers
            start_time = timeit.default_timer()
            received_clients_training_stats = []
            for metrics in metrics_accumulator:
                tmp_clients_training_stats = metrics[1].pop("stats", None)
                if tmp_clients_training_stats is not None:
                    received_clients_training_stats.append(
                        get_table_from_pyarrow_buffer(
                            cast(pa.Buffer, tmp_clients_training_stats)
                        )
                    )
            extraction_time = timeit.default_timer() - start_time
        except TooManyFailuresError as e:
            if self.ignore_failed_rounds:
                log(
                    ERROR,
                    """Ignoring failed round %s: %s,
                    there are %s failures: %s,
                    there are %s intentional failures: %s""",
                    server_round,
                    e,
                    len(failures),
                    failures if self.print_failures else [],
                    len(intentional_failures),
                    intentional_failures if self.print_intentional_failures else [],
                )
                return None
            else:
                raise
        # NOTE: Now, this time includes the aggregation time
        server_fit_time = timeit.default_timer() - start_time

        # Collect the new statistics relative to the clients' training times
        start_time = timeit.default_timer()
        if self.clients_training_stats is None:
            # Create the table from scratch
            self.clients_training_stats = pa.concat_tables(
                received_clients_training_stats
            )
        else:
            # Skim the `clients_training_stats` to keep just the average per n_samples
            self.clients_training_stats = skim_clients_training_stats(
                self.clients_training_stats
            )
            # Append to the global statistics
            self.clients_training_stats = pa.concat_tables(
                [self.clients_training_stats] + received_clients_training_stats
            )
        concatenation_time = timeit.default_timer() - start_time

        # NOTE: We must assign this only after we can the first iter over the results
        # generator
        self.correction_tables = auxiliary_fit_results.correction_tables
        self.pollen_models = auxiliary_fit_results.pollen_models

        # Return the aggregated results
        parameters_aggregated, metrics_aggregated = aggregated_result
        metrics_aggregated = metrics_aggregated | {
            "server/fit_config_time": fit_config_time,
            "server/fit_clients_time": server_fit_time,
            "server/stats_extraction_time": extraction_time,
            "server/stats_concatenation_time": concatenation_time,
            "server/aggregate_fit_time": aggregation_time,
            "server/fit_pollen_models_time": (
                auxiliary_fit_results.fit_pollen_models_time
            ),
        }
        return (
            parameters_aggregated,
            metrics_aggregated,
            (metrics_accumulator, failures, intentional_failures),
        )


# NEW FUNCTIONS ###############################################################


def pollen_fit_clients(
    node_instructions: list[tuple[ClientProxy, FitIns]],
    max_workers: int | None,
    timeout: float | None,
    auxiliary_fit_results: AuxiliaryFitClientsResults,
    cids: dict[str | int, int],
    clients_stats: pa.Table | None = None,
    batch_size: int = 1,
    placement_policy: str = "rr",
) -> Generator[concurrent.futures.Future[tuple[ClientProxy, FitRes]], Any, None]:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in node_instructions
        }
        # Fit the Pollen models and collect their parameters and correction tables in
        # the `auxiliary_fit_results` variables
        start_time = timeit.default_timer()
        auxiliary_fit_results.pollen_models, auxiliary_fit_results.correction_tables = (
            get_pollen_models(
                placement_policy=placement_policy,
                clients_stats=clients_stats,
                batch_size=batch_size,
                cids=cids,
                server_round=int(node_instructions[0][1].config["server_round"]),
            )
        )
        auxiliary_fit_results.fit_pollen_models_time = (
            timeit.default_timer() - start_time
        )
        # Constructing the generator of training results
        while submitted_fs:
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in finished_fs:
                submitted_fs.remove(future)
                yield future


def get_nodes_properties(
    node_managers: dict[str, ClientProxy],
    max_workers: int | None,
    timeout: float | None = None,
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
    results: list[tuple[ClientProxy, Node]] = []
    failures: list[tuple[ClientProxy, Node] | BaseException] = []
    for future in finished_fs:
        _handle_finished_future_after_get_properties(
            future=future, results=results, failures=failures
        )
    return results, failures


def get_properties_client(
    client: ClientProxy, timeout: float | None
) -> tuple[ClientProxy, Node]:
    """Get properties from a Node."""
    ins = GetPropertiesIns(config={})
    node_properties_res = client.get_properties(ins=ins, timeout=timeout)
    node_properties: Properties = node_properties_res.properties
    log(
        DEBUG,
        "NodeManger properties received from %s: %s",
        client,
        node_properties,
    )
    return client, Node.from_str(str(node_properties["node"]))


def _handle_finished_future_after_get_properties(
    future: concurrent.futures.Future,
    results: list[tuple[ClientProxy, Node]],
    failures: list[tuple[ClientProxy, Node] | BaseException],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, Node] = future.result()
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
    old_connected_node_managers_cid: list[str],
    new_connected_node_managers_cid: list[str],
) -> tuple[list[str], list[str]]:
    dropped: list[str] = []
    new: list[str] = []
    for old_cid in old_connected_node_managers_cid:
        if old_cid not in set(new_connected_node_managers_cid):
            dropped.append(old_cid)
    for new_cid in new_connected_node_managers_cid:
        if new_cid not in set(old_connected_node_managers_cid):
            new.append(new_cid)
    return dropped, new


def _handle_finished_future_after_fit_async(
    future: concurrent.futures.Future,
) -> (
    tuple[Literal[True], tuple[ClientProxy, FitRes]]
    | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException]
):
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        return (False, failure)

    # Successfully received a result from a client
    result: tuple[ClientProxy, FitRes] = future.result()
    _, res = result
    # Check result status code
    if res.status.code == Code.OK:
        return (True, result)

    # Not successful, client returned a result where the status code is not OK
    return (False, result)


def get_handle_success_and_failure(
    metrics_accumulator: list[tuple[ClientProxy, dict[str, Scalar], Status, int]],
    failures: list[tuple[ClientProxy, FitRes] | BaseException],
    intentional_failures: list[BaseException],
    accept_failures_cnt: int | None,
) -> Callable[
    [
        tuple[Literal[True], tuple[ClientProxy, FitRes]]
        | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException]
    ],
    tuple[Literal[True], tuple[ClientProxy, FitRes]]
    | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException],
]:
    """Closure to generate a function which handles client success and failure.

    The function distinguishes between intentional and unintentional failures.
    It enforces constraints on the number of unintentional failures.
    It stores results and failures in the respective lists.

    Parameters
    ----------
    metrics_accumulator : List[Tuple[ClientProxy, Dict[str, Scalar], Status, int]]
        The list where the metrics are accumulated.
    failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        The list where the failures are accumulated.
    intentional_failures : List[BaseException]
        The list where the intentional failures are accumulated.
    accept_failures_cnt : int | None
        The maximum number of unintentional failures to accept.

    Returns
    -------
    handle_success_and_failure : Callable[
        [
            Tuple[Literal[True], Tuple[ClientProxy, FitRes]]
            | Tuple[Literal[False], Tuple[ClientProxy, FitRes] | BaseException]
        ],
        Tuple[Literal[True], Tuple[ClientProxy, FitRes]]
        | Tuple[Literal[False], Tuple[ClientProxy, FitRes] | BaseException],
    ]
        The function which handles client success and failure while saving the outputs.
    """

    def handle_success_and_failure(
        result: (
            tuple[Literal[True], tuple[ClientProxy, FitRes]]
            | tuple[Literal[False], (tuple[ClientProxy, FitRes] | BaseException)]
        ),
    ) -> (
        tuple[Literal[True], tuple[ClientProxy, FitRes]]
        | tuple[Literal[False], (tuple[ClientProxy, FitRes] | BaseException)]
    ):
        cnt_failures = 0

        match result:
            case (True, res):
                cast_res = cast(tuple[ClientProxy, FitRes], res)
                client_proxy, fit_res = cast_res
                metrics_accumulator.append((
                    client_proxy,
                    fit_res.metrics,
                    fit_res.status,
                    fit_res.num_examples,
                ))
                return (True, cast_res)
            case (False, res) if isinstance(res, IntentionalClientDropoutError):
                intentional_failures.append(res)
                return (False, res)
            case (False, res):
                cast_failure_res = cast(tuple[ClientProxy, FitRes] | BaseException, res)
                if isinstance(cast_failure_res, BaseException):
                    log(ERROR, "Unintentional failure.", exc_info=cast_failure_res)
                cnt_failures += 1
                if (
                    accept_failures_cnt is not None
                    and cnt_failures > accept_failures_cnt
                ):
                    raise TooManyFailuresError(f"""Unintentional failures passed
                        the maximum: {accept_failures_cnt}""")
                failures.append(cast_failure_res)
                return (False, cast_failure_res)
        return result

    return handle_success_and_failure
