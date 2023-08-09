# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower Pollen server."""

import sys
import concurrent.futures
import time
import timeit
from copy import copy, deepcopy
from logging import DEBUG, INFO, ERROR
from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)
from flwr.common.logger import log
from flwr.common.typing import (
    GetParametersIns,
    GetPropertiesIns,
    GetPropertiesRes,
    Properties,
)
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

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

from placements import get_placement_fn
from pollen_client_manager import PollenClientManager
from utils import invert_many_to_one_dictionary


class PollenServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: PollenClientManager,
        cids: Dict[str, int],
        client_fn: Callable[[int, str], ClientProxy],
        strategy: Optional[Strategy] = None,
        placement_policy: str = "rr",
    ) -> None:
        self._client_manager: PollenClientManager = client_manager
        self.cids = cids
        self.client_fn = client_fn
        self.placement_policy = placement_policy
        self.placement_fn = get_placement_fn(self.placement_policy)
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        check_strategy_for_pollen(self.strategy)
        self.on_fit_config: Callable[[int], Dict[str, Scalar]] = self.strategy.on_fit_config_fn
        self.max_workers: Optional[int] = None
        self.worker_to_resource: Dict[str, Tuple[str, str, int, int]] = {}

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> PollenClientManager:
        """Return PollenClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

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

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        # NOTE: Register VirtualClients to the PollenClientManager
        self._client_manager.clients = {
            str(i): self.client_fn(k, "") for i, (k, _) in enumerate(self.cids.items())
        }
        # TODO/FIXME: Hardcoded to allow all the workers to connect
        time.sleep(10)
        # Set the number of connected NodeManagers
        connected_node_managers: Dict[str, ClientProxy] = copy(
            self._client_manager.node_managers
        )
        # TODO/FIXME: Collect and allocate available resources
        self.worker_to_resource = assign_worker_to_resource(connected_node_managers)
        for current_round in range(1, num_rounds + 1):
            # Check for changes in connected NodeManagers
            dropped, new, connected_node_managers = check_connected_node_managers(
                old_connected_node_managers=connected_node_managers,
                new_connected_node_managers=self._client_manager.node_managers,
            )
            if bool(dropped):
                # Handle dropped NodeManagers
                log(DEBUG, f"{len(dropped)} NodeManagers have been dropped")
                [self.worker_to_resource.pop(k) for k, _ in dropped.items()]
            if bool(new):
                # TODO: Handle newly added NodeManagers, don't know how to do this rn
                log(DEBUG, f"There are new {len(new)} NodeManagers connected")

            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
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
        lists_cids = self.placement_fn(
            sampled_virtual_cids=[
                (int(client.cid), self.cids[client.cid])
                for client, _ in client_instructions
            ],
            workers_dict=self._client_manager.node_managers,
            batch_size=self.on_fit_config(server_round)["batch_size"],
            verbose=False,
        )
        node_instructions = []
        for node_id, node in self._client_manager.node_managers.items():
            node_fit_config = self.on_fit_config(server_round)
            # NOTE: This key is used only when the training policy of workers
            # is not `sequential`, and for setting the `num_workers` parameter
            # in the `DataLoader`
            if "server_round" not in node_fit_config:
                node_fit_config["server_round"] = server_round
            if "workers_policy" not in node_fit_config:
                node_fit_config["workers_policy"] = "split"
            # TODO/FIXME: Set the level of concurrency
            node_fit_config["concurrency"] = self.worker_to_resource[node_id][3]
            # TODO/FIXME: Assign `cids`
            node_fit_config["cids"] = (
                "[[" + ",".join([str(cid) for cid in lists_cids[node_id]]) + "]]"
            )
            # TODO/FIXME: Assign `gpu_ids`
            node_fit_config["gpu_ids"] = f"[{self.worker_to_resource[node_id][2]}]"
            # Append instruction
            node_instructions.append(
                (node, FitIns(self.parameters, node_fit_config))
            )

        log(
            DEBUG,
            "fit_round %s: sending instructions to %s NodeManagers",
            server_round,
            len(node_instructions),
        )

        # Collect `fit` results from all NodeManagers participating in this round
        results, failures = fit_clients(
            client_instructions=node_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: Tuple[
            Optional[Parameters],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(self, timeout: Optional[float]) -> Parameters:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(ins=ins, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: List[Tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, DisconnectRes]] = []
    failures: List[Union[Tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
    )
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
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
    return results, failures


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: List[Tuple[ClientProxy, EvaluateRes]],
    failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""

    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: Tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


####################### NEW STUFF #######################


def check_strategy_for_pollen(
    strategy: Strategy,
) -> bool:
    if not isinstance(
        strategy.on_fit_config_fn,
        Callable[[int], Dict],
    ):
        log(
            ERROR,
            "The strategy, %s, passed to the `PollenServer` doesn't have a proper `on_fit_config_fn` attribute."
            "The user must define such method as type `Callable[[int], Dict]`"
            "Currently, `on_fit_config_fn` is %s.",
            strategy, strategy.on_fit_config_fn
        )
        sys.exit(0)
    if "batch_size" not in strategy.on_fit_config_fn(0) or not isinstance(
        strategy.on_fit_config_fn(0)["batch_size"], int
    ):
        log(
            ERROR,
            "The `on_fit_config_fn` function of the strategy passed to the `PollenServer`"
            " must have a proper `batch_size` key with an `int` value."
            "The call `strategy.on_fit_config_fn(0)` returned %s instead",
            strategy.on_fit_config_fn(0),
        )
        sys.exit(0)


def check_connected_node_managers(
    old_connected_node_managers: Dict[str, ClientProxy],
    new_connected_node_managers: Dict[str, ClientProxy],
) -> Tuple[Dict[str, ClientProxy], Dict[str, ClientProxy], Dict[str, ClientProxy]]:
    dropped: Dict[str, ClientProxy] = {}
    new: Dict[str, ClientProxy] = {}
    for key in old_connected_node_managers.keys():
        if key not in new_connected_node_managers:
            dropped[key] = old_connected_node_managers[key]
    for key in new_connected_node_managers.keys():
        if key not in old_connected_node_managers:
            new[key] = new_connected_node_managers[key]
    return dropped, new, new_connected_node_managers


def get_all_workers_properties(
    connected_node_managers: Dict[str, ClientProxy],
) -> Dict[str, Properties]:
    ins = GetPropertiesIns(config={})
    all_workers_properties: Dict[str, GetPropertiesRes] = {}
    for worker_dict in connected_node_managers.items():
        worker_id, worker = worker_dict
        worker_properties = worker.get_properties(ins=ins, timeout=60).properties
        all_workers_properties[worker_id] = worker_properties
    return all_workers_properties


# TODO/FIXME: This might need to change in light of the change of abstraction
def assign_worker_to_resource(
    connected_node_managers: Dict[str, ClientProxy],
) -> Dict[str, Tuple[str, str, int, int]]:
    # Getting nodes' resources information
    worker_node_map: Dict = {}
    node_gpu_n_workers_dict: Dict = {}
    all_workers_properties = get_all_workers_properties(
        connected_node_managers=connected_node_managers
    )
    for worker_id, properties in all_workers_properties.items():
        worker_node_map[worker_id] = properties["node_name"]
        gpu_workers_map = eval(properties["node_gpus"])
        # log(
        #     DEBUG, 'Worker %s in node %s has gpu_worker_map %s',
        #     worker_id, node_name, gpu_workers_map
        # )
        if worker_node_map[worker_id] not in node_gpu_n_workers_dict:
            node_gpu_n_workers_dict[worker_node_map[worker_id]] = gpu_workers_map
    log(
        DEBUG,
        "Built the node to GPUs and number of workers map %s",
        node_gpu_n_workers_dict,
    )
    log(DEBUG, "Built the worker to node map %s", worker_node_map)
    # Assigning workers to resources
    worker_resource_map: Dict[str, Tuple[str, int]] = {}
    node_worker_map = invert_many_to_one_dictionary(worker_node_map)
    for node, workers in node_worker_map.items():
        workers_in_this_node: List[str] = deepcopy(workers)
        concurrency = len(workers_in_this_node)
        # Dict[<gpu_id>, Tuple[<gpu_name>, <n_workers>]]
        node_resources: Dict[str, Tuple[str, int]] = deepcopy(
            node_gpu_n_workers_dict[node]
        )
        while len(workers_in_this_node) > 0:
            for gpu_id, (gpu_name, n_workers) in node_resources.items():
                if n_workers > 0:
                    worker_resource_map[workers_in_this_node.pop(0)] = (
                        node,
                        gpu_name,
                        gpu_id,
                        concurrency,
                    )
                    node_resources[gpu_id] = (gpu_name, n_workers - 1)
                    break
    log(DEBUG, "Assigned workers to resources %s", worker_resource_map)
    return worker_resource_map
