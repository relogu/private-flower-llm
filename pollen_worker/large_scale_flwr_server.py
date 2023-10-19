"""Large-scale Flower server to solve Ray issue when submitting too many clients."""


import concurrent.futures
from logging import DEBUG, INFO, WARNING
from typing import List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import (
    FitResultsAndFailures,
    Server,
    _handle_finished_future_after_fit,
    fit_client,
)
from flwr.server.strategy import FedAvg

from pollen_worker.utils import aggregate, aggregate_inplace, chunks_idx


class LargeScaleServer(Server):
    """Large-scale Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[FedAvg] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: FedAvg = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[Tuple[Optional[Parameters], Metrics, FitResultsAndFailures]]:
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

        # NOTE: This is what changes compared to the standard server object
        # Collect `fit` results from all clients participating in this round
        return fit_clients(
            strategy=self.strategy,
            server_round=server_round,
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )


# NOTE: This is what changes compared to the standard server object
def fit_clients(
    strategy: FedAvg,
    server_round: int,
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> Optional[Tuple[Optional[Parameters], Metrics, FitResultsAndFailures]]:
    """Refine parameters concurrently on all selected clients."""
    finished_fs: List[concurrent.futures.Future] = []
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    tmp_results: List[Tuple[NDArrays, int]] = []
    tmp_metrics: List[Tuple[int, Metrics]] = []
    empty_res_and_fail: Tuple[
        List[Tuple[ClientProxy, FitRes]],
        List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ] = ([], [])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        indices = chunks_idx(
            range(len(client_instructions)), (len(client_instructions) // 100) + 1
        )
        for start_idx, end_idx in indices:
            submitted_fs = {
                executor.submit(fit_client, client_proxy, ins, timeout)
                for client_proxy, ins in client_instructions[start_idx:end_idx]
            }
            # Partial aggregation
            if len(finished_fs) > 0:
                for future in finished_fs:
                    _handle_finished_future_after_fit(
                        future=future, results=results, failures=failures
                    )
                log(
                    DEBUG,
                    "fit_round %s received %s results and %s failures",
                    server_round,
                    len(results),
                    len(failures),
                )
                # Partially aggregate custom metrics if aggregation fn was provided
                if strategy.fit_metrics_aggregation_fn:
                    fit_metrics = [
                        (res.num_examples, res.metrics) for _, res in results
                    ]
                    tmp_metrics.append(
                        (
                            sum([num_examples for num_examples, _ in fit_metrics]),
                            strategy.fit_metrics_aggregation_fn(fit_metrics),
                        )
                    )
                # Partially aggregate parameters
                if results:
                    tmp_results.append(aggregate_inplace(results))
                # Clear lists
                finished_fs.clear()
                results.clear()
                failures.clear()
            # Wait for futures to complete
            tmp_finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )
            finished_fs.extend(tmp_finished_fs)
    # Partial aggregation
    if len(finished_fs) > 0:
        for future in finished_fs:
            _handle_finished_future_after_fit(
                future=future, results=results, failures=failures
            )
        log(
            DEBUG,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        # Partially aggregate custom metrics if aggregation fn was provided
        if strategy.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            tmp_metrics.append(
                (
                    sum([num_examples for num_examples, _ in fit_metrics]),
                    strategy.fit_metrics_aggregation_fn(fit_metrics),
                )
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        # Partially aggregate parameters
        if results:
            tmp_results.append(aggregate_inplace(results))
        # Clear lists
        finished_fs.clear()
        results.clear()
        failures.clear()
    # Aggregate partial aggregations
    parameters_aggregated: Optional[Parameters] = None
    if tmp_results:
        parameters_aggregated = ndarrays_to_parameters(aggregate(tmp_results))
    if strategy.fit_metrics_aggregation_fn:
        metrics_aggregated = strategy.fit_metrics_aggregation_fn(tmp_metrics)

    return parameters_aggregated, metrics_aggregated, empty_res_and_fail
