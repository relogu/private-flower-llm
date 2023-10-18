"""Large-scale Flower server to solve Ray issue when submitting too many clients."""


import concurrent.futures
from logging import DEBUG, INFO
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import (
    FitResultsAndFailures,
    Server,
    _handle_finished_future_after_fit,
    fit_client,
)

from pollen_worker.utils import chunks_idx


class LargeScaleServer(Server):
    """Large-scale Flower server."""

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

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
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


# NOTE: This is what changes compared to the standard server object
def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    finished_fs: List[concurrent.futures.Future] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        indices = chunks_idx(
            range(len(client_instructions)), (len(client_instructions) // 100) + 1
        )
        for start_idx, end_idx in indices:
            submitted_fs = {
                executor.submit(fit_client, client_proxy, ins, timeout)
                for client_proxy, ins in client_instructions[start_idx:end_idx]
            }
            tmp_finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )
            finished_fs.extend(tmp_finished_fs)

    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return results, failures
