"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""

from itertools import starmap
import pickle
import random
from collections.abc import Callable, Iterable
from logging import DEBUG, WARNING
from pathlib import Path
import time

from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
import numpy as np


def aggregate_cumulative_average(
    results: Iterable[tuple[ClientProxy, FitRes]],
) -> tuple[NDArrays | None, float]:
    """Compute in-place weighted average, lazily and async."""
    # Initialize params,
    # the iterator may not contain anything
    # and we do not want a next+try-except
    params: NDArrays | None = None

    num_total_examples: int = 0  # total number of examples, aggregated over time
    total_aggregated_time: float = 0.0  # total time spent aggregating
    for client_proxy, fit_res in results:
        start_time = time.time()
        log(
            DEBUG,
            f"Started aggregating cid: {client_proxy.cid}",
        )
        # Compute the new total number of samples
        new_total_samples = num_total_examples + fit_res.num_examples

        # Compute scaling factor for the update
        scaling_factor: float = float(fit_res.num_examples) / new_total_samples

        # Generator to multiply the layers by the scaling factor
        # Lazy operation, will be expanded in the zip
        res = (scaling_factor * x for x in parameters_to_ndarrays(fit_res.parameters))

        if params is None:
            # Initialize with the first set of parameters
            params = list(res)
        else:
            # Invert the previous division now
            # that we have updated information on the number of samples
            # Then divide by the new total number
            general_scaling = float(num_total_examples) / new_total_samples

            # Lazy generator for scaled layers
            scaled_params = (general_scaling * x for x in params)

            # Create new parameters
            params = list(starmap(np.add, zip(scaled_params, res, strict=False)))

        # Update total number of examples
        num_total_examples = new_total_samples
        total_aggregated_time += time.time() - start_time
        log(
            DEBUG,
            f"""Aggregated cid: {client_proxy.cid}
                                with samples: {fit_res.num_examples}
                                total samples used: {num_total_examples}
                                time: {time.time() - start_time} """,
        )

    return params, total_aggregated_time


# flake8: noqa: E501
class FedAvgReproducibleSampling(FedAvg):
    """Configurable FedAvgReproducibleSampling strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        seed: int = 1337,
    ) -> None:
        """Federated Averaging strategy with reproducible sampling.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        seed : int, optional
            Seed for reproducibility. Defaults to 1337.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.seed = seed

    def configure_fit(  # type: ignore[reportIncompatibleMethodOverride,override]
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: SimpleClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, _ = self.num_fit_clients(client_manager.num_available())

        if sample_size > len(client_manager.clients):
            log(
                WARNING,
                "sample_size > len(client_manager.clients), to satisfy this"
                " condition, we will sample clients with replacement",
            )

            # Setting seed for reproducibility of client selection
            random.seed(self.seed + server_round)

            # Generate random selection of virtual clients (number of virtual clients per round)
            sampled_virtual_cids = random.choices(
                list(client_manager.clients), k=sample_size
            )
        else:
            # Wait for the minimum number of clients to be available
            client_manager.wait_for(sample_size)

            # Setting seed for reproducibility of client selection
            random.seed(self.seed + server_round)

            # Generate random selection of virtual clients (number of virtual clients per round)
            sampled_virtual_cids = random.sample(
                list(client_manager.clients), sample_size
            )

        # Get the actual clients from the client manager
        clients = [client_manager.clients[cid] for cid in sampled_virtual_cids]

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


# flake8: noqa: E501
class FedAvgRSModel(FedAvgReproducibleSampling):
    """Configurable FedAvgRSModel strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        saving_path: Path | None = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        seed: int = 1337,
        freq: int = 1,
    ) -> None:
        """Federated Averaging strategy.

        It uses reproducible sampling and model saving.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        seed : int, optional
            Seed for reproducibility. Defaults to 1337.
        freq : int, optional
            Frequency of model saving. Defaults to 1.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            seed=seed,
        )
        if saving_path is None:
            saving_path = Path(Path.cwd())
        self.saving_path = saving_path

        self.freq = freq

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        fedavg_result, aggregation_time = aggregate_cumulative_average(results)

        # Return None if no results were aggregated
        if fedavg_result is None:
            return None, {}
        parameters_aggregated = ndarrays_to_parameters(fedavg_result)
        if server_round % self.freq == 0:
            # Save `parameters_aggregated`` to file
            with open(
                self.saving_path / f"parameters_aggregated_{server_round}", "wb"
            ) as f:
                pickle.dump(parameters_aggregated, f)
        # NOTE: This is handled by the server
        # # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated: dict[str, Scalar] = {}
        # if self.fit_metrics_aggregation_fn:
        #     fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")
        metrics_aggregated |= {"server/aggregate_fit_time": aggregation_time}

        return parameters_aggregated, metrics_aggregated
