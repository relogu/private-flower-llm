"""Federated Averaging with Nestorov Momentum (FedAvgM) strategy.

This aggregation mechanism is used and discussed in several papers:
[Hsu et al., 2019], [Huo et al., 2020]

Papers:
- https://arxiv.org/pdf/1909.06335.pdf
- https://arxiv.org/pdf/2002.02090.pdf
"""

from collections.abc import Callable, Iterable
from logging import INFO, WARNING
from pathlib import Path

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    log,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from pollen_worker.strategy.aggregation import aggregate_cumulative_average
from pollen_worker.strategy.rs_fedavg import FedAvgReproducibleSampling
from pollen_worker.utils import l1_norm


# flake8: noqa: E501
class FedNesterov(FedAvgReproducibleSampling):
    """Configurable FedNesterov strategy implementation."""

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
        server_learning_rate: float = 0.7,  # default DiLoCo value
        server_momentum: float = 0.9,  # default DiLoCo value
        track_norms: bool = True,
        track_inplace_aggregation: bool = False,
    ) -> None:
        """Federated Averaging with Nestorov Momentum strategy.

        It uses reproducible sampling and model saving.

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
        server_learning_rate : float, optional
            Learning rate used by the server-side optimizer. Defaults to 0.7.
        server_momentum: float, optional
            Momentum coefficient used by the server-side optimizer. Defaults to 0.9.
        track_norms: bool, optional
            Flag for tracking the norms of the aggregated updates. Defaults to True.
        track_inplace_aggregation: bool, optional
            Flag for tracking the difference between standard and in-place aggregation. Defaults to False.
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

        # Default to DiLoCo values
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum

        # Avoid translating between parameters and NDArrays every time unnecessarily
        self.ndarray_parameters: NDArrays | None = (
            parameters_to_ndarrays(initial_parameters)
            if initial_parameters is not None
            else None
        )

        log(
            INFO,
            "Using Nesterov Momentum with server_learning_rate=%s and"
            " server_momentum=%s",
            self.server_learning_rate,
            self.server_momentum,
        )
        self.momentum_vector: NDArrays | None = None

        self.track_norms = track_norms
        self.track_inplace_aggregation = track_inplace_aggregation

    def aggregate_fit(
        self,
        server_round: int,
        results: Iterable[tuple[ClientProxy, FitRes]],
        failures: Iterable[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        assert (
            self.ndarray_parameters is not None
        ), "When using server-side optimization, model needs to be initialized."

        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []

        def acc_metrics(
            result: tuple[ClientProxy, FitRes]
        ) -> tuple[ClientProxy, FitRes]:
            _, fit_res = result
            fit_metrics.append((fit_res.num_examples, fit_res.metrics))
            return result

        results = (acc_metrics(result) for result in results)

        results_cached: list[tuple[ClientProxy, FitRes]] = []

        if self.track_inplace_aggregation:
            results_cached = list(results)
            results = (val for val in results_cached)

        fedavg_result = aggregate_cumulative_average(results)

        if fedavg_result is None:
            return None, {}

        pseudo_gradient: NDArrays = [
            x - y for x, y in zip(self.ndarray_parameters, fedavg_result, strict=False)
        ]

        if server_round > 1:
            assert self.momentum_vector, "Momentum should have been created on round 1."

            self.momentum_vector = [
                self.server_momentum * v + w
                for w, v in zip(pseudo_gradient, self.momentum_vector, strict=False)
            ]
        else:  # Round 1
            # Initialize server-side model

            # Initialize momentum vector
            self.momentum_vector = pseudo_gradient

        # Applying Nesterov
        pseudo_gradient = [
            g + self.server_momentum * v
            for g, v in zip(pseudo_gradient, self.momentum_vector, strict=False)
        ]

        # Federated Averaging with Server Momentum
        fedavgm_result = [
            w - self.server_learning_rate * v
            for w, v in zip(self.ndarray_parameters, pseudo_gradient, strict=False)
        ]

        self.ndarray_parameters = fedavgm_result

        parameters_aggregated = ndarrays_to_parameters(fedavgm_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if self.track_norms:
            metrics_aggregated |= {
                "l1_norm_pseudo_gradient": l1_norm(pseudo_gradient),
                "l1_norm_momentum_vector": l1_norm(self.momentum_vector),
                "l1_norm_model": l1_norm(fedavgm_result),
                "l1_norm_fedavg_result": l1_norm(fedavg_result),
            }
            log(
                INFO,
                "Nesterov Momentum: l1_norm(pseudo_gradient)=%s,"
                " l1_norm(self.momentum_vector)=%s, l1_norm(model)=%s,"
                " l1_norm(fedavg_result)=%s",
                l1_norm(pseudo_gradient),
                l1_norm(self.momentum_vector),
                l1_norm(fedavgm_result),
                l1_norm(fedavg_result),
            )

        if self.track_inplace_aggregation:
            normal_result = aggregate([
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results_cached
            ])
            layer_by_layer_diff = 0.0
            for x, y in zip(normal_result, fedavg_result, strict=False):
                layer_by_layer_diff += l1_norm([x - y])
            metrics_aggregated |= {
                "l1_norm_fedavg_gap": layer_by_layer_diff,
            }
            log(
                INFO,
                "Inplace aggregation gap: l1_norm(normal_result -"
                " fedavg_result)=%s, len_results: %s",
                layer_by_layer_diff,
                len(results_cached),
            )

        return parameters_aggregated, metrics_aggregated
