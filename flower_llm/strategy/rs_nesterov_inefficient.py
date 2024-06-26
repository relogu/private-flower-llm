"""Federated Averaging with Nestorov Momentum (FedAvgM) strategy.

This aggregation mechanism is used and discussed in several papers:
[Hsu et al., 2019], [Huo et al., 2020]

Papers:
- https://arxiv.org/pdf/1909.06335.pdf
- https://arxiv.org/pdf/2002.02090.pdf
"""

from collections.abc import Callable, Iterable
from copy import deepcopy
from logging import DEBUG, INFO
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
import numpy as np

from flower_llm.strategy.aggregation import aggregate_cumulative_average
from flower_llm.strategy.rs_fedavg import FedAvgReproducibleSampling
from flower_llm.utils import l2_norm, sum_of_squares


# flake8: noqa: E501
class FedNesterov(FedAvgReproducibleSampling):
    """Configurable FedNesterov strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        initial_parameters: Parameters,
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
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        seed: int = 1337,
        server_learning_rate: float = 0.7,  # default DiLoCo value
        server_momentum: float = 0.9,  # default DiLoCo value
        rescale_global_model: bool = False,
        rescale_momentum_vector: bool = False,
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

        # Rescale global model
        self.rescale_global_model = rescale_global_model

        # Rescale momentum vector
        self.rescale_momentum_vector = rescale_momentum_vector

        # NOTE: This avoids translating between parameters and NDArrays every time.
        self.ndarray_parameters: NDArrays = parameters_to_ndarrays(initial_parameters)

        log(
            INFO,
            "Using Nesterov Momentum with server_learning_rate=%s and"
            " server_momentum=%s",
            self.server_learning_rate,
            self.server_momentum,
        )
        self.momentum_vector: NDArrays = deepcopy(
            parameters_to_ndarrays(initial_parameters)
        )

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
            result: tuple[ClientProxy, FitRes],
        ) -> tuple[ClientProxy, FitRes]:
            _, fit_res = result
            fit_metrics.append((fit_res.num_examples, fit_res.metrics))
            return result

        results = (acc_metrics(result) for result in results)

        results_cached: list[tuple[ClientProxy, FitRes]] = []

        if self.track_inplace_aggregation:
            results_cached = list(results)
            results = (val for val in results_cached)

        # Get the cumulative average of the results
        fedavg_result = aggregate_cumulative_average(results)

        # Return None if no results were aggregated
        if fedavg_result is None:
            return None, {}

        # Get FedAvg pseudo-gradient from FedAvg aggregated model
        fedavg_pseudo_gradient: NDArrays = [
            x - y for x, y in zip(self.ndarray_parameters, fedavg_result, strict=False)
        ]

        # Compute the new momentum vector
        new_momentum_vector: NDArrays = [
            w_old - self.server_learning_rate * g
            for w_old, g in zip(
                self.ndarray_parameters, fedavg_pseudo_gradient, strict=False
            )
        ]
        # Rescale the momentum vector if asked to
        if self.rescale_momentum_vector:
            log(DEBUG, "Rescaling the momentum vector")
            fedavg_norm = l2_norm(fedavg_result)
            current_norm = l2_norm(new_momentum_vector)
            # Choose the minimum norm as the target norm
            target_norm = min(fedavg_norm, current_norm)
            # Compute the scaling factor
            scaling_factor = target_norm / current_norm
            # Rescale the norm of the fedavgm result to match the norm of the fedavg result
            new_momentum_vector = [scaling_factor * v for v in new_momentum_vector]

        # Compute the new model
        fedavgm_result: NDArrays = [
            (1 + self.server_momentum) * v_new - self.server_momentum * v_old
            for v_new, v_old in zip(
                new_momentum_vector, self.momentum_vector, strict=False
            )
        ]

        # Rescale the global model if asked to
        if self.rescale_global_model:
            log(DEBUG, "Rescaling the global model")
            fedavg_norm = l2_norm(fedavg_result)
            current_norm = l2_norm(fedavgm_result)
            # Choose the minimum norm as the target norm
            target_norm = min(fedavg_norm, current_norm)
            # Compute the scaling factor
            scaling_factor = target_norm / current_norm
            # Rescale the norm of the fedavgm result to match the norm of the fedavg result
            fedavgm_result = [scaling_factor * v for v in fedavgm_result]

        # Update the momentum vector and the model
        self.momentum_vector = new_momentum_vector
        self.ndarray_parameters = fedavgm_result

        parameters_aggregated = ndarrays_to_parameters(fedavgm_result)

        # NOTE: This is handled by the server
        # # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated: dict[str, Scalar] = {}
        # if self.fit_metrics_aggregation_fn:
        #     metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        # elif server_round == 1:  # Only log this warning once
        #     log(WARNING, "No fit_metrics_aggregation_fn provided")

        if self.track_norms:
            metrics_aggregated |= {
                "server/l2_norm_pseudo_gradient": l2_norm(fedavg_pseudo_gradient),
                "server/l2_norm_momentum_vector": l2_norm(self.momentum_vector),
                "server/l2_norm_model": l2_norm(fedavgm_result),
                "server/l2_norm_fedavg_result": l2_norm(fedavg_result),
            }
            for i, plnpg in enumerate(
                [l2_norm([layer]) for layer in fedavg_pseudo_gradient]
            ):
                metrics_aggregated |= {
                    f"server/layer/{i}/l2_norm_pseudo_gradient": plnpg
                }
            for i, plnpg in enumerate(
                [l2_norm([layer]) for layer in self.momentum_vector]
            ):
                metrics_aggregated |= {
                    f"server/layer/{i}/l2_norm_momentum_vector": plnpg
                }
            for i, plnpg in enumerate([l2_norm([layer]) for layer in fedavgm_result]):
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_model": plnpg}
            for i, plnpg in enumerate([l2_norm([layer]) for layer in fedavg_result]):
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_fedavg_result": plnpg}
            log(
                INFO,
                "Nesterov Momentum: l2_norm(pseudo_gradient)=%s,"
                " l2_norm(self.momentum_vector)=%s, l2_norm(model)=%s,"
                " l2_norm(fedavg_result)=%s",
                l2_norm(fedavg_pseudo_gradient),
                l2_norm(self.momentum_vector),
                l2_norm(fedavgm_result),
                l2_norm(fedavg_result),
            )

        if self.track_inplace_aggregation:
            normal_result = aggregate(
                [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                    for _, fit_res in results_cached
                ]
            )
            layer_by_layer_diff = 0.0
            for x, y in zip(normal_result, fedavg_result, strict=False):
                layer_by_layer_diff += sum_of_squares([x - y])
            metrics_aggregated |= {
                "server/l2_norm_fedavg_gap": float(np.sqrt(layer_by_layer_diff)),
            }
            log(
                INFO,
                "Inplace aggregation gap: l2_norm(normal_result -"
                " fedavg_result)=%s, len_results: %s",
                layer_by_layer_diff,
                len(results_cached),
            )

        return parameters_aggregated, metrics_aggregated
