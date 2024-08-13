"""Adaptive Federated Optimization using Yogi (FedYogi) [Reddi et al., 2020] strategy.

The paper can be found at [this link](https://arxiv.org/abs/2003.00295).
"""

from collections.abc import Callable, Iterable
from logging import INFO

import numpy as np

from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    log,
    ndarray_to_bytes,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from flower_llm.strategy.aggregation import (
    aggregate_cumulative_average,
    parameters_to_ndarrays_gen,
)
from flower_llm.utils import l2_norm


# pylint: disable=line-too-long
class FedYogi(FedAvg):
    """FedYogi [Reddi et al., 2020] strategy.

    Implementation based on https://arxiv.org/abs/2003.00295v5

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        )
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters
        Initial global model parameters.
    fit_metrics_aggregation_fn : MetricsAggregationFn | None
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: MetricsAggregationFn | None
        Metrics aggregation function, optional.
    eta : float, optional
        Server-side learning rate. Defaults to 1e-2.
    eta_l : float, optional
        Client-side learning rate. Defaults to 0.0316.
    beta_1 : float, optional
        Momentum parameter. Defaults to 0.9.
    beta_2 : float, optional
        Second moment parameter. Defaults to 0.99.
    tau : float, optional
        Controls the algorithm's degree of adaptability.
        Defaults to 1e-3.
    use_gradients: bool, optional
        Flag for using gradients in the aggregation instead of model parameters.
        Defaults to True.
    track_norms: bool, optional
        Flag for tracking the norms of the aggregated updates. Defaults to True.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals, line-too-long
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
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        use_gradients: bool = True,
        track_norms: bool = True,
    ) -> None:
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

        # NOTE: This avoids translating between parameters and NDArrays every time.
        # However, it incurs in a higher memory peak. We decided to go for the previous
        # approach that uses a pointer to the parameters at the server.
        # self.ndarray_parameters: NDArrays = parameters_to_ndarrays(initial_parameters)
        self.parameters: Parameters = initial_parameters
        assert self.parameters is self.initial_parameters

        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # Lazy initialization
        self.momentum_vector: NDArrays = [
            np.zeros_like(x) for x in parameters_to_ndarrays(self.parameters)
        ]
        self.second_momentum_vector: NDArrays = [
            np.zeros_like(x) for x in parameters_to_ndarrays(self.parameters)
        ]

        # Aggregation implementation
        self.use_gradients = use_gradients

        # Metrics tracking
        self.track_norms = track_norms

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedYogi(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: Iterable[tuple[ClientProxy, FitRes]],
        failures: Iterable[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        assert (
            self.parameters is not None
        ), "When using server-side optimization, model needs to be initialized."

        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []

        def acc_metrics(
            result: tuple[ClientProxy, FitRes],
        ) -> tuple[ClientProxy, FitRes]:
            _, fit_res = result
            fit_metrics.append((fit_res.num_examples, fit_res.metrics))
            return result

        results = (acc_metrics(result) for result in results)

        # Get the cumulative average of the results
        fedavg_result = aggregate_cumulative_average(
            results,
            old_parameters=(
                parameters_to_ndarrays(self.parameters) if self.use_gradients else None
            ),
        )

        # Return None if no results were aggregated
        if fedavg_result is None:
            return None, {}

        # Initialize the metrics
        layerwise_l2_norms_pseudo_gradient: list[float] = []
        layerwise_l2_norms_momentum_vector: list[float] = []
        layerwise_l2_norms_second_momentum_vector: list[float] = []
        layerwise_l2_norms_fedavg_result: list[float] = []
        layerwise_l2_norms_model: list[float] = []
        # Loop over layer, apply the server optimizer and compute metrics
        for i, x in enumerate(parameters_to_ndarrays_gen(self.parameters)):
            # Layer i pseudo-gradient
            layer_pseudo_gradient = (
                fedavg_result[i] if self.use_gradients else x - fedavg_result[i]
            )
            # Compute first momentum of layer i
            self.momentum_vector[i] = (
                self.beta_1 * self.momentum_vector[i]
                + (1 - self.beta_1) * layer_pseudo_gradient
            ) * (1 / (self.beta_1**server_round))
            # Compute second momentum of layer i
            self.second_momentum_vector[i] = (
                self.second_momentum_vector[i]
                + (1 - self.beta_2)
                * np.multiply(layer_pseudo_gradient, layer_pseudo_gradient)
                * np.sign(
                    np.multiply(layer_pseudo_gradient, layer_pseudo_gradient)
                    - self.second_momentum_vector[i]
                )
            ) * (1 / (self.beta_2**server_round))
            # Compute the new weights of layer i
            layer_fedyogi_result = x + self.eta * np.divide(
                self.momentum_vector[i],
                (np.sqrt(self.second_momentum_vector[i]) + self.tau),
            )
            # Assign new values to the parameters variable
            self.parameters.tensors[i] = ndarray_to_bytes(layer_fedyogi_result)

            # Metrics collection
            layerwise_l2_norms_pseudo_gradient.append(l2_norm([layer_pseudo_gradient]))
            layerwise_l2_norms_momentum_vector.append(
                l2_norm([self.momentum_vector[i]])
            )
            layerwise_l2_norms_second_momentum_vector.append(
                l2_norm([self.second_momentum_vector[i]])
            )
            layerwise_l2_norms_fedavg_result.append(
                l2_norm([x - layer_pseudo_gradient])
            )
            layerwise_l2_norms_model.append(l2_norm([layer_fedyogi_result]))

        metrics_aggregated: dict[str, Scalar] = {}
        if self.track_norms:
            metrics_aggregated |= {
                "server/l2_norm_pseudo_gradient": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_pseudo_gradient))
                ),
                "server/l2_norm_momentum_vector": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_momentum_vector))
                ),
                "server/l2_norm_second_momentum_vector": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_second_momentum_vector))
                ),
                "server/l2_norm_fedavg_result": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_fedavg_result))
                ),
                "server/l2_norm_model": np.sqrt(
                    np.sum(np.square(layerwise_l2_norms_model))
                ),
            }
            for i, (a, b, c, d, e) in enumerate(
                zip(
                    layerwise_l2_norms_pseudo_gradient,
                    layerwise_l2_norms_momentum_vector,
                    layerwise_l2_norms_second_momentum_vector,
                    layerwise_l2_norms_fedavg_result,
                    layerwise_l2_norms_model,
                    strict=True,
                )
            ):
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_pseudo_gradient": a}
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_momentum_vector": b}
                metrics_aggregated |= {
                    f"server/layer/{i}/l2_norm_second_momentum_vector": c
                }
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_fedavg_result": d}
                metrics_aggregated |= {f"server/layer/{i}/l2_norm_model": e}
            log(
                INFO,
                "FedYogi:"
                " l2_norm(pseudo_gradient)=%s,"
                " l2_norm(momentum_vector)=%s,"
                " l2_norm(second_momentum_vector)=%s,"
                " l2_norm(fedavg_result)=%s"
                " l2_norm(model)=%s,",
                metrics_aggregated["server/l2_norm_pseudo_gradient"],
                metrics_aggregated["server/l2_norm_momentum_vector"],
                metrics_aggregated["server/l2_norm_second_momentum_vector"],
                metrics_aggregated["server/l2_norm_fedavg_result"],
                metrics_aggregated["server/l2_norm_model"],
            )

        return self.parameters, metrics_aggregated
