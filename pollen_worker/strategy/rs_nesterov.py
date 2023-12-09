"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""

import os
import pickle
import random
from logging import WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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
from flwr.server.strategy.aggregate import aggregate

from pollen_worker.strategy.rs_fedavg import FedAvgReproducibleSampling


# flake8: noqa: E501
class FedAvgRSModel(FedAvgReproducibleSampling):
    """Configurable FedAvgRSModel strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        saving_path: Optional[Path] = None,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        seed: int = 1337,
        freq: int = 1,
        server_learning_rate: float = 0.7,  # default DiLoCo value
        server_momentum: float = 0.9,  # default DiLoCo value
    ) -> None:
        """Federated Averaging strategy with with reproducible sampling and model
        saving.

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
            saving_path = Path(os.getcwd())
        self.saving_path = saving_path

        self.freq = freq

        # Default to DiLoCo values
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum

        # Avoid translating between parameters and NDArrays every time unnecessarily
        self.ndarray_parameters: Optional[NDArrays] = (
            parameters_to_ndarrays(initial_parameters)
            if initial_parameters is not None
            else None
        )

        self.momentum_vector: Optional[NDArrays] = None

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        assert (
            self.ndarray_parameters is not None
        ), "When using server-side optimization, model needs to be initialized."

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        pseudo_gradient: NDArrays = [
            x - y for x, y in zip(self.ndarray_parameters, weights_results)
        ]

        if server_round > 1:
            assert self.momentum_vector, "Momentum should have been created on round 1."

            self.momentum_vector = [
                self.server_momentum * v + w
                for w, v in zip(pseudo_gradient, self.momentum_vector)
            ]
        else:  # Round 1
            # Initialize server-side model

            # Initialize momentum vector
            self.momentum_vector = pseudo_gradient

        # TODO: Optimize this into a single computation to avoid iterating pseudo_gradient twice

        # Applying Nesterov
        pseudo_gradient = [
            g + self.server_momentum * v
            for g, v in zip(pseudo_gradient, self.momentum_vector)
        ]

        # Federated Averaging with Server Momentum
        fedavgm_result = [
            w - self.server_learning_rate * v
            for w, v in zip(self.ndarray_parameters, pseudo_gradient)
        ]

        self.ndarray_parameters = fedavgm_result

        parameters_aggregated = ndarrays_to_parameters(fedavgm_result)

        if server_round % self.freq == 0:
            # Save `parameters_aggregated`` to file
            with open(
                self.saving_path / f"parameters_aggregated_{server_round}", "wb"
            ) as f:
                pickle.dump(parameters_aggregated, f)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
