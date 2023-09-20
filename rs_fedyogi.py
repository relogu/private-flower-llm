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
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""

import os
import pickle
import random
from logging import DEBUG, INFO, WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Optional

import numpy as np
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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedYogi
from flwr.server.strategy.aggregate import aggregate


# flake8: noqa: E501
class FedYogiReproducibleSampling(FedYogi):
    """Configurable FedYogiReproducibleSampling strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
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
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        seed: int = 1337,
    ) -> None:
        """Federated learning strategy using Yogi on server-side with reproducible sampling.

        Implementation based on https://arxiv.org/abs/2003.00295v5

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
        eta : float, optional
            Server-side learning rate. Defaults to 1e-1.
        eta_l : float, optional
            Client-side learning rate. Defaults to 1e-1.
        beta_1 : float, optional
            Momentum parameter. Defaults to 0.9.
        beta_2 : float, optional
            Second moment parameter. Defaults to 0.99.
        tau : float, optional
            Controls the algorithm's degree of adaptability.
            Defaults to 1e-9.
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
            initial_parameters=initial_parameters, # type: ignore
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )
        self.seed = seed

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # Wait for the minimum number of clients to be available
        client_manager.wait_for(sample_size) # type: ignore

        # Setting seed for reproducibility of client selection
        random.seed(self.seed + server_round)

        # Generate random selection of virtual clients (number of virtual clients per round)
        sampled_virtual_cids = random.sample(list(client_manager.clients), sample_size) # type: ignore

        # Get the actual clients from the client manager
        clients = [client_manager.clients[cid] for cid in sampled_virtual_cids] # type: ignore

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


# flake8: noqa: E501
class FedYogiRSModel(FedYogiReproducibleSampling):
    """Configurable FedYogiRSModel strategy implementation."""

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
        eta: float = 1e-2,
        eta_l: float = 0.0316,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
        seed: int = 1337,
        freq: int = 1,
    ) -> None:
        """Federated learning strategy using Yogi on server-side with reproducible sampling and model saving.

        Implementation based on https://arxiv.org/abs/2003.00295v5

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
        eta : float, optional
            Server-side learning rate. Defaults to 1e-1.
        eta_l : float, optional
            Client-side learning rate. Defaults to 1e-1.
        beta_1 : float, optional
            Momentum parameter. Defaults to 0.9.
        beta_2 : float, optional
            Second moment parameter. Defaults to 0.99.
        tau : float, optional
            Controls the algorithm's degree of adaptability.
            Defaults to 1e-9.
        seed : int, optional
            Seed for reproducibility. Defaults to 1337.
        freq : int, optional
            Frequency of saving the aggregated parameters. Defaults to 1.
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
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
            seed=seed,
        )
        if saving_path is None:
            saving_path = Path(os.getcwd())
        self.saving_path = saving_path
        self.freq = freq

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round=server_round, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

        # Yogi
        delta_t: NDArrays = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
            for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        # Save `self.current_weights`` to file
        if server_round % self.freq == 0:
            with open(
                self.saving_path / f"parameters_aggregated_{server_round}", "wb"
            ) as f:
                pickle.dump(ndarrays_to_parameters(self.current_weights), f)

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated
