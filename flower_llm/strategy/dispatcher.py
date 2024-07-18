"""Dispatch strategy based on configuration."""

import sys

from flower_llm.conf.base_schema import BaseConfig, StrategyName
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
from flower_llm.strategy.aggregation import weighted_average
from flower_llm.strategy.rs_nesterov import FedNesterov


def dispatch_strategy(
    cfg: BaseConfig,
) -> FedNesterov | FedAvg:
    """Dispatch strategy based on configuration."""
    # NOTE: We need to lowercase the match as the Enum instantiate with auto() in Python
    # lowercases by default
    match cfg.fl.strategy_name.lower():
        case StrategyName.NESTOROV:
            assert (
                cfg.fl.strategy_kwargs.server_learning_rate is not None
            ), "Server learning rate is required for Nestorov strategy."
            assert (
                cfg.fl.strategy_kwargs.server_momentum is not None
            ), "Server momentum is required for Nestorov strategy."
            return FedNesterov(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                server_learning_rate=cfg.fl.strategy_kwargs.server_learning_rate,
                server_momentum=cfg.fl.strategy_kwargs.server_momentum,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
            )
        case StrategyName.FEDAVG:
            return FedAvg(
                # NOTE: We put a fake array as it will be touched on again later
                initial_parameters=Parameters(tensors=[], tensor_type="empty"),
                fit_metrics_aggregation_fn=weighted_average,
                evaluate_metrics_aggregation_fn=weighted_average,
                inplace=True,
                # These are not really important anymore with this new server
                fraction_fit=sys.float_info.min,
                fraction_evaluate=sys.float_info.min,
                min_fit_clients=cfg.fl.n_clients_per_round,
                min_evaluate_clients=1,
                min_available_clients=cfg.fl.n_clients_per_round,
                evaluate_fn=None,
                on_fit_config_fn=None,
                on_evaluate_config_fn=None,
                accept_failures=False,
            )
        case _:
            raise ValueError("Unknown strategy")
