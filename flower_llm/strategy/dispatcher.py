"""Dispatch strategy based on configuration."""

from collections.abc import Callable
from flower_llm.conf.base_schema import BaseConfig, NesterovConfig, StrategyName
from flwr.server.strategy import FedAvg
from flower_llm.strategy.rs_nesterov import FedNesterov
from flwr.common import Scalar, NDArrays, Parameters, MetricsAggregationFn


def dispatch_strategy(
    cfg: BaseConfig,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    evaluate_fn: (
        None
        | Callable[
            [int, NDArrays, dict[str, Scalar]],
            None | tuple[float, dict[str, Scalar]],
        ]
    ) = None,
    on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
    on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
    accept_failures: bool = True,
    initial_parameters: Parameters | None = None,
    fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    inplace: bool = True,
) -> FedNesterov | FedAvg:
    """Dispatch strategy based on configuration."""
    assert initial_parameters is not None
    match cfg.fl.strategy_config.strategy_name:
        case StrategyName.nesterov:
            if not isinstance(cfg.fl.strategy_config, NesterovConfig):
                raise TypeError("Invalid configuration")
            return FedNesterov(
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
                server_learning_rate=cfg.fl.strategy_config.server_learning_rate,
                server_momentum=cfg.fl.strategy_config.server_momentum,
            )
        case StrategyName.fedavg:
            return FedAvg(
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
                inplace=inplace,
            )
        case _:
            raise ValueError("Unknown strategy")
