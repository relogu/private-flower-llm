"""Handle aggregation in-place and potentially async."""

from typing import Any
from flwr.common import NDArrays

from flower_llm.conf.base_schema import BaseConfig
from flower_llm.wandb_history import WandbHistory


class ServerMetricCallback:
    """Callback for collecting metrics on the server side."""

    def __init__(self, metrics: dict[str, Any], conf: BaseConfig) -> None:
        """Initialize the ServerMetricCallback object."""
        self.metrics = metrics
        self.config: BaseConfig = conf

    def add_per_client_metrics(self, client_results: tuple[NDArrays, int]) -> None:
        """Add per-client metrics to the metrics dictionary."""

    def round_end(self, current_round: int, pseudo_gradient: NDArrays | None) -> None:
        """Add parameters metrics to the metrics dictionary."""


class SimpleNoiseScale(ServerMetricCallback):
    """Callback for collecting metrics on the server side."""

    def __init__(self, history: WandbHistory, conf: BaseConfig) -> None:
        """Initialize the ServerMetricCallback object."""
        self.history = history
        self.summed_grads_squares: list[float] | None = None
        self.b_small = conf.llm_config.global_train_batch_size
        self.b_big = self.b_small * conf.fl.n_clients_per_round

    def add_per_client_metrics(self, client_results: tuple[NDArrays, int]) -> None:
        """Add per-client metrics to the metrics dictionary."""
        grad, _ = client_results
        # Sum the square of the gradients for each client
        # Check if we have allocated a variable to do so
        if self.summed_grads_squares is None:
            self.summed_grads_squares = [(x**2).sum() for x in grad]
        else:
            self.summed_grads_squares = [
                x + (y**2).sum()
                for x, y in zip(self.summed_grads_squares, grad, strict=True)
            ]

    def round_end(self, current_round: int, pseudo_gradient: NDArrays | None) -> None:
        """Add parameters metrics to the metrics dictionary."""
        # Using the formula from "Towards an Empirical Model of Large-Batch Training"
        # https://arxiv.org/abs/1812.06162
        # Compute the noise scale

        if pseudo_gradient is None or self.summed_grads_squares is None:
            return

        b_small = self.b_small
        b_big = self.b_big
        g_small = sum(self.summed_grads_squares)
        g_big = sum((x**2).sum() for x in pseudo_gradient)
        noise = (b_big * g_big - b_small * g_small) / (b_big - b_small)
        scale = (g_small - g_big) / ((1 / b_small) - (1 / b_big))

        noise_scale = scale / noise

        self.history.add_metrics_centralized(
            current_round, metrics={"noise_scale": noise_scale}
        )
