"""Empty Flower Client for Pollen server."""

from collections.abc import Callable
from typing import Any

import flwr as fl
from flwr.common.typing import Config, NDArrays, Scalar


class EmptyVirtualClient(fl.client.NumPyClient):
    """Implement the most lightweight Flower Client."""

    def __init__(
        self,
        *,
        cid: int | str,
    ) -> None:
        self.cid = cid

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"EmptyVirtualClient(cid={self.cid})"

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return {}

    def get_parameters(
        self,
        config: Config,
    ) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        return []

    def fit(
        self, parameters: NDArrays, config: dict
    ) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any]]:
        """Implement the fit step."""
        return [], 0, {}

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Implement the evaluation step."""
        return 0.0, 0, {}


def gen_client_fn() -> Callable[[int], EmptyVirtualClient]:
    """Return generic `client_fn` for Flower Framework."""

    def client_fn(client_id: int) -> EmptyVirtualClient:
        client = EmptyVirtualClient(
            cid=client_id,
        )
        return client

    return client_fn
