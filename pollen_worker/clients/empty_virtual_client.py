"""Empty Flower Client for Pollen server."""
from typing import Any, Dict, Union

import flwr as fl
from flwr.common.typing import Config, NDArrays, Scalar


class EmptyVirtualClient(fl.client.NumPyClient):
    """Implement the most lightweight Flower Client."""

    def __init__(
        self,
        *,
        cid: Union[int, str],
    ) -> None:
        self.cid = cid

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"EmptyVirtualClient(cid={self.cid})"

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
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
        self, parameters: NDArrays, config: Dict
    ) -> tuple[NDArrays, int, Union[Dict[str, Scalar], dict[Any, Any]]]:
        """Implement the fit step."""
        return [], 0, {}

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> tuple[float, int, Dict[str, Scalar]]:
        """Implement the evaluation step."""
        return 0.0, 0, {}
