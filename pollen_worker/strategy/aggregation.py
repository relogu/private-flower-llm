"""Handle aggregation in-place and potentially async."""
from typing import Iterable, Tuple

import numpy as np
from flwr.common import FitRes, NDArrays, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy


def aggregate_cumulative_average(
    results: Iterable[Tuple[ClientProxy, FitRes]]
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async."""
    # Initialize params,
    # the iterator may not contain anything
    # and we do not want a next+try-except
    params: NDArrays | None = None

    num_total_examples: int = 0  # total number of examples, aggregated over time

    for _, fit_res in results:
        # Compute the new total number of samples
        new_total_samples = num_total_examples + fit_res.num_examples

        # Compute scaling factor for the update
        scaling_factor: float = float(fit_res.num_examples) / new_total_samples

        # Generator to multiply the layers by the scaling factor
        # Lazy operation, will be expanded in the zip
        res = (scaling_factor * x for x in parameters_to_ndarrays(fit_res.parameters))

        if params is None:
            # Initialize with the first set of parameters
            params = list(res)
        else:
            # Invert the previous division now
            # that we have updated information on the number of samples
            # Then divide by the new total number
            general_scaling = float(num_total_examples) / new_total_samples

            # Lazy generator for scaled layers
            scaled_params = (general_scaling * x for x in params)

            # Create new parameters
            params = [np.add(x, y) for x, y in zip(scaled_params, res)]

        # Update total number of examples
        num_total_examples = new_total_samples

    return params
