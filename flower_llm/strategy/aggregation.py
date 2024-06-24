"""Handle aggregation in-place and potentially async."""

from functools import reduce
import time
from collections.abc import Iterable
from logging import DEBUG
from flwr.common import FitRes, NDArrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flower_llm.utils import parameters_to_ndarrays_gen


def aggregate_parameters(
    accumulator: tuple[NDArrays | None, int], current: tuple[ClientProxy, FitRes]
) -> tuple[NDArrays | None, int]:
    """Aggregate parameters in-place.

    Having this as a function avoids leaking variables
    Since python for-loops are unscoped.

    Parameters
    ----------
        accumulator: Tuple containing the parameters and the total number of samples
        current: Tuple containing the client proxy and the fit result

    Returns
    -------
        Tuple containing the updated parameters and the new total number of samples
    """
    client_proxy, fit_res = current
    start_time = time.time()
    log(DEBUG, f"Started aggregating cid: {client_proxy.cid}")

    params, prev_total_examples = accumulator

    # Compute the new total number of samples
    new_total_samples = prev_total_examples + fit_res.num_examples

    # Compute scaling factor for the accumulator
    acc_scaling_factor = float(prev_total_examples) / new_total_samples

    # Compute scaling factor for the update
    scaling_factor = float(fit_res.num_examples) / new_total_samples

    # Only one ndarray exists at a time
    current_params = parameters_to_ndarrays_gen(fit_res.parameters)

    if params is None:
        params = list(current_params)
    else:
        # Avoid allocating any temporaries
        for x, y in zip(params, current_params, strict=True):
            x *= acc_scaling_factor
            y *= scaling_factor
            x += y
            # Lack of scoping requires this
            del y

    del fit_res.parameters

    log(
        DEBUG,
        f"""Aggregated cid: {client_proxy.cid}
                    with samples: {fit_res.num_examples}
                    total samples used: {new_total_samples}
                    time: {time.time() - start_time} """,
    )

    return params, new_total_samples


def aggregate_cumulative_average(
    results: Iterable[tuple[ClientProxy, FitRes]],
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async."""
    # Holds the parameters and the total number of samples
    accumulator: tuple[NDArrays | None, int] = (None, 0)

    params, _num_total_examples = reduce(aggregate_parameters, results, accumulator)

    return params
