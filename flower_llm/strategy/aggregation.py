"""Handle aggregation in-place and potentially async."""

import ast
from collections import defaultdict
import copy
from functools import partial, reduce
import time
from collections.abc import Generator, Iterable
from logging import DEBUG
from typing import Any
from flwr.common import FitRes, NDArrays, Parameters, bytes_to_ndarray, Config
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
import numpy as np


def aggregate_gradients(
    old_parameters: NDArrays,
    accumulator: tuple[NDArrays | None, int],
    current: tuple[NDArrays, int],
) -> tuple[NDArrays | None, int]:
    """Aggregate gradients from multiple clients on the accumulated gradients.

    Parameters
    ----------
        old_parameters : NDArrays
            The original model parameters before the current update.
        accumulator : tuple[NDArrays | None, int]
            A tuple containing the accumulated gradients and the total number of
            examples seen so far.
        current : tuple[NDArrays, int]
            A tuple containing the current gradients and the number of examples in the
            current update.

    Returns
    -------
        tuple[NDArrays | None, int]: A tuple containing the updated accumulated
        gradients and the new total number of examples.
    """
    current_params, num_examples = current
    start_time = time.time()
    log(DEBUG, f"Started aggregating client gradients with samples: {num_examples}")

    grads, prev_total_examples = accumulator

    # Compute the new total number of samples
    new_total_samples = prev_total_examples + num_examples

    # Compute scaling factor for the accumulator
    acc_scaling_factor = float(prev_total_examples) / new_total_samples

    # Compute scaling factor for the update
    scaling_factor = float(num_examples) / new_total_samples

    # Compute the pseudo-gradients
    current_grads = [x - y for x, y in zip(old_parameters, current_params, strict=True)]

    # NOTE: Maybe be useless but let's help the Python GC figure out what to do
    del current_params

    if grads is None:
        grads = list(current_grads)
    else:
        # Avoid allocating any temporaries
        for x, y in zip(grads, current_grads, strict=True):
            x *= acc_scaling_factor
            y *= scaling_factor
            x += y
            # Lack of scoping requires this
            del y

    # NOTE: Maybe be useless but let's help the Python GC figure out what to do
    del current_grads

    log(
        DEBUG,
        f"""Aggregated client with samples: {num_examples}
                total samples used: {new_total_samples}
                time: {time.time() - start_time} """,
    )

    return grads, new_total_samples


def aggregate_parameters(
    accumulator: tuple[NDArrays | None, int], current: tuple[NDArrays, int]
) -> tuple[NDArrays | None, int]:
    """Aggregate parameters in-place.

    Having this as a function avoids leaking variables
    Since python for-loops are unscoped.
    The function will use the memory of the the passed `current` parameter.

    Parameters
    ----------
        accumulator: Tuple containing the parameters and the total number of samples
        current: Tuple containing the client proxy and the fit result

    Returns
    -------
        Tuple containing the updated parameters and the new total number of samples
    """
    current_params, num_examples = current
    start_time = time.time()
    log(DEBUG, f"Started aggregating client parameters with samples: {num_examples}")

    params, prev_total_examples = accumulator

    # Compute the new total number of samples
    new_total_samples = prev_total_examples + num_examples

    # Compute scaling factor for the accumulator
    acc_scaling_factor = float(prev_total_examples) / new_total_samples

    # Compute scaling factor for the update
    scaling_factor = float(num_examples) / new_total_samples

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

    # NOTE: Maybe be useless but let's help the Python GC figure out what to do
    del current_params

    log(
        DEBUG,
        f"""Aggregated client with samples: {num_examples}
                total samples used: {new_total_samples}
                time: {time.time() - start_time} """,
    )

    return params, new_total_samples


def aggregate_inplace(
    results: Iterable[tuple[NDArrays, int]],
    old_parameters: NDArrays | None,
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async."""
    # Holds the parameters and the total number of samples
    accumulator: tuple[NDArrays | None, int] = (None, 0)

    # Choose the aggregation function
    aggregation_fn = (
        aggregate_parameters
        if old_parameters is None
        else partial(
            aggregate_gradients,
            old_parameters,
        )
    )

    # Aggregate the parameters
    agg_results, _ = reduce(aggregation_fn, results, accumulator)  # type: ignore[call-overload]

    return agg_results


def aggregate_cumulative_average(
    results: Iterable[tuple[ClientProxy, FitRes]],
    old_parameters: NDArrays | None,
) -> NDArrays | None:
    """Compute in-place weighted average, lazily and async."""
    # NOTE: Only one ndarray exists at a time
    return aggregate_inplace(
        results=(
            (
                parameters_to_ndarrays_gen(fit_res.parameters),  # type: ignore[reportArgumentType, misc]
                fit_res.num_examples,
            )
            for _, fit_res in results
        ),
        old_parameters=old_parameters,
    )


def parameters_to_ndarrays_gen(
    parameters: Parameters,
) -> Generator[np.ndarray, None, None]:
    """Convert parameters object to NumPy ndarrays."""
    return (bytes_to_ndarray(tensor) for tensor in parameters.tensors)


def partially_aggregate(
    current_agg: tuple[NDArrays, int], new_results: tuple[NDArrays, int]
) -> tuple[NDArrays, int]:
    """Aggregate partially parameters."""
    updated_agg = None
    # Assuming that the partially aggregate is empty when n_samples is 0
    if current_agg[1] == 0:
        updated_agg = copy.deepcopy(new_results[0])
        total_num_examples = copy.deepcopy(new_results[1])
    else:
        updated_agg = aggregate_inplace([current_agg, new_results], None)
        assert updated_agg is not None
        total_num_examples = copy.deepcopy(current_agg[1]) + copy.deepcopy(
            new_results[1]
        )
    return updated_agg, total_num_examples


def partially_aggregate_metrics(
    current_agg: tuple[int, Config], new_results: tuple[int, Config]
) -> tuple[int, Config]:
    """Aggregate partially parameters."""
    updated_agg = None
    # Assuming that the partially aggregate is empty when n_samples is 0
    if current_agg[0] == 0:
        total_num_examples = copy.deepcopy(new_results[0])
        updated_agg = copy.deepcopy(new_results[1])
    else:
        total_num_examples = current_agg[0] + copy.deepcopy(new_results[0])
        updated_agg = weighted_average([current_agg, new_results])
    return total_num_examples, updated_agg


def weighted_average(
    metrics: list[tuple[int, dict]],
) -> dict:
    """Compute a weighted average over pre-defined metrics.

    Parameters
    ----------
    metrics : List[Tuple[int, Dict]]
        The metrics to aggregate.

    Returns
    -------
    Dict
        The weighted average over pre-defined metrics.
    """
    client_state_accumulator: dict[int | str, dict[str, Any]] = {}
    total_num_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_metrics: dict = defaultdict(float)

    for num_examples, metric in metrics:
        if metric is not None:
            cid = metric.pop("cid", None)
            client_state = metric.pop("client_state", None)
            client_state_acc = metric.pop("client_state_acc", None)
            for key, value in metric.items():
                if not isinstance(value, str):
                    weighted_metrics[key] += num_examples * value
            if cid is not None and client_state is not None:
                client_state_accumulator[cid] = ast.literal_eval(client_state)
            if client_state_acc is not None:
                client_state_accumulator |= ast.literal_eval(client_state_acc)

    ret_dict = {
        key: value / total_num_examples for key, value in weighted_metrics.items()
    }
    if client_state_accumulator:
        ret_dict |= {"client_state_acc": str(client_state_accumulator)}

    return ret_dict
