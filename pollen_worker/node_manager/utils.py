"""TODO: Add description here."""

import copy
import pickle
from logging import ERROR
from multiprocessing import resource_tracker as res_track
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from flwr.common import Config, NDArrays
from flwr.common.logger import log
from flwr.server.strategy.aggregate import aggregate

from pollen_worker.utils import (
    partially_aggregate,
    partially_aggregate_metrics,
    weighted_average,
)

POLLEN_CONFIG_SHM = "_pollen_config_shm"
POLLEN_PARAMETERS_SHM = "_pollen_parameters_shm"
POLLEN_N_SAMPLES_SHM = "_pollen_n_samples_shm"
POLLEN_EVAL_LOSS_SHM = "_pollen_eval_loss_shm"
POLLEN_METRICS_SHM = "_pollen_metrics_shm"


def aggregate_training_results(
    parameters: list[tuple[NDArrays, int]],
    samples: list[int],
    metrics: list[tuple[int, dict]],
) -> tuple[NDArrays, int, dict]:
    """Aggregate the training results."""
    return (
        aggregate(parameters),
        sum(samples),
        weighted_average(metrics),
    )


def partially_aggregate_training_results(
    old_results: tuple[NDArrays, int, dict],
    new_results: tuple[NDArrays, int, dict],
) -> tuple[NDArrays, int, dict]:
    """Aggregate partially the training results."""
    # Partial aggregation for parameters and n_samples
    (p_agg_params, p_agg_samples) = partially_aggregate(
        (old_results[0], old_results[1]),
        (copy.deepcopy(new_results[0]), copy.deepcopy(new_results[1])),
    )
    # Partial aggregation for metrics
    (p_agg_samples, p_agg_metrics) = partially_aggregate_metrics(
        (old_results[1], old_results[2]),
        (copy.deepcopy(new_results[1]), copy.deepcopy(new_results[2])),
    )
    return (
        p_agg_params,
        p_agg_samples,
        p_agg_metrics,
    )


def get_ndarrays_size_and_bounds(
    ndarrays: NDArrays,
) -> tuple[int, list[tuple[int, int]]]:
    """Return the total byte size of the NDArrays and their bounds."""
    nbytes = [val.nbytes for val in ndarrays]
    array_bounds = [(sum(nbytes[:i]), sum(nbytes[: i + 1])) for i in range(len(nbytes))]
    return sum(nbytes), array_bounds


def zero_out_shm(
    shm: SharedMemory,
) -> None:
    """Zero out the Shared Memory object."""
    shm.buf[:] = b"\0" * shm.size


def get_config_shm(
    config: Config,
    create: bool = False,
    name: str = POLLEN_CONFIG_SHM,
) -> tuple[Config, SharedMemory]:
    """Get a Shared Memory object and its backed config."""
    if create and config is None:
        raise ValueError("Cannot create config without config object.")
    if create:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        # TODO: Evaluate if we need to set up some margin here
        shm = SharedMemory(create=True, size=len(config_bytes), name=name)
        config_sh = config
    else:
        shm = SharedMemory(name=name)
        config_sh = pickle.loads(shm.buf)
    return config_sh, shm


def set_config_shm(
    config: Config,
    shm: SharedMemory,
) -> None:
    """Set Shared Memory object and backed config."""
    config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
    shm.buf[:] = config_bytes


def get_parameters_shm(
    parameters: NDArrays,
    create: bool = False,
    name: str = POLLEN_PARAMETERS_SHM,
) -> tuple[NDArrays, SharedMemory]:
    """Allocate a Shared Memory object and backed arrays."""
    total_num_bytes, array_bounds = get_ndarrays_size_and_bounds(parameters)
    if create:
        shm = SharedMemory(create=True, size=total_num_bytes, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh: NDArrays = [
        np.ndarray(shape=x.shape, dtype=x.dtype, buffer=shm.buf[y[0] : y[1]])
        for x, y in zip(parameters, array_bounds, strict=False)
    ]
    return params_sh, shm


def set_parameters_shm(
    old_parameters_sh: NDArrays,
    new_parameters: NDArrays,
) -> None:
    """Set Shared Memory object and backed arrays."""
    for i in range(len(new_parameters)):
        if len(new_parameters[i].shape) == 0:
            old_parameters_sh[i] = new_parameters[i]
        else:
            old_parameters_sh[i][:] = new_parameters[i][:]


def get_num_samples_shm(
    create: bool = False,
    name: str = POLLEN_N_SAMPLES_SHM,
) -> tuple[np.ndarray, SharedMemory]:
    """Allocate a Shared Memory object and backed arrays."""
    if create:
        shm = SharedMemory(create=True, size=np.dtype(np.int64).itemsize, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    num_samples_sh: np.ndarray = np.ndarray((1,), dtype=np.int64, buffer=shm.buf)
    return num_samples_sh, shm


def set_num_samples_shm(
    old_num_samples_sh: np.ndarray,
    new_num_samples: int,
) -> None:
    """Set Shared Memory object and backed arrays."""
    old_num_samples_sh[0] = new_num_samples


def get_eval_loss_shm(
    create: bool = False,
    name: str = POLLEN_N_SAMPLES_SHM,
) -> tuple[np.ndarray, SharedMemory]:
    """Allocate a Shared Memory object and backed arrays."""
    if create:
        shm = SharedMemory(create=True, size=np.dtype(np.float64).itemsize, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    eval_loss_sh: np.ndarray = np.ndarray((1,), dtype=np.float64, buffer=shm.buf)
    return eval_loss_sh, shm


def set_eval_loss_shm(
    old_eval_loss_sh: np.ndarray,
    new_eval_loss: float,
) -> None:
    """Set Shared Memory object and backed arrays."""
    old_eval_loss_sh[0] = new_eval_loss


def close_all_shms(process_uuid: str) -> None:
    """Close all Shared Memories of the process with the given UUID."""
    shms_names = [
        process_uuid + POLLEN_CONFIG_SHM,
        process_uuid + POLLEN_PARAMETERS_SHM,
        process_uuid + POLLEN_N_SAMPLES_SHM,
        process_uuid + POLLEN_EVAL_LOSS_SHM,
        process_uuid + POLLEN_METRICS_SHM,
    ]
    for shm_name in shms_names:
        try:
            shm = SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except Exception as e:
            if "[Errno 2] No such file or directory" in str(e):
                continue
            else:
                log(
                    ERROR,
                    "Removing Shared Memory %s failed because of %s",
                    shm_name,
                    e,
                )


def remove_shm_from_resource_tracker() -> None:
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked.

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype) -> None:
        if rtype == "shared_memory":
            return None
        return res_track._resource_tracker.register(name, rtype)

    res_track.register = fix_register

    def fix_unregister(name, rtype) -> None:
        if rtype == "shared_memory":
            return None
        return res_track._resource_tracker.unregister(name, rtype)

    res_track.unregister = fix_unregister

    if "shared_memory" in res_track._CLEANUP_FUNCS:  # type: ignore[attr-defined]
        del res_track._CLEANUP_FUNCS["shared_memory"]  # type: ignore[attr-defined]
