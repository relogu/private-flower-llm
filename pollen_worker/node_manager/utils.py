"""TODO: Add description here."""
import pickle
from logging import ERROR
from multiprocessing.shared_memory import SharedMemory
from typing import List, Optional, Tuple

import numpy as np
from flwr.common import Config, NDArrays
from flwr.common.logger import log

POLLEN_CONFIG_SHM = "pollen_config_shm"
POLLEN_PARAMETERS_SHM = "pollen_parameters_shm"
POLLEN_N_SAMPLES_SHM = "pollen_n_samples_shm"
POLLEN_TRAIN_METRICS_SHM = "pollen_train_metrics_shm"


def get_ndarrays_size_and_bounds(
    ndarrays: NDArrays,
) -> Tuple[int, List[Tuple[int, int]]]:
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
    config: Optional[Config] = None,
    create: bool = False,
    name: str = POLLEN_CONFIG_SHM,
) -> Tuple[Config, SharedMemory]:
    """Get a Shared Memory object and backed config."""
    if create and config is None:
        raise ValueError("Cannot create config without config object.")
    if create:
        config_bytes = pickle.dumps(config, protocol=pickle.HIGHEST_PROTOCOL)
        # TODO: Evaluate if we need to set up some margin here
        shm = SharedMemory(create=True, size=len(config_bytes), name=name)
        shm.buf[:] = config_bytes
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
) -> Tuple[NDArrays, SharedMemory]:
    """Allocate a Shared Memory object and backed arrays."""
    total_num_bytes, array_bounds = get_ndarrays_size_and_bounds(parameters)
    if create:
        shm = SharedMemory(create=True, size=total_num_bytes, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh: NDArrays = [
        np.ndarray(shape=x.shape, dtype=x.dtype, buffer=shm.buf[y[0] : y[1]])
        for x, y in zip(parameters, array_bounds)
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
) -> Tuple[np.ndarray, SharedMemory]:
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


def close_all_shms(process_uuid: str) -> None:
    """Close all Shared Memories of the process with the given UUID."""
    shms_names = [
        process_uuid + POLLEN_CONFIG_SHM,
        process_uuid + POLLEN_PARAMETERS_SHM,
        process_uuid + POLLEN_N_SAMPLES_SHM,
        process_uuid + POLLEN_TRAIN_METRICS_SHM,
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
