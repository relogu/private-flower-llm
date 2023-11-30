"""TODO: Add description here."""
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Tuple

import numpy as np
from flwr.common import NDArrays

POLLEN_CONFIG_SHM = "pollen_config_shm"
POLLEN_PARAMETERS_SHM = "pollen_parameters_shm"
POLLEN_WORKER_SHM = "pollen_worker_"


def allocate_shm(
    parameters: NDArrays,
    create: bool = False,
    name: str = POLLEN_PARAMETERS_SHM,
) -> Tuple[NDArrays, np.ndarray, np.ndarray, np.ndarray, SharedMemory]:
    """Allocate a Shared Memory object and backed arrays."""
    # Allocate memory for parameters and num_samples
    nbytes_params = [val.nbytes for val in parameters]
    nbytes_int = np.dtype(np.int64).itemsize
    nbytes_float = np.dtype(np.float64).itemsize
    array_bounds = [
        (sum(nbytes_params[:i]), sum(nbytes_params[: i + 1]))
        for i in range(len(nbytes_params))
    ]
    if create:
        total_num_bytes = sum(nbytes_params) + nbytes_int + 2 * nbytes_float
        shm = SharedMemory(create=True, size=total_num_bytes, name=name)
        shm.buf[:] = b"\0" * shm.size
    else:
        shm = SharedMemory(name=name)
    params_sh: NDArrays = [
        np.ndarray(shape=x.shape, dtype=x.dtype, buffer=shm.buf[y[0] : y[1]])
        for x, y in zip(parameters, array_bounds)
    ]
    # Create shared memory for num_samples, train loss, and train accuracy
    num_samples_sh: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
        (1,),
        dtype=np.int64,
        buffer=shm.buf[-int(nbytes_int + 2 * nbytes_float) : -int(2 * nbytes_float)],
    )
    train_loss_sh: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
        (1,), dtype=np.float64, buffer=shm.buf[-int(2 * nbytes_float) : -nbytes_float]
    )
    train_accuracy_sh: np.ndarray[Any, np.dtype[Any]] = np.ndarray(
        (1,), dtype=np.float64, buffer=shm.buf[-nbytes_float:]
    )
    return params_sh, num_samples_sh, train_loss_sh, train_accuracy_sh, shm


def write_to_fit_result_shm(
    buffer_backed_ndarrays: NDArrays,
    buffer_backed_num_samples: np.ndarray,
    buffer_backed_train_loss: np.ndarray,
    buffer_backed_train_accuracy: np.ndarray,
    new_ndarrays: NDArrays,
    new_num_samples: int,
    new_train_loss: float,
    new_train_accuracy: float,
) -> None:
    """Write to Shared Memory through backed arrays."""
    for i in range(len(new_ndarrays)):
        if len(new_ndarrays[i].shape) == 0:
            buffer_backed_ndarrays[i] = new_ndarrays[i]
        else:
            buffer_backed_ndarrays[i][:] = new_ndarrays[i][:]
    buffer_backed_num_samples[0] = new_num_samples
    buffer_backed_train_loss[0] = new_train_loss
    buffer_backed_train_accuracy[0] = new_train_accuracy
