"""Utility functions for FL and experiment management.

They assure compatibility with the Flower and wandb APIs.
"""

import copy
import gc
import shutil
from collections import OrderedDict, defaultdict
from functools import reduce
from logging import INFO, WARN
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import psutil
import pyarrow as pa
import ray
import torch
from composer import Engine, Trainer
from flwr.common import Config, FitRes, NDArrays, Scalar, log, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from torch import device as device_type

import wandb

# NOTE: Setting the maximum value according to the documentation
# https://github.com/grpc/grpc/blob/eeae8e635a896bfa420d21e476221af652fd9986/include/grpc/impl/codegen/grpc_types.h#L150
POLLEN_LLM_MAX_MESSAGE_LENGTH = -1


#### Server ####
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
    total_num_examples = sum(
        [num_examples for num_examples, _ in metrics],
    )
    weighted_metrics: dict = defaultdict(float)
    for num_examples, metric in metrics:
        if metric is not None:
            for key, value in metric.items():
                weighted_metrics[key] += num_examples * value

    return {key: value / total_num_examples for key, value in weighted_metrics.items()}


def partially_aggregate(
    current_agg: Tuple[NDArrays, int], new_results: Tuple[NDArrays, int]
) -> Tuple[NDArrays, int]:
    """Aggregate partially parameters."""
    updated_agg = None
    # Assuming that the partially aggregate is empty when n_samples is 0
    if current_agg[1] == 0:
        updated_agg = copy.deepcopy(new_results[0])
        total_num_examples = copy.deepcopy(new_results[1])
    else:
        updated_agg = aggregate([current_agg, new_results])
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


#### Client ####
## General
def get_parameters(net: torch.nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client."""
    net.eval()
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(
    net: torch.nn.Module, parameters: NDArrays, device: str = "cpu"
) -> None:
    """Implement generic `set_parameters` for Flower Client."""
    net.eval()
    keys = [k for k in net.state_dict().keys() if "bn" not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict(
        {k: torch.tensor(v, device=device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict=state_dict, strict=False)


def invert_many_to_one_dictionary(
    input: Dict,
) -> Dict:
    """Invert the mapping given by a dictionary when it is many-to-one."""
    output: Dict = defaultdict(list)
    for k, v in input.items():
        output[v] = output.get(v, []) + [k]
    return output


def invert_one_to_many_dictionary(
    input: Dict,
) -> Dict:
    """Invert the mapping given by a dictionary when it is one-to-many."""
    output: Dict = {}
    for k, v in input.items():
        for w in v:
            output[w] = k
    return output


def gen_on_fit_config_fn(
    batch_size: int = 10,
    local_epochs: int = 1,
    learning_rate: float = 0.1,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    is_fake: bool = False,
    n_workers: int = 0,
) -> Callable[[int], Dict[str, Scalar]]:
    """Return generic `on_fit_config_fn` for Flower Client."""

    def on_fit_config_fn(server_round: int) -> Dict[str, Scalar]:
        """Return `Config` for fit/evaluate rounds."""
        return {
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "learning_rate": learning_rate,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "server_round": server_round,
            "is_fake": is_fake,
            # TODO: Brainstorm how to set this hyperparameter
            "n_workers": n_workers,
        }

    return on_fit_config_fn


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        """Do nothing."""


def wandb_init(
    wandb_enabled: bool, *args, **kwargs
) -> Optional[Union[NoOpContextManager, Any]]:
    """Initialize wandb if enabled."""
    if wandb_enabled:
        return wandb.init(*args, **kwargs)

    return NoOpContextManager()


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self):
        """Initialize the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Cleanup the files."""
        if ray.is_initialized():
            temp_dir = Path(
                ray.worker._global_node.get_session_dir_path()  # type: ignore
            )
            ray.shutdown()
            directory_size = shutil.disk_usage(temp_dir).used
            shutil.rmtree(temp_dir)
            print(
                f"Cleaned up ray temp session: {temp_dir} with size: {directory_size}"
            )


def chunks_idx(list: Sequence, n_chunks: int) -> Generator[tuple[int, int], Any, None]:
    """Split a list in n_chunks of equal length."""
    d, r = divmod(len(list), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
        yield si, si + (d + 1 if i < r else d)


def l1_norm(arrays: NDArrays) -> float:
    """Compute the L1 norm of a list of arrays.

    Parameters
    ----------
    arrays : NDArrays
        List of arrays to compute the L1 norm of.

    Returns
    -------
    float
        The L1 norm of the list of arrays.
    """
    return sum(np.sum(np.abs(arr)) for arr in arrays)


def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum([fit_res.num_examples for _, fit_res in results])

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / num_examples_total for _, fit_res in results
    ]

    # Let's do in-place aggregation
    # get first result, then add up each other
    params = [
        scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
    ]
    for i, (_, fit_res) in enumerate(results[1:]):
        res = (
            scaling_factors[i + 1] * x
            for x in parameters_to_ndarrays(fit_res.parameters)
        )
        params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

    return params


def get_device() -> device_type:
    """Determine which device to use for PyTorch.

    Returns
    -------
        str: device for PyTorch
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif (
        torch.backends.mps.is_available()  # type: ignore
        and torch.backends.mps.is_built()  # type: ignore
    ):
        device = "mps"
    return cast(device_type, device)


def get_n_cuda_devices() -> int:
    """Get the number of CUDA devices available."""
    if get_device() == "cuda":
        return torch.cuda.device_count()
    else:
        return 0


def get_n_cpu_cores() -> int:
    """Get the number of CPU cores available."""
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    return cpus


def get_pyarrow_buffer_from_table(table: pa.Table) -> pa.Buffer:
    """Cast a PyArrow Table into a Buffer."""
    buffer = pa.BufferOutputStream()
    with pa.ipc.new_file(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def get_table_from_pyarrow_buffer(buffer: pa.Buffer) -> pa.Table:
    """Cast a Buffer into a PyArrow Table ."""
    ret_table = None
    with pa.ipc.open_file(buffer) as reader:
        ret_table = reader.read_all()
    return ret_table


def namestr(obj, namespace):
    """Return the name of an object in the given namespace."""
    return [name for name in namespace if namespace[name] is obj]


def get_referenced_tensors_summary(cuda_only: bool = True, verbose: bool = True) -> str:
    """Inspect the tensors in the current Python session."""
    # Initalizing the summary string and variables
    summary = ""
    counter, total_size, gpu_size = 0, 0, 0
    gc.collect()
    # Looping over the objects in the current Python session
    for obj in gc.get_objects():
        # Surrounding the tensor inspection with a try-except block
        try:
            # Checking if the object is a tensor or a tensor data attribute
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                # Skip if the tensor is on CPU and `cuda_only` is True
                if cuda_only and not obj.is_cuda:
                    continue
                else:
                    # Getting the memory allocation of the current object
                    mem_alloc = obj.element_size() * obj.nelement()
                    # Getting the referrers of the current object
                    # NOTE: This creates a new referrer!
                    referrers = gc.get_referrers(obj)
                    # Building the summary for the current object:
                    # ( type (some tensor type), size (shape)
                    summary += f"(type{type(obj)}, {obj.size()}, "
                    # whether it requires grad, memory allocation
                    summary += f"r_g={obj.requires_grad}, mem={mem_alloc}, "
                    # whether it is on GPU, the number of referrers
                    summary += f"cuda={obj.is_cuda}, n_ref={len(referrers)}, "
                    # referrers
                    summary += f"refs={[r for r in referrers if type(r) is not list]}, "
                    # # looking for names of the first referrer (DOESN'T WORK)
                    # summary += f"{namestr(referrers[0], globals())}, "
                    # summary += f"{namestr(referrers[0], locals())}, "
                    # type of the referrers
                    summary += f"type_ref={[type(r) for r in referrers]}, "
                    # # referrers of the referrers

                    # summary += f"{[gc.get_referrers(referrers) for r in referrers]})"
                    summary += "\n"
                    # Updating the counters
                    counter += 1
                    total_size += mem_alloc
                    if obj.is_cuda:
                        gpu_size += mem_alloc
        except Exception:
            # log(
            #     ERROR,
            #     "get_referenced_tensors_summary :: error while inspecting ",
            #     "object of type %s",
            #     type(obj),
            #     exc_info=e,
            #     stack_info=True,
            # )
            pass
    if verbose:
        # Converting the size from bytes to MiB
        total_size_mb = total_size / 1e6
        gpu_size_mb = gpu_size / 1e6
        # More verbose logging
        log(
            INFO,
            "get_referenced_tensors_summary :: there are %s "
            "referenced tensors for a total size of %s MiB "
            "(%s MiB on GPU, %s MiB on CPU). Summary is:\n%s",
            counter,
            total_size_mb,
            gpu_size_mb,
            total_size_mb - gpu_size_mb,
            summary,
        )
        # # Less verbose logging
        # log(
        #     INFO,
        #     "get_referenced_tensors_summary :: there are %s"
        #     "referenced tensors for a size of %s",
        #     counter,
        #     total_size,
        # )
    return summary


def get_referenced_trainers_and_engines(verbose: bool = True) -> str:
    """Inspect the tensors in the current Python session."""
    # Initalizing the summary string and variables
    summary = ""
    n_trainers, n_engines = 0, 0
    gc.collect()
    # Looping over the objects in the current Python session
    for obj in gc.get_objects():
        # Surrounding the tensor inspection with a try-except block
        try:
            # Checking if the object is a tensor or a tensor data attribute
            if type(obj) is Trainer:
                n_trainers += 1
            if type(obj) is Engine:
                n_engines += 1
        except Exception:
            # log(
            #     ERROR,
            #     "get_referenced_tensors_summary :: error while inspecting ",
            #     "object of type %s",
            #     type(obj),
            #     exc_info=e,
            #     stack_info=True,
            # )
            pass
    if verbose:
        log(
            INFO,
            "get_referenced_trainers_and_engines ::"
            "there are %s engines and %s trainers",
            n_engines,
            n_trainers,
        )
    return summary


# NOTE: This doesn't work as expected
def force_referenced_tensors_destruction() -> None:
    """Force destruction of the tensors in the current Python session."""
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                try:
                    if obj.is_cuda:
                        obj.to("cpu")
                    # referrers = gc.get_referrers(obj)
                    # for referrer in referrers:
                    #     parent_referrers = gc.get_referrers(referrer)
                    #     for p_r in parent_referrers:
                    #         del p_r
                    #     del referrer
                    del obj
                except Exception as e:
                    log(
                        INFO,
                        "force_referenced_tensors_destruction ::"
                        "error while deleting object of type %s: %s",
                        type(obj),
                        e,
                    )
        except Exception:
            # log(
            #     ERROR,
            #     "get_referenced_tensors_summary :: error while inspecting ",
            #     "object of type %s",
            #     type(obj),
            #     exc_info=e,
            #     stack_info=True,
            # )
            pass
    log(INFO, "force_referenced_tensors_destruction :: done")
    gc.collect()
    torch.cuda.empty_cache()


def clean_trainer_state(trainer: Trainer) -> None:
    """Clean the state of the trainer."""
    # for attribute_name in trainer.state.serialized_attributes:
    #     current_attr = getattr(trainer.state, attribute_name)
    #     log(INFO, "State's attribute %s: %s", attribute_name, current_attr)
    try:
        trainer.state.model.cpu()
        delattr(trainer.state, "model")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        for optimizer in trainer.state.optimizers:
            del optimizer
        delattr(trainer.state, "_optimizers")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        for scheduler in trainer.state.schedulers:
            del scheduler
        delattr(trainer.state, "_schedulers")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        for callback in trainer.state.callbacks:
            del callback
        delattr(trainer.state, "_callbacks")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer.state, "scaler")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer.state, "timestamp")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        for _k, t_m in trainer.state.train_metrics.items():
            t_m.cpu()
            del t_m
        delattr(trainer.state, "train_metrics")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        for _k, e_m in trainer.state.eval_metrics.items():
            for _kk, ee_m in e_m.items():
                ee_m.cpu()
                del ee_m
            del e_m
        delattr(trainer.state, "eval_metrics")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer.state, "batch")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer.state, "loss")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer.state, "outputs")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer, "state")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer, "engine")
    except Exception as e:
        log(WARN, "Exception %s", e)
    try:
        delattr(trainer, "_original_model")
    except Exception as e:
        log(WARN, "Exception %s", e)


class IntentionalClientDropout(Exception):
    """Exception raised when a client is dropped out of the tree."""
