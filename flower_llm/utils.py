"""Utility functions for FL and experiment management.

They assure compatibility with the Flower and wandb APIs.
"""

from dataclasses import dataclass
import fcntl
import gc
import os
import pickle
import re
import resource
import shutil
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Generator, Sequence
from logging import DEBUG, ERROR
from pathlib import Path
from typing import Any, Literal, cast

from composer.loggers import RemoteUploaderDownloader

import numpy as np
import psutil
import pyarrow as pa
import ray
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
from composer import Trainer
from composer.utils import dist
from flwr.common import Config, NDArrays, log, parameters_to_ndarrays
from torch import device as device_type
from typing_extensions import Self
from composer.utils.file_helpers import list_remote_objects

import wandb


# NOTE: Setting the maximum value according to the documentation
# https://github.com/grpc/grpc/blob/eeae8e635a896bfa420d21e476221af652fd9986/include/grpc/impl/codegen/grpc_types.h#L150
POLLEN_LLM_MAX_MESSAGE_LENGTH = -1


@dataclass
class ClientState:
    """Dataclass for client state."""

    local_steps_cumulative: int
    steps_done: int = 0


class NoOpContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:
        """Do nothing."""
        return

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Do nothing."""


def parameters_checker(
    current_parameters: NDArrays, reference_parameters: NDArrays, is_equal: bool = False
) -> None:
    """Checker trainer's parameters (in)compatibility with the given parameters."""
    list_of_conditions = []
    for i, (current_param, param) in enumerate(
        zip(current_parameters, reference_parameters, strict=True)
    ):
        # NOTE: Skip check if the size of the `current_param` is 0
        if current_param.size == 0:
            continue
        # Skip ranks > 0 b/c they are not meant to be consistent
        if int(os.getenv("LOCAL_RANK", "-1")) > 0:
            continue
        # Reshape the parameters if the shapes are not equal, which happens for
        # flattened parameters in FSDP)
        if current_param.shape != param.shape:
            try:
                current_param = current_param.reshape(param.shape)
            except Exception as e:
                log(
                    ERROR,
                    "Error in reshaping parameter, Rank %s, Component %s,"
                    " Trainer shape %s, Param shape %s",
                    int(os.getenv("LOCAL_RANK", "-1")),
                    i,
                    current_param.shape,
                    param.shape,
                    exc_info=e,
                )
                # If the reshaping fails, skip the check assuming split tensor by FSDP
                continue
        # Assert the shape of the parameters are equal
        assert current_param.shape == param.shape
        # Append the condition to the list of conditions
        list_of_conditions.append(np.array_equal(current_param, param))
    # Assert all the conditions are true if `is_equal` is True
    local_rank = int(os.getenv("LOCAL_RANK", "-1"))
    if is_equal:
        list_of_conditions.append(True)
        assert all(
            list_of_conditions
        ), f"Parameters on rank {local_rank} are not equal: {list_of_conditions}"
    # Assert not all the conditions are true (at least one is False) if `is_equal` is
    # False
    else:
        list_of_conditions.append(False)
        assert not all(
            list_of_conditions
        ), f"Parameters on rank {local_rank} are not different: {list_of_conditions}"


def get_parameters_from_state(config: Config, trainer: Trainer) -> NDArrays:
    """Implement how to get parameters."""
    model_parameters_dict = get_trainable_params_dict(trainer.state.model)
    return [val.detach().to("cpu").numpy() for _, val in model_parameters_dict.items()]


def get_trainable_params_dict(
    model: torch.nn.Module, sort_dict: bool = True
) -> dict[str, torch.nn.Parameter] | dict[str, torch.Tensor]:
    """Get the trainable parameters of a model as a dictionary."""
    params_dict: dict[str, torch.nn.Parameter] | dict[str, torch.Tensor] = {}
    # NOTE: This function is weird because the encapsulation done to support FSDP and
    # DDP is weird. Since they are both likely to change, we MUST maintain this very
    # well and implement as many checkers as we can.
    if hasattr(model, "model") and type(model.model) is FullyShardedDataParallel:
        assert model.model is not None
        inner_model = model.model
        # NOTE: This doesn't work in the case in use_orig_params is True if the FSDP
        # configuration as the tensors returned are flattened breaking some assumptions
        # of the rest of the codebase
        with FullyShardedDataParallel.summon_full_params(
            inner_model,
            recurse=True,
            writeback=False,
            rank0_only=True,
            offload_to_cpu=True,
            with_grads=False,
        ):
            # NOTE: This parameter dict using the above parameters, i.e., (recurse=True,
            # writeback=False, rank0_only=True, offload_to_cpu=True, with_grads=False,),
            # will be complete only on rank 0. The other ranks will have zero-shaped
            # tensors for those layers that are not "living" in there.
            # NOTE: If the FSDP configuration use the original parameters
            # (use_orig_params=true), then the tensors in rank 0 have the correct
            # original shape. In the other ranks they are flattened anyway.
            # NOTE: On ranks > 0 the dictionary won't be empty. It will contain the
            # parameters that are "living" in that rank and will have zero-shaped
            # tensors for the others.
            params_dict = {
                name: param.detach().clone()
                for name, param in inner_model.named_parameters()
                if param.requires_grad
            }
    else:
        params_dict = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
    if sort_dict:
        params_dict = dict(sorted(params_dict.items()))
    dist.barrier()
    return params_dict


def set_trainer_trainable_params_dict(
    trainer: Trainer,
    parameters_dict: OrderedDict[str, torch.Tensor],
) -> None:
    """Set the trainable parameters of a model."""
    # NOTE: This function is weird because the encapsulation done to support FSDP and
    # DDP is weird. Since they are both likely to change, we MUST maintain this very
    # well and implement as many checkers as we can.
    if (
        hasattr(trainer.state.model, "model")
        and type(trainer.state.model.model) is FullyShardedDataParallel
    ):
        # Get the state dict of the model on rank 0 offloading to CPU
        # NOTE: This assumes there's enough RAM on rank 0 to hold the model state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            trainer.state.model.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = trainer.state.model.model.state_dict()
            # If the state dict exists (only on rank 0), modify ion place the parameters
            # to those passed as argument
            if cpu_state:
                # Set the parameters only if they require gradients
                for name, param in cpu_state.items():
                    # NOTE: We need to add the prefix "model." to the name of the
                    # parameter to match the state dict
                    cpu_state[name] = parameters_dict["model." + name].to(param.device)
            # Broadcast the state dict across all ranks
            # NOTE: This step is necessary as all the ranks must load the same state
            # dict concurrently
            list_of_objects = [cpu_state]
            dist.broadcast_object_list(list_of_objects, src=0)
            # Load the state dict back to the model
            trainer.state.model.model.load_state_dict(list_of_objects[0])
    else:
        for name, param in trainer.state.model.named_parameters():
            # Set the parameters only if they require gradients
            if param.requires_grad:
                # DDP
                if name.startswith("module."):
                    param.data = parameters_dict[name.replace("module.", "")].to(
                        param.device
                    )
                # Single GPU
                else:
                    param.data = parameters_dict[name].to(param.device)
    dist.barrier()


# NOTE: This is unused but it is kept for reference
def set_trainable_params_dict(
    model: torch.nn.Module,
    parameters_dict: OrderedDict[str, torch.Tensor],
) -> None:
    """Set the trainable parameters of a model."""
    # NOTE: This function is weird because the encapsulation done to support FSDP and
    # DDP is weird. Since they are both likely to change, we MUST maintain this very
    # well and implement as many checkers as we can.
    if hasattr(model, "model") and type(model.model) is FullyShardedDataParallel:
        assert model.model is not None
        inner_model = model.model
        # NOTE: This doesn't work in the case in use_orig_params is True if the FSDP
        # configuration as the tensors returned are flattened breaking some assumptions
        # of the rest of the codebase
        with FullyShardedDataParallel.summon_full_params(
            inner_model,
            recurse=True,
            # Writing back is not compatible with rank 0 only
            writeback=True,
            rank0_only=False,
            # Prevent moving to CPU device
            offload_to_cpu=False,
            with_grads=False,
        ):
            # NOTE: !!! THIS REQUIRES INVESTIGATION AS IT DOESN'T WORK AS EXPECTED !!!
            # NOTE: This parameter dict using the above parameters, i.e.,
            # (recurse=True, writeback=True, rank0_only=False, offload_to_cpu=False,
            # with_grads=False,), won't be complete in any rank if the model i sharded.
            # Each rank will have zero-size tensors for those layers that are not
            # "living" in there and the flattened/unflatten complete tensors for those
            # blocks living there.
            # NOTE: If the FSDP configuration use the original parameters
            # (use_orig_params=true), then the tensors in rank 0 have the correct
            # original shape. In the other ranks they are flattened anyway.
            for name, param in inner_model.named_parameters():
                # Set the parameters only if they require gradients & have non-zero size
                if param.requires_grad and param.size != 0:
                    param.data = parameters_dict[name].to(param.device)
    else:
        for name, param in model.named_parameters():
            # Set the parameters only if they require gradients
            if param.requires_grad:
                # NOTE: DDP pre-pends "module." to the name of the parameter
                if name.startswith("module."):
                    param.data = parameters_dict[name.replace("module.", "")].to(
                        param.device
                    )
                # Single GPU
                else:
                    param.data = parameters_dict[name].to(param.device)
    dist.barrier()


def get_list_of_parameters_names(
    model: torch.nn.Module, sort_dict: bool = True
) -> list[str]:
    """Return the list of parameters names."""
    params_dict = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    if sort_dict:
        params_dict = dict(sorted(params_dict.items()))
    return list(params_dict.keys())


def construct_parameters_dict(
    parameters_names: list[str], parameters: NDArrays
) -> OrderedDict[str, torch.Tensor]:
    """Construct a dictionary of parameters."""
    zipped_lists = zip(parameters_names, parameters, strict=True)
    return OrderedDict({k: torch.as_tensor(v) for k, v in zipped_lists})


def download_file_from_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path | str,
) -> None:
    """Download a file from S3."""
    remote_up_down._check_workers()
    remote_up_down.download_file(
        remote_file_name=remote_file_name,
        destination=str(local_file_name),
        overwrite=True,
    )


def upload_file_to_s3(
    remote_up_down: RemoteUploaderDownloader,
    remote_file_name: str,
    local_file_name: Path,
) -> None:
    """Download a file from S3."""
    remote_up_down._check_workers()
    remote_up_down.upload_file(
        state=None,
        remote_file_name=remote_file_name,
        file_path=local_file_name,
        overwrite=True,
    )


def load_model_parameters_from_file(file_path: Path) -> NDArrays:
    """Load model parameters from a file."""
    if file_path.suffix in {".npz", ".npzc"}:
        with np.load(file_path) as data:
            return [data[key] for key in data.files]
    elif file_path.suffix == ".bin":
        with open(file_path, "rb") as file:
            return parameters_to_ndarrays(pickle.load(file))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def dump_model_parameters_to_file(file_path: Path, model_parameters: NDArrays) -> None:
    """Load model parameters from a file."""
    # NOTE: Very slow for big models b/c compression. Good benchmark available here: https://stackoverflow.com/questions/30329726/fastest-save-and-load-options-for-a-numpy-array
    if file_path.suffix == ".npzc":
        with open(file_path, "wb") as file:
            np.savez_compressed(file, *model_parameters)
    elif file_path.suffix == ".bin":
        with open(file_path, "wb") as file:
            pickle.dump(model_parameters, file)
    elif file_path.suffix == ".npz":
        with open(file_path, "wb") as file:
            np.savez(file, *model_parameters)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


# Client ####
# General
def get_parameters(net: torch.nn.Module) -> NDArrays:
    """Implement generic `get_parameters` for Flower Client."""
    net.eval()
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(
    net: torch.nn.Module, parameters: NDArrays, device: str = "cpu"
) -> None:
    """Implement generic `set_parameters` for Flower Client."""
    net.eval()
    model_parameters_dict = get_trainable_params_dict(net)
    params_dict = zip(model_parameters_dict.keys(), parameters, strict=True)
    state_dict = OrderedDict(
        {k: torch.as_tensor(v, device=device) for k, v in params_dict}
    )
    net.load_state_dict(state_dict=state_dict, strict=False)
    del state_dict


def invert_many_to_one_dictionary(
    input_dict: dict,
) -> dict:
    """Invert the mapping given by a dictionary when it is many-to-one."""
    output: dict = defaultdict(list)
    for k, v in input_dict.items():
        output[v] = output.get(v, []) + [k]
    return output


def invert_one_to_many_dictionary(
    input_dict: dict,
) -> dict:
    """Invert the mapping given by a dictionary when it is one-to-many."""
    output: dict = {}
    for k, v in input_dict.items():
        for w in v:
            output[w] = k
    return output


def wandb_init(
    wandb_enabled: bool, *args: dict, **kwargs: dict
) -> NoOpContextManager | Any | None:
    """Initialize wandb if enabled."""
    if wandb_enabled:
        # Add server suffix to the name of the run
        name = kwargs.pop("name", "")
        assert type(name) is str, f"Name must be a string, not {type(name)}"
        name += "_server"
        return wandb.init(*args, **kwargs, name=name)  # type: ignore[arg-type,misc]

    return NoOpContextManager()


class RayContextManager:
    """A context manager for cleaning up after ray."""

    def __enter__(self) -> Self:
        """Initialize the context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Cleanup the files."""
        if ray.is_initialized():
            temp_dir = Path(ray.worker._global_node.get_session_dir_path())
            ray.shutdown()
            directory_size = shutil.disk_usage(temp_dir).used
            shutil.rmtree(temp_dir)
            log(
                DEBUG,
                f"Cleaned up ray temp session: {temp_dir} with size:{directory_size}",
            )


def chunks_idx(
    list_of_stuff: Sequence, n_chunks: int
) -> Generator[tuple[int, int], Any, None]:
    """Split a list in n_chunks of equal length."""
    d, r = divmod(len(list_of_stuff), n_chunks)
    for i in range(n_chunks):
        si = (d + 1) * (min(r, i)) + d * (0 if i < r else i - r)
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


def sum_of_squares(arrays: NDArrays) -> float:
    """Compute the sum of squares of a list of arrays.

    Parameters
    ----------
    arrays : NDArrays
        List of arrays to compute the sum of squares of.

    Returns
    -------
    float
        The sum of squares of the list of arrays.
    """
    return sum(np.sum(np.square(arr)) for arr in arrays)


def l2_norm(arrays: NDArrays) -> float:
    """Compute the L2 norm of a list of arrays.

    Parameters
    ----------
    arrays : NDArrays
        List of arrays to compute the L2 norm of.

    Returns
    -------
    float
        The L2 norm of the list of arrays.
    """
    return float(np.sqrt(sum_of_squares(arrays)))


def get_device() -> device_type:
    """Determine which device to use for PyTorch.

    Returns
    -------
        str: device for PyTorch
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
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
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore[reportArgumentType]
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


def namestr(obj: object, namespace: dict) -> list:
    """Return the name of an object in the given namespace."""
    return [name for name in namespace if namespace[name] is obj]


def get_referenced_tensors_summary(cuda_only: bool = True, verbose: bool = True) -> str:
    """Inspect the tensors in the current Python session."""
    # Initializing the summary string and variables
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
            DEBUG,
            "get_referenced_tensors_summary :: there are %s "
            "referenced tensors for a total size of %s MiB "
            "(%s MiB on GPU, %s MiB on CPU).",
            # "Summary is:\n%s",
            counter,
            total_size_mb,
            gpu_size_mb,
            total_size_mb - gpu_size_mb,
            # summary,
        )
        # # Less verbose logging
        # log(
        #     DEBUG,
        #     "get_referenced_tensors_summary :: there are %s"
        #     "referenced tensors for a size of %s",
        #     counter,
        #     total_size,
        # )
    return summary


def get_selected_objects_types(
    selection: list[str], second_selection: list[str], verbose: bool = True
) -> str:
    """Inspect the tensors in the current Python session."""
    # Initializing the summary string and variables
    summary = ""
    gc.collect()
    # Looping over the objects in the current Python session
    for obj in gc.get_objects():
        # Surrounding the tensor inspection with a try-except block
        try:
            if any(s in f"{type(obj)}" for s in selection):
                summary += f"{type(obj)}"
                if any(s in f"{type(obj)}" for s in second_selection):
                    referrers = gc.get_referrers(obj)
                    summary += f" :: {len(referrers)}"
                    # filename = f"{type(obj)}.png".replace(" ", "_")
                    # backref_filename = f"backref{type(obj)}.png".replace(" ", "_")
                    # objgraph.show_refs(obj, filename=filename)
                    # objgraph.show_backrefs(obj, filename=backref_filename)
                    # if "state" in f"{type(obj)}":
                    #     referrers = gc.get_referrers(obj)
                    #     for referrer in referrers:
                    #         if "tuple" in f"{type(referrer)}":
                    #             ref_ref = gc.get_referrers(referrer)
                    #             summary += f" :: {len(ref_ref)}"
                    #             for i, ref_referrer in enumerate(ref_ref):
                    #                 tuple_fn = f"{i}tuple_{type(obj)}.png"
                    #                 tuple_fn.replace(" ", "_")
                    #                 tuple_bkref_fn = f"{i}tuple_bkref{type(obj)}.png"
                    #                 tuple_bkref_fn.replace(" ", "_")
                    #                 objgraph.show_refs(
                    #                     ref_referrer, filename=tuple_fn
                    #                 )
                    #                 objgraph.show_backrefs(
                    #                     ref_referrer, filename=tuple_bkref_fn
                    #                 )
                    # referrers = gc.get_referrers(obj)
                    # summary += f" :: {len(referrers)}"
                    # for referrer in referrers:
                    #     if any(s in f"{type(referrer)}" for s in selection):
                    #         summary += f" :: {type(referrer)}"
                summary += "\n"
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
            DEBUG,
            "get_objects_types :: %s",
            summary,
        )
    return summary


def clean_trainer_state(trainer: Trainer) -> None:
    """Clean the state of the trainer."""
    # Evaluators
    try:
        for evaluator in trainer.state._evaluators:
            iterator = evaluator.dataloader.dataloader._iterator
            try:
                # Shut down the workers of the evaluation dataloaders, if any,
                # to avoid memory leaks
                iterator._shutdown_workers()
            except AttributeError:
                pass
            except Exception as e:
                log(
                    ERROR,
                    "Error running evaluator(s).dataloader.dataloader"
                    "._iterator._shutdown_workers().",
                    exc_info=e,
                    stack_info=True,
                )
        for evaluator in trainer.state._evaluators:
            try:
                # Delete the evaluation dataloaders, if any
                delattr(evaluator, "dataloader")
            except AttributeError:
                pass
            except Exception as e:
                log(
                    ERROR,
                    'Error running `delattr(evaluator, "dataloader")`',
                    exc_info=e,
                    stack_info=True,
                )
        # Delete the evaluators, if any
        delattr(trainer.state, "_evaluators")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "_evaluators")`',
            exc_info=e,
            stack_info=True,
        )
    # State
    # Model
    try:
        trainer.state.model.cpu()
        delattr(trainer.state, "model")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "model")`',
            exc_info=e,
            stack_info=True,
        )
    # Latest dataloader used
    try:
        delattr(trainer.state, "_dataloader")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "_dataloader")`',
            exc_info=e,
            stack_info=True,
        )
    # Train dataloader
    try:
        delattr(trainer.state, "_train_dataloader")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "_train_dataloader")`',
            exc_info=e,
            stack_info=True,
        )
    # Optimizers
    try:
        for optimizer in trainer.state.optimizers:
            try:
                delattr(optimizer, "state")
            except AttributeError:
                pass
            except Exception as e:
                log(
                    ERROR,
                    'Error running `delattr(optimizer, "state")`',
                    exc_info=e,
                    stack_info=True,
                )
            for p_g in optimizer.params_groups:
                p_g.cpu()
                del p_g
            try:
                delattr(optimizer, "param_groups")
            except AttributeError:
                pass
            except Exception as e:
                log(
                    ERROR,
                    'Error running `delattr(optimizer, "param_groups")`',
                    exc_info=e,
                    stack_info=True,
                )
            del optimizer
        delattr(trainer.state, "_optimizers")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "_optimizers")`',
            exc_info=e,
            stack_info=True,
        )
    # Schedulers
    try:
        for scheduler in trainer.state.schedulers:
            del scheduler
        delattr(trainer.state, "_schedulers")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "_schedulers")`',
            exc_info=e,
            stack_info=True,
        )
    # Callbacks
    try:
        for callback in trainer.state.callbacks:
            del callback
        delattr(trainer.state, "_callbacks")
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "model")`',
            exc_info=e,
            stack_info=True,
        )
    # Scaler
    try:
        delattr(trainer.state, "scaler")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "scaler")`',
            exc_info=e,
            stack_info=True,
        )
    # Timestamp
    try:
        delattr(trainer.state, "timestamp")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "timestamp")`',
            exc_info=e,
            stack_info=True,
        )
    # Train metrics
    try:
        if trainer.state.train_metrics is not None:
            for t_m in trainer.state.train_metrics.values():
                t_m.cpu()
                del t_m
            delattr(trainer.state, "train_metrics")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "train_metrics")`',
            exc_info=e,
            stack_info=True,
        )
    # Eval metrics
    try:
        for e_m in trainer.state.eval_metrics.values():
            for ee_m in e_m.values():
                ee_m.cpu()
                del ee_m
            del e_m
        delattr(trainer.state, "eval_metrics")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "eval_metrics")`',
            exc_info=e,
            stack_info=True,
        )
    # Latest batch
    try:
        delattr(trainer.state, "batch")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "batch")`',
            exc_info=e,
            stack_info=True,
        )
    # Latest loss
    try:
        delattr(trainer.state, "loss")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "loss")`',
            exc_info=e,
            stack_info=True,
        )
    # Latest outputs
    try:
        delattr(trainer.state, "outputs")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.state, "outputs")`',
            exc_info=e,
            stack_info=True,
        )
    # State object
    try:
        delattr(trainer, "state")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer, "state")`',
            exc_info=e,
            stack_info=True,
        )
    # Engine
    # Logger
    try:
        delattr(trainer.engine, "logger")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.engine, "logger")`',
            exc_info=e,
            stack_info=True,
        )
    # State
    try:
        delattr(trainer.engine, "state")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.engine, "state")`',
            exc_info=e,
            stack_info=True,
        )
    # Engine object
    try:
        delattr(trainer, "engine")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer, "engine")`',
            exc_info=e,
            stack_info=True,
        )
    # Trainer
    # Model
    try:
        delattr(trainer, "_original_model")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer, "_original_model")`',
            exc_info=e,
            stack_info=True,
        )
    # Start batch for checkpoint saver
    try:
        delattr(trainer._checkpoint_saver, "start_batch")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer._checkpoint_saver, "start_batch")`',
            exc_info=e,
            stack_info=True,
        )
    # Checkpoint saver
    try:
        delattr(trainer, "_checkpoint_saver")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer, "_checkpoint_saver")`',
            exc_info=e,
            stack_info=True,
        )
    # State
    try:
        delattr(trainer.logger, "_state")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer.logger, "_state")`',
            exc_info=e,
            stack_info=True,
        )
    # Logger
    try:
        delattr(trainer, "logger")
    except AttributeError:
        pass
    except Exception as e:
        log(
            ERROR,
            'Error running `delattr(trainer, "logger")`',
            exc_info=e,
            stack_info=True,
        )


def get_open_fds() -> list[int]:
    """Return the list of open file descriptors."""
    fds = []
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    for fd in range(3, soft):
        try:
            fcntl.fcntl(fd, fcntl.F_GETFD)
        except OSError:
            continue
        fds.append(fd)
    return fds


def get_file_names_from_file_number(fds: list[int]) -> list[str]:
    """Return a list of file names given a list of file descriptor numbers."""
    names = []
    for fd in fds:
        names.append(str(Path.readlink(Path("/proc/self/fd/%d" % fd))))
    return names


class Capture:
    """Class for capturing the arguments passed to a function.

    Usage:
    >>> c = Capture()
    >>> import atexit
    >>> atexit.unregister(c)
    >>> print(f"Captured arguments: {c.captured}")
    """

    def __init__(self) -> None:
        self.captured: list[Callable | object] = []

    def __eq__(self, other: object) -> Literal[False]:
        """Override the `__eq__` function to always return False and append."""
        self.captured.append(other)
        return False

    def __hash__(self) -> int:
        """Override the `__hash__` function to return the hash of the captured."""
        return hash(self.captured)


class IntentionalClientDropoutError(Exception):
    """Exception raised when a client is dropped out of the tree."""


def obtain_sorted_runs(server_path: str) -> list[int]:
    """Obtain the sorted runs from the server path.

    Parameters
    ----------
    server_path : str
        The path to the server.

    Returns
    -------
    List[int]
        The sorted runs.
    """
    remote_objects = list_remote_objects(server_path)
    log(DEBUG, "Found files %s", remote_objects)
    # Take only the unique indices
    return sorted(
        {
            int(reg.group(1))
            for path in remote_objects
            if (reg := re.search(r"server/(\d+)/.*$", path)) is not None
        }
    )


def create_remote_up_down(
    bucket_name: str,
    prefix: str,
    run_uuid: str,
    num_attempts: int,
    client_config: dict[str, Any],
    num_concurrent_uploads: int = 1,
    upload_staging_folder: str | None = None,  # Don't touch, it's /tmp by default
    use_procs: bool = True,
) -> RemoteUploaderDownloader:
    """Create the remote uploader/downloader.

    Parameters
    ----------
    bucket_name : str
        The name of the bucket.
    run_uuid : str
        The UUID of the run.
    num_attempts : int
        The number of attempts.
    client_config : dict[str, Any]
        The configuration of the client.
    num_concurrent_uploads : int, optional
        The number of concurrent uploads, by default 1.
    upload_staging_folder : str | None, optional
        The upload staging folder, dont't touch, by default None.
    use_procs : bool, optional
        Whether to use processes, by default True. Don't touch.

    Returns
    -------
    RemoteUploaderDownloader
        The remote uploader/downloader.
    """
    bucket_uri = f"s3://{bucket_name}"
    remote_up_down = RemoteUploaderDownloader(
        bucket_uri=bucket_uri,
        backend_kwargs={
            "bucket": bucket_name,
            "prefix": prefix,  # Don't touch
            "region_name": None,  # Not necessary
            "endpoint_url": None,  # Will be read from env var
            "aws_access_key_id": None,  # Will be read from config file
            "aws_secret_access_key": None,  # Will be read from config file
            "aws_session_token": None,  # Will be automatically generated
            "client_config": client_config,  # And using defaults
            "transfer_config": None,  # Using defaults
        },
        file_path_format_string="{remote_file_name}",  # Don't touch
        num_concurrent_uploads=num_concurrent_uploads,
        upload_staging_folder=upload_staging_folder,  # Don't touch, default: /tmp
        use_procs=use_procs,  # Don't touch
        num_attempts=num_attempts,
    )
    remote_up_down.init(run_name=run_uuid)  # Don't touch
    return remote_up_down
