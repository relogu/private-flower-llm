"""Provides functionality for manipulating MosaicML configs."""

import ast
import os
from logging import DEBUG, INFO, WARN, WARNING
import re
from typing import Any

import torch
from composer.utils.file_helpers import list_remote_objects
from composer.devices import DeviceGPU, DeviceCPU, Device
from flwr.common.logger import log


from omegaconf import DictConfig, ListConfig


from flower_llm.utils import (
    get_n_cpu_cores,
    get_n_cuda_devices,
)
from dataclasses import dataclass, asdict
import operator


@dataclass
class StreamDict:
    """Dataclass for stream dictionary."""

    remote: str | None = None
    local: str | None = None
    split: str | None = None
    proportion: float | None = None
    repeat: float | None = None
    choose: int | None = None
    download_retry: int | None = None
    download_timeout: float | None = None
    validate_hash: str | None = None
    keep_zip: bool | None = None


def set_icl_tasks_root_dir(icl_tasks_listconfig: ListConfig, root_dir: str) -> None:
    """Update the dataset URI for each ICL task in the given ListConfig.

    The update is performed by prepending the specified root directory.

    Parameters
    ----------
        icl_tasks_listconfig : ListConfig
            A ListConfig object containing ICL tasks, each with a `dataset_uri` attr.
        root_dir : str
            The root directory to prepend to each task's `dataset_uri`.

    Returns
    -------
        None
    """
    for icl_task in icl_tasks_listconfig:
        old_dataset_uri = icl_task.dataset_uri
        icl_task.dataset_uri = root_dir + "/" + old_dataset_uri


def client_set_data_config(cid: int | str | None, cfg: DictConfig) -> None:
    """Set the client data configuration for the client.

    Parameters
    ----------
    cid : int | str | None
        The client id.
    cfg : DictConfig
        The configuration object.

    Returns
    -------
    DictConfig
        The updated configuration object.
    """
    # Retrieve the train config to construct the dataset for the train loader
    dataset_config: DictConfig
    for split in ["train", "val"]:
        if split == "train":
            dataset_config = cfg.train_loader.dataset
        elif split == "val":
            dataset_config = cfg.eval_loader.dataset
        else:
            raise ValueError(f"Split {split} is not supported.")
        # Get the root path for remote and local data
        root_remote = dataset_config.pop("root_remote", "")
        root_remote = root_remote + "/" if root_remote else root_remote
        root_local = dataset_config.pop("root_local", "")
        root_local = root_local + "/" if root_local else root_local
        split = dataset_config.pop("split", "")
        # Get the clients streams available
        clients_streams = dataset_config.streams
        # Extract the current client train stream -- it contains a dict of buckets
        # NOTE: Here, we circumvent the possible limited size of the number of client
        # streams since it should have been handled elsewhere
        current_client_stream: dict[str, Any] = {}
        if cid is not None:
            current_client_stream |= clients_streams[int(cid) % len(clients_streams)][
                "client_streams"
            ]
        else:
            # Concatenate all the streams
            counter = 0
            for client_stream in clients_streams:
                assert "client_streams" in client_stream
                client_streams = client_stream["client_streams"]
                assert isinstance(client_streams, DictConfig)
                for stream in client_streams.values():
                    current_client_stream |= {f"stream_{counter}": stream}
                    counter += 1
        # Set streams dictionary for the train loader
        actual_streams = {
            key: StreamDict(**value) for key, value in current_client_stream.items()
        }
        # Propagate the split and the remote and local paths to each stream
        for stream in actual_streams.values():
            # Set the split, remote, and local paths
            stream.split = split or stream.split
            if root_local:
                stream.local = root_local + stream.local if stream.local else root_local
            if root_remote:
                stream.remote = (
                    root_remote + stream.remote if stream.remote else root_remote
                )
            # Remove potential trailing slashes
            stream.local = stream.local.rstrip("/") if stream.local else stream.local
            stream.remote = (
                stream.remote.rstrip("/") if stream.remote else stream.remote
            )
        # Convert the streams to dictionaries
        streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}
        # Assign the streams to the appropriate loaders
        if split == "train":
            cfg.train_loader.dataset.streams = streams_dict
        elif split == "val":
            cfg.eval_loader.dataset.streams = streams_dict


def set_dataset_default_params(cfg: DictConfig) -> None:
    """Set the default parameters for the dataset."""
    # Set the `pre-download` value as 8*batch_size
    if cfg.train_loader.dataset.get("predownload", None) is None:
        cfg.train_loader.dataset.predownload = 8 * cfg.global_train_batch_size
    if cfg.eval_loader.dataset.get("pre_download", None) is None:
        cfg.eval_loader.dataset.predownload = 8 * cfg.device_eval_batch_size
    # NOTE: Set the `num_canonical_nodes` value as 64*`num_physical_nodes`, assuming
    # that we will always have just 1 real node (server)
    if cfg.train_loader.dataset.get("num_canonical_nodes", None) is None:
        cfg.train_loader.dataset.num_canonical_nodes = 64 * 1
    if cfg.eval_loader.dataset.get("num_canonical_nodes", None) is None:
        cfg.eval_loader.dataset.num_canonical_nodes = 64 * 1
    # Set the `shuffle_block_size` value as 8*batch_size
    if cfg.train_loader.dataset.get("shuffle_block_size", None) is None:
        cfg.train_loader.dataset.shuffle_block_size = max(
            4_000_000 // cfg.train_loader.dataset.num_canonical_nodes, 1 << 18
        )
    if cfg.eval_loader.dataset.get("shuffle_block_size", None) is None:
        cfg.eval_loader.dataset.shuffle_block_size = max(
            4_000_000 // cfg.eval_loader.dataset.num_canonical_nodes, 1 << 18
        )


def set_client_save_and_load_path(cfg: DictConfig, cid: int | str) -> None:
    """Set the save and load path given the server round and client id."""
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        cfg.save_folder = (  # type: ignore[union-attr]
            cfg.save_folder
            + "/client_"  # type: ignore[union-attr]
            + str(cid)  # type: ignore[union-attr]
        )
        log(DEBUG, "Set save folder: %s", cfg.save_folder)


def set_client_load_path(cfg: DictConfig, cid: int | str, n_steps: int) -> bool:
    """Set the save and load path given the server round and client id."""
    # Set client load path
    set_client_save_and_load_path(cfg, cid)
    # Flag to notify whether to skip this iteration or not
    skip_iteration = False
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        try:
            # Are there any checkpoints?
            remote_objects = list_remote_objects(cfg.save_folder)
            if not remote_objects:
                log(
                    INFO,
                    "No checkpoints found in %s. Starting training from scratch.",
                    cfg.save_folder,
                )
                assert cfg.load_path is None
                return skip_iteration
            # NOTE: We always need to check all of the checkpoints
            # Given the epoch change
            # As such we extract the epoch number and number of batches
            # The number of epochs
            sorted_pairs = sorted(
                [
                    (
                        int(reg.group(1)),  # epoch number
                        int(reg.group(2)),  # number of batches
                    )
                    for path in remote_objects
                    if (
                        reg := re.search(
                            r"client_" + str(cid) + r"/ep(\d+)-ba(\d+)", path
                        )
                    )
                    is not None
                ],
                key=operator.itemgetter(1),
            )

            log(
                INFO,
                "Found the following sorted checkpoint epochs and batches: %s",
                sorted_pairs,
            )

            # Is there the next checkpoint?
            log(INFO, "Looking for the next checkpoint in %s", cfg.save_folder)
            # See if we have a checkpoint with a matching number of steps
            path_to_check = next(
                (
                    (epoch, batches)
                    for epoch, batches in sorted_pairs
                    if batches == n_steps
                ),
                None,
            )
            # NOTE: ruff is not bright and cannot see through the condition
            skip_iteration = path_to_check is not None
            if skip_iteration and path_to_check is not None:
                epoch, batches = path_to_check
                cfg.load_path = (
                    cfg.save_folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
                )
                log(
                    INFO,
                    "Skipping training iteration as checkpoint %s already exists.",
                    cfg.load_path,
                )
                # NOTE: Don't re-save the checkpoint when resuming mid-round
                cfg.save_folder = None
                return skip_iteration
            # Load the latest checkpoint
            log(
                INFO, "Looking for the latest checkpoint to load in %s", cfg.save_folder
            )
            epoch, batches = sorted_pairs[-1]
            cfg.load_path = (
                cfg.save_folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
            )
            log(INFO, "Set checkpoint to load: %s", cfg.load_path)
        except Exception as e:
            log(WARNING, "The `load_path` wasn't set.", exc_info=e, stack_info=True)
    return skip_iteration


def set_client_wandb_logger(cfg: DictConfig, log_name: str) -> None:
    """Set the wandb logger for the client."""
    # Set the wandb run name
    if cfg.loggers is not None and cfg.loggers.wandb is not None:
        # Get the server run name
        run_name = cfg.loggers.wandb.init_kwargs.name
        # Add the client id to the run name
        new_run_name = run_name + f"{log_name}"
        server_id = cfg.loggers.wandb.init_kwargs.id
        cfg.loggers.wandb.init_kwargs.id = server_id + f"{log_name}"
        # Set the new run name
        cfg.loggers.wandb.init_kwargs.name = new_run_name

        # NOTE: This part won't catch any client-level modification to the config and
        # use directly the one taken form the whole run
        # Get the environmental variable for the dump folder
        save_path = os.environ.get("POLLEN_SAVE_PATH", "")
        # Raise an error if the environmental variable is not set
        if not save_path:
            raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
        # Add configuration to the wandb config parameter
        cfg.loggers.wandb["config_file"] = save_path + "/config.yaml"


def set_client_tensorboard_logger(cfg: DictConfig, log_name: str) -> None:
    """Set the tensorboard logger for the client."""
    # Set the tensorboard run name
    if cfg.loggers is not None and "tensorboard" in cfg.loggers:
        assert cfg.loggers.tensorboard is not None, "Tensorboard logger is not set."
        # Add the client id to the parameters
        cfg.loggers.tensorboard.log_name = log_name


def validate_config(cfg: DictConfig) -> None:
    """Validate compatible model and dataloader selection."""
    loaders = [cfg.train_loader]
    if "eval_loader" in cfg:
        eval_loader = cfg.eval_loader
        if isinstance(eval_loader, ListConfig):
            for loader in eval_loader:
                if loader.label is None:
                    raise ValueError(
                        "When specifying multiple evaluation datasets, each one must"
                        "include the `label` attribute."
                    )
                loaders.append(loader)
        else:
            loaders.append(eval_loader)
    for loader in loaders:
        if loader is not None:
            if loader.name == "text":
                if cfg.model.name in {"hf_prefix_lm", "hf_t5"}:
                    raise ValueError(
                        f'Model type "{cfg.model.name}" is not supported when using the'
                        '"text " dataloader. Please use the "text_denoising" dataloader'
                        "to pre-train that model type."
                    )
            elif loader.name == "text_denoising":
                if cfg.model.name == "hf_causal_lm":
                    raise ValueError(
                        f'Model type "{cfg.model.name}" is not supported when using the'
                        '"text_denoising"  dataloader. Please use the "text" dataloader'
                        "to pre-train that model type."
                    )
                if (
                    loader.mixture_of_denoisers.decoder_only_format
                    and cfg.model.name == "hf_t5"
                ):
                    log(
                        WARN,
                        'Model type "hf_t5" requires `decoder_only_format` to be '
                        "``False``. Overriding `decoder_only_format` from ``True`` "
                        "to ``False``.",
                    )
                    loader.mixture_of_denoisers.decoder_only_format = False
                if (
                    not loader.mixture_of_denoisers.decoder_only_format
                ) and cfg.model.name == "hf_prefix_lm":
                    log(
                        WARN,
                        'Model type "hf_prefix_lm" requires `decoder_only_format`'
                        " to be``True``. Overriding `decoder_only_format` from"
                        " ``False`` to``True``.",
                    )
                    loader.mixture_of_denoisers.decoder_only_format = True

    if "icl_tasks" in cfg and cfg.model.name == "hf_t5":
        raise ValueError(
            "ICL evaluation does not currently support Encoder-Decoder models, such"
            'as "hf_t5".'
        )

    if (
        cfg.model.get("fc_type", "torch") != "te"
        and "te" not in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
        and "fp8" in cfg.precision
    ):
        log(
            WARN,
            "fp8 only supported for te.Linear layers. Either set"
            "`cfg.model.fc_typ='te'` or `cfg.model.ffn_config.ffn_type='te_ln_mlp'`"
            "to enable layers using fp8 precision.",
        )

    fsdp_config = cfg.get("fsdp_config", None)
    if (
        cfg.model.get("fc_type", "torch") == "te"
        or "te" in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp")
    ) and fsdp_config is not None:
        act_ckpt = fsdp_config.get("activation_checkpointing", False)
        act_ckpt_reentrant = fsdp_config.get("activation_checkpointing_reentrant", True)
        if fsdp_config is not None and act_ckpt is True and act_ckpt_reentrant is False:
            log(
                WARN,
                "`te.Linear` layers do not support activation_checkpointing with "
                "`activation_checkpointing_reentrant = False`. "
                "Setting cfg.fsdp_config.activation_checkpointing_reentrant=True.",
            )
            cfg.fsdp_config.activation_checkpointing_reentrant = True

    if "te" in cfg.model.get("ffn_config", {}).get("ffn_type", "mptmlp"):
        log(
            WARN,
            "`te.LayerNormMLP` requires has issues with torch._dynamo."
            " Setting`torch._dynamo.config.suppress_errors = True` and falling back"
            " to eager.",
        )
        torch._dynamo.config.suppress_errors = True  # type: ignore[reportAttributeAccessIssue]

    if cfg.model.get("load_in_8bit", False):
        raise ValueError(
            "`load_in_8bit` is only supported for evaluation rather than training."
        )


def adapt_train_batch_size_to_num_devices(cfg: DictConfig) -> None:
    """Adapt the batch size to the number of devices."""
    visible_devices = ast.literal_eval(str(os.getenv("APPOINTED_CUDA_DEVICE", "null")))
    if type(visible_devices) is tuple:
        assert len(visible_devices) > 1
        original_batch_size = cfg.global_train_batch_size
        ratio = cfg.global_train_batch_size // len(visible_devices)
        estimated_batch_size = int(ratio * len(visible_devices))
        if estimated_batch_size != cfg.global_train_batch_size:
            cfg.global_train_batch_size = estimated_batch_size
            log(
                WARNING,
                "Train batch size (%s) was not appropriate for %s GPUs available. "
                "New train batch size: %s",
                original_batch_size,
                len(visible_devices),
                cfg.global_train_batch_size,
            )


def set_n_workers_dataloaders(
    cfg: DictConfig,
    device: Device | DeviceGPU | DeviceCPU | None,
    cap: int = 32,
) -> None:
    """Set the `n_workers` parameter for all dataloaders in the config."""
    # Retrieve system information
    n_cpu_cores_available = get_n_cpu_cores()
    n_workers: int
    if isinstance(device, DeviceCPU):
        # CPU-only environment that cannot be collaborative
        cpu_concurrency = int(os.getenv("CPU_CONCURRENCY", "1"))
        n_workers = n_cpu_cores_available // cpu_concurrency
    elif isinstance(device, DeviceGPU) or device is None:
        # Collaborative or not environment: multiple GPUs are concurrently used for
        # training each having its own dataloader process
        n_cuda_device = get_n_cuda_devices()
        n_workers = n_cpu_cores_available // n_cuda_device
    else:
        raise TypeError(f"Device type {type(device)} is not supported.")
    if cfg.train_loader.num_workers == "auto":
        cfg.train_loader.num_workers = min(n_workers, cap)
    if cfg.eval_loader.num_workers == "auto":
        cfg.eval_loader.num_workers = min(n_workers, cap)
