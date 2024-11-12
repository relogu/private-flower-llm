"""Provides functionality for manipulating MosaicML configs."""

import ast
import copy
import json
import os
from logging import DEBUG, INFO, WARN, WARNING
import re
import tempfile
from typing import Any

import torch
from composer.devices import DeviceGPU, DeviceCPU, Device
from flower_llm.conf.base_schema import S3CommConfig
from flower_llm.utils import (
    download_file_from_s3,
    merge_freq_dicts,
    create_remote_up_down,
)
from flwr.common.logger import log


from omegaconf import DictConfig, ListConfig, OmegaConf


from flower_llm.server.s3_utils import list_objects
from flower_llm.utils import (
    get_n_cpu_cores,
    get_n_cuda_devices,
)
from dataclasses import dataclass, asdict
import operator


# Constant for the frequency dictionary name
FREQ_DICT_NAME = "1_gram.json"
FREQ_DICT_CACHE_NAME = "_freq_dict.json"


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


def preprocess_stream_paths(dataset_config: DictConfig) -> tuple[str, str, str]:
    """Preprocess the stream paths for the dataset.

    Parameters
    ----------
    dataset_config : DictConfig
        The dataset configuration.

    Returns
    -------
    None
    """
    root_remote = dataset_config.pop("root_remote", "")
    root_remote = root_remote + "/" if root_remote else root_remote
    root_local = dataset_config.pop("root_local", "")
    root_local = root_local + "/" if root_local else root_local
    split = dataset_config.pop("split", "")
    return root_remote, root_local, split


def concatenate_streams(clients_streams: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate the streams for all clients.

    Parameters
    ----------
    clients_streams : list[dict[str, Any]]
        The clients streams.

    Returns
    -------
    dict[str, Any]
        The concatenated streams.
    """
    counter = 0
    current_client_stream: dict[str, Any] = {}
    for client_stream in clients_streams:
        assert "client_streams" in client_stream
        client_streams = client_stream["client_streams"]
        assert isinstance(client_streams, DictConfig)
        for stream in client_streams.values():
            current_client_stream |= {f"stream_{counter}": stream}
            counter += 1

    return current_client_stream


def get_actual_stream(
    root_local: str, root_remote: str, split: str, current_client_stream: dict[str, Any]
) -> dict[str, StreamDict]:
    """Get the actual streams for the client.

    Parameters
    ----------
    root_local : str
        The root local path.
    root_remote : str
        The root remote path.
    split : str
        The split.
    current_client_stream : dict[str, Any]
        The current client stream.

    Returns
    -------
    dict[str, StreamDict]
        The actual streams.
    """
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
        stream.remote = stream.remote.rstrip("/") if stream.remote else stream.remote

    return actual_streams


def client_set_data_config(
    cid: int | str | None,
    cfg: DictConfig,
    split_eval: bool = False,
) -> None:
    """Set the client data configuration for the client.

    Parameters
    ----------
    cid : int | str | None
        The client id.
    cfg : DictConfig
        The configuration object.
    split_eval : bool
        Whether to split the evaluation data.

    Returns
    -------
    DictConfig
        The updated configuration object.
    """
    # Retrieve the train config to construct the dataset for the train loader
    train_split: tuple[str, DictConfig] = ("train", cfg.train_loader.dataset)
    val_split: tuple[str, DictConfig] = ("val", cfg.eval_loader.dataset)
    for loop_split, dataset_config in (
        (train_split, val_split) if not split_eval else (train_split,)
    ):
        # Get the root path for remote and local data
        root_remote, root_local, split = preprocess_stream_paths(dataset_config)
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
            current_client_stream |= concatenate_streams(clients_streams)

        actual_streams = get_actual_stream(
            root_local, root_remote, split, current_client_stream
        )

        # Convert the streams to dictionaries
        streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}
        # Assign the streams to the appropriate loaders
        if loop_split == "train":
            cfg.train_loader.dataset.streams = streams_dict
        elif loop_split == "val":
            cfg.eval_loader.dataset.streams = streams_dict

    if split_eval:
        # Set the evaluation split to be the same as the training split
        loop_split, dataset_config = val_split
        clients_streams = dataset_config.streams
        eval_loaders = []
        root_remote, root_local, split = preprocess_stream_paths(dataset_config)
        for inner_cid in range(len(clients_streams)):
            current_client_stream = {}
            current_client_stream |= clients_streams[
                int(inner_cid) % len(clients_streams)
            ]["client_streams"]

            actual_streams = get_actual_stream(
                root_local, root_remote, split, current_client_stream
            )

            streams_dict = {
                name: asdict(stream) for name, stream in actual_streams.items()
            }

            client_eval_loader = copy.deepcopy(cfg.eval_loader)
            client_eval_loader.dataset.streams = streams_dict
            client_eval_loader.label = f"client_{inner_cid}"
            eval_loaders.append(client_eval_loader)

        cfg.eval_loader = ListConfig(eval_loaders)


def set_dataset_default_params(cfg: DictConfig) -> None:
    """Set the default parameters for the dataset."""
    # Set the `pre-download` value as 8*batch_size
    if cfg.train_loader.dataset.get("predownload", None) is None:
        cfg.train_loader.dataset.predownload = 8 * cfg.device_train_batch_size
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


def set_client_load_path(
    cfg: DictConfig, cid: int | str, n_steps: int
) -> tuple[bool, bool]:
    """Set the save and load path given the server round and client id."""
    # Set client load path
    set_client_save_and_load_path(cfg, cid)
    # Flag to notify whether to skip this iteration or not
    skip_iteration = False
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        try:
            # Are there any checkpoints?
            _is_remote, remote_objects = list_objects(cfg.save_folder)
            if not remote_objects:
                log(
                    INFO,
                    "No checkpoints found in %s. Starting training from scratch.",
                    cfg.save_folder,
                )
                assert cfg.load_path is None
                return skip_iteration, False
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
                return skip_iteration, True
            # Load the latest checkpoint
            log(
                INFO, "Looking for the latest checkpoint to load in %s", cfg.save_folder
            )
            epoch, batches = sorted_pairs[-1]
            if batches < n_steps:
                cfg.load_path = (
                    cfg.save_folder + f"/ep{epoch}-ba{batches}-" + "rank{rank}.pt"
                )
                log(INFO, "Set checkpoint to load: %s", cfg.load_path)
        except Exception as e:
            log(WARNING, "The `load_path` wasn't set.", exc_info=e, stack_info=True)
    return skip_iteration, True


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


def get_stream_freq_dict_for_client(
    client_streams: dict[str, dict[str, Any]],
    s3_comm_config: S3CommConfig | None,
    run_uuid: str | None,
    cid: int | str | None,
    allow_failures: bool = False,
) -> dict[int, tuple[int, str]]:
    """Get the token frequencies for a single client's streams.

    Parameters
    ----------
    client_streams : dict[str, dict[str, Any]]
        The client streams.
    s3_comm_config : S3CommConfig | None
        The S3 communication configuration.
    run_uuid : str | None
        The run UUID.
    """
    actual_streams = {key: StreamDict(**value) for key, value in client_streams.items()}
    # Stores the merged frequency dictionary across streams
    stream_freq_dict: dict[int, tuple[int, str]] = {}
    tmp_dir = tempfile.gettempdir()

    cached_file_name = os.path.join(  # noqa: PTH118
        tmp_dir, str(cid) + FREQ_DICT_CACHE_NAME
    )

    failed_cnt = 0

    if not os.path.exists(cached_file_name):  # noqa: PTH110
        for stream in actual_streams.values():
            assert stream.local is not None, "Local path is not set."
            assert stream.split is not None, "Split is not set."
            local_file_name = os.path.join(  # noqa: PTH118
                stream.local, stream.split, FREQ_DICT_NAME
            )
            if not os.path.exists(local_file_name):  # noqa: PTH110
                assert stream.remote is not None, "Remote path is not set."
                assert s3_comm_config is not None, "S3 communication config is not set."
                assert run_uuid is not None, "Run UUID is not set."
                stream_remote_post_processed = stream.remote.replace("s3://", "")

                root_remote, *rest = stream_remote_post_processed.split("/")

                remote_up_down = create_remote_up_down(
                    bucket_name=root_remote,
                    prefix="",
                    run_uuid=run_uuid,
                    num_attempts=s3_comm_config.num_attempts,
                    client_config=OmegaConf.to_container(
                        s3_comm_config.backend_kwargs.client_config
                    ),  # type: ignore[reportArgumentType, arg-type]
                )
                remote_path = os.path.join(  # noqa: PTH118
                    os.path.join(*rest),  # noqa: PTH118
                    stream.split,
                    FREQ_DICT_NAME,
                )
                try:
                    download_file_from_s3(remote_up_down, remote_path, local_file_name)
                except FileNotFoundError as _:
                    if not allow_failures:
                        raise
            try:
                with open(local_file_name, encoding="utf-8") as f:
                    loaded_map: dict = json.load(f).items()

                freq_map: dict[int, tuple[int, str]]
                try:
                    freq_map = {int(ast.literal_eval(k)[0]): v for k, v in loaded_map}
                except TypeError:
                    freq_map = {int(ast.literal_eval(k)): v for k, v in loaded_map}
                stream_freq_dict = merge_freq_dicts(stream_freq_dict, freq_map)
            except FileNotFoundError as _:
                if not allow_failures:
                    raise
                failed_cnt += 1
        log(
            DEBUG,
            "Loaded stream_freq_dict, len: %s, failures %s",
            len(stream_freq_dict),
            failed_cnt,
        )
        with open(cached_file_name, "w", encoding="utf-8") as f:
            json.dump(stream_freq_dict, f, indent=4)
    else:
        with open(cached_file_name, encoding="utf-8") as f:
            stream_freq_dict = {int(k): (v[0], v[1]) for k, v in json.load(f).items()}
        log(
            DEBUG,
            "Loaded stream_freq_dict from cache %s, len: %s",
            cached_file_name,
            len(stream_freq_dict),
        )

    return stream_freq_dict
