"""Provides the internal functions used by the LLM client."""

import atexit
import copy
import gc
import logging
import os
import re
import time
import warnings
from collections import OrderedDict
from logging import DEBUG, ERROR, WARN
from typing import Any, cast

import streaming
import torch
from composer import Callback, Evaluator, Trainer
from composer.devices import DeviceGPU, DeviceCPU
from composer.profiler import JSONTraceHandler, Profiler, TraceHandler, cyclic_schedule
from composer.utils import dist, reproducibility, get_device
from composer.utils.file_helpers import validate_given_remote_path
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from llmfoundry.data.dataloader import build_dataloader

from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_icl_data_and_gauntlet,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
    build_composer_model,
)
from llmfoundry.utils.config_utils import (
    pop_config,
    process_init_device,
    update_batch_size_info,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from streaming.base.shared.memory import SharedMemory, shared_memory_list

from composer.loggers import RemoteUploaderDownloader
from composer.utils import S3ObjectStore
from composer.utils.file_helpers import list_remote_objects

import numpy as np
from flower_llm.clients.llm_config_functions import (
    adapt_train_batch_size_to_num_devices,
    client_set_data_config,
    set_client_load_path,
    set_client_tensorboard_logger,
    set_client_wandb_logger,
    set_dataset_default_params,
    validate_config,
    set_n_workers_dataloaders,
)
from flower_llm.utils import (
    construct_parameters_dict,
    get_list_of_parameters_names,
    get_trainable_params_dict,
    parameters_checker,
    set_trainer_trainable_params_dict,
    sum_of_squares,
    get_parameters_from_state,
)
from dataclasses import asdict
import ast
from flower_llm.utils import ClientState


def copy_old_checkpoints_to_new_run(
    remote_up_down: RemoteUploaderDownloader,
    bucket_uri: str,
    run_uuid: str,
    restore_run_uuid: str,
    restore_run_round: int,
    restore_run_step: int,
    n_total_clients: int | None,
) -> None:
    """Copy old checkpoints to the new run folder.

    Parameters
    ----------
        remote_up_down (RemoteUploaderDownloader): The remote uploader and downloader.
        bucket_uri (str): The bucket URI.
        run_uuid (str): The run UUID.
        restore_run_uuid (str): The restore run UUID.
        restore_run_round (int): The restore run round.
        restore_run_step (int): The restore run step.
        n_total_clients (int): The total number of clients.

    Returns
    -------
        None

    Raises
    ------
        NotImplementedError: If the backend is not an S3ObjectStore.
        ValueError: If the old run folder or the new run folder is not found.
    """
    backend = remote_up_down.remote_backend
    if not isinstance(backend, S3ObjectStore):
        raise NotImplementedError(
            "Support for resuming from non-S3 backends is not yet implemented."
        )

    new_run_folder = bucket_uri + f"/{run_uuid}"
    old_run_folder = bucket_uri + f"/{restore_run_uuid}"
    if (old_run_val := validate_given_remote_path(old_run_folder)) and (
        _new_run_val := validate_given_remote_path(new_run_folder)
    ):
        state_bin = restore_run_uuid + f"/server/{restore_run_round}/state.bin"

        momentum_vec = (
            old_run_folder + f"/server/{restore_run_round}/current_momentum_vector.npz"
        )

        parameters_no_ext = (
            old_run_folder + f"/server/{restore_run_round}/current_server_parameters"
        )
        parameters = (
            parameters_no_ext.replace(bucket_uri + "/", "") + ".bin"
            if validate_given_remote_path(parameters_no_ext + ".bin")
            else (parameters_no_ext.replace(bucket_uri + "/", "") + ".npz")
        )
        old_run_folder_no_prefix = old_run_folder.replace(bucket_uri + "/", "")
        # NOTE: cumbersome way of dealing with character escaping
        client_paths = [
            client_path
            for client_path in list_remote_objects(old_run_folder)
            if re.match(
                f"{old_run_folder_no_prefix}/"
                r"client_.*/ep(?:\d+)"
                f"-ba{restore_run_step}",
                client_path,
            )
        ]

        if (
            n_total_clients is not None
            and (found_clients := len(client_paths)) != n_total_clients
        ):
            raise ValueError(
                f"Found {found_clients} clients in the old run folder {old_run_folder},"
                f" but expected {n_total_clients}."
            )

        paths_to_copy = [state_bin, parameters]

        if validate_given_remote_path(momentum_vec):
            paths_to_copy.append(momentum_vec.replace(bucket_uri + "/", ""))
        else:
            log(
                DEBUG,
                f"Could not find momentum vector to copy from {momentum_vec}",
            )

        paths_to_copy.extend(client_paths)

        for path in paths_to_copy:
            copy_source = {"Bucket": backend.bucket, "Key": path}
            target_key = path.replace(restore_run_uuid, run_uuid)
            log(DEBUG, "Copying %s to %s", path, target_key)
            backend.client.copy(copy_source, backend.bucket, target_key)

    else:
        if not old_run_val:
            raise ValueError(
                f"Could not find the old run folder {old_run_folder} to copy"
                " checkpoints."
            )
        raise ValueError(
            f"Could not find the new run folder {new_run_folder} to copy checkpoints."
        )


def set_trainer_timestamp(trainer: Trainer, timestamp: int) -> None:
    """Set the timestamp of the trainer."""
    log(DEBUG, "Stepping the timestamp.")
    while trainer.state.timestamp.batch.value < timestamp:
        trainer.state.timestamp = trainer.state.timestamp.to_next_batch()


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log(
        DEBUG,
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable params (%): {100 * trainable_params / all_param}",
    )


def get_raw_model_parameters(
    _cfg: DictConfig,
    verbose: bool = False,
    return_names: bool = False,
) -> NDArrays | tuple[NDArrays, list[str]]:
    """Get the raw model parameters."""
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=(
            "torch.distributed.*_base is a private functionand will be deprecated.*"
        ),
    )
    # Check for incompatibilities between the model and data loaders
    validate_config(_cfg)
    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(_cfg)
    # Get model config
    model_config: DictConfig = pop_config(_cfg, "model", must_exist=True)
    # Get tokenizer config
    tokenizer_config: dict[str, Any] = pop_config(
        _cfg, "tokenizer", must_exist=True, convert=True
    )
    tokenizer_name = tokenizer_config["name"]
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    # Get model while forcing cpu to prevent any GPU allocation
    model_config.init_device = "cpu"
    model = build_composer_model(
        name=model_config.name,
        cfg=model_config,
        tokenizer=build_tokenizer(tokenizer_name, tokenizer_kwargs),
        init_context=process_init_device(model_config, None),
        master_weights_dtype=model_config.get("master_weights_dtype", None),
    )
    model.cpu()
    # Get model summary
    if verbose:
        log(DEBUG, model)
    parameters_ndarrays = [
        val.detach().to("cpu").numpy()
        for _, val in get_trainable_params_dict(model).items()
    ]
    if return_names:
        return parameters_ndarrays, list(get_trainable_params_dict(model).keys())
    else:
        return parameters_ndarrays


def _get_trainer_object(
    _cfg: DictConfig,
    cid: int | str,
) -> tuple[Trainer, bool, DictConfig, list[str]]:
    # Filter deprecation warning from torch internal usage
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=(
            "torch.distributed.*_base is a private functionand will be deprecated.*"
        ),
    )

    # Check for incompatibilities between the model and data loaders
    validate_config(_cfg)

    # Resolve all interpolation variables as early as possible
    OmegaConf.resolve(_cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(_cfg)

    # Get max split size mb
    max_split_size_mb: int | None = _cfg.pop("max_split_size_mb", None)
    if max_split_size_mb is not None:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{max_split_size_mb}"

    # Set CUDA lazy loading which can save a bit of memory if not all modules are needed
    cuda_load_lazy: bool = _cfg.pop("cuda_load_lazy", False)
    if cuda_load_lazy:
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # Set seed first
    seed: int = pop_config(_cfg, "seed", must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: int | float = pop_config(
        _cfg, "dist_timeout", must_exist=False, default_value=600.0
    )

    # Set the device in case multiple GPUs are requested to be
    # independent and not collaborative. If `device == None` the
    # Trainer will automatically initialize PyTorch Distributed
    # with the parameters from the environmental variables.
    visible_devices = ast.literal_eval(str(os.getenv("APPOINTED_CUDA_DEVICE", "null")))
    log(DEBUG, f"Visible devices: {visible_devices}")
    # The worker has been appointed a single GPU
    if type(visible_devices) is int:
        device: DeviceGPU | DeviceCPU | None = DeviceGPU(device_id=int(visible_devices))
        log(DEBUG, f"Selecting device {visible_devices}, {device}")
    # The worker has been appointed all GPUs available
    elif type(visible_devices) is tuple:
        assert len(visible_devices) > 1
        device = None
    # The worker is in a CPU-only environment
    else:
        assert visible_devices is None
        device = DeviceCPU()
        log(DEBUG, f"Selecting device CPU, {device}")
    log(DEBUG, "Initializing dist with device...")
    dist.initialize_dist(get_device(device), timeout=dist_timeout)
    log(DEBUG, "Testing barrier with device...")
    dist.barrier()
    log(DEBUG, "Barrier test passed with device.")

    # Get global and device batch size information from distributed/single node setting
    adapt_train_batch_size_to_num_devices(_cfg)
    _cfg = update_batch_size_info(_cfg)
    logged_cfg.update(_cfg, merge=True)

    # Mandatory model training configs
    set_n_workers_dataloaders(cfg=_cfg, device=device)
    client_set_data_config(cfg=_cfg, cid=cid)
    # Apply dataset defaults
    set_dataset_default_params(_cfg)
    model_config: DictConfig = pop_config(_cfg, "model", must_exist=True)
    tokenizer_config: dict[str, Any] = pop_config(
        _cfg, "tokenizer", must_exist=True, convert=True
    )
    optimizer_config: dict[str, Any] = pop_config(
        _cfg, "optimizer", must_exist=True, convert=True
    )
    scheduler_config: dict[str, Any] = pop_config(
        _cfg, "scheduler", must_exist=True, convert=True
    )
    train_loader_config: DictConfig = pop_config(
        _cfg, "train_loader", must_exist=False, default_value=None
    )

    # Optional fsdp data, fine-tuning, and eval configs
    fsdp_config: dict[str, Any] | None = pop_config(
        _cfg, "fsdp_config", must_exist=False, default_value=None, convert=True
    )
    eval_loader_config: DictConfig | ListConfig | None = pop_config(
        _cfg, "eval_loader", must_exist=False, default_value=None
    )
    icl_tasks_config: ListConfig | str | None = pop_config(
        _cfg, "icl_tasks", must_exist=False, default_value=None
    )
    eval_gauntlet_config: DictConfig | str | None = pop_config(
        _cfg, "eval_gauntlet", must_exist=False, default_value=None
    )
    if eval_gauntlet_config is None:
        eval_gauntlet_config = pop_config(
            _cfg, "model_gauntlet", must_exist=False, default_value=None
        )
        if eval_gauntlet_config is not None:
            log(
                DEBUG,
                "Use of the key `model_gauntlet` is deprecated, please use the key"
                "`eval_gauntlet`",
            )
    icl_subset_num_batches: int | None = pop_config(
        _cfg, "icl_subset_num_batches", must_exist=False, default_value=None
    )
    icl_seq_len: int | None = pop_config(
        _cfg, "icl_seq_len", must_exist=False, default_value=None
    )
    # Optional logging, evaluation and callback configs
    set_client_wandb_logger(_cfg, cid)
    set_client_tensorboard_logger(_cfg, cid)
    logger_configs: DictConfig | None = pop_config(
        _cfg, "loggers", must_exist=False, default_value=None
    )
    callback_configs: DictConfig | None = pop_config(
        _cfg, "callbacks", must_exist=False, default_value=None
    )
    algorithm_configs: DictConfig | None = pop_config(
        _cfg, "algorithms", must_exist=False, default_value=None
    )

    # Mandatory hyperparameters for training
    # Adapt batch sizes to the number of GPUs available
    device_train_batch_size: int = pop_config(
        _cfg, "device_train_batch_size", must_exist=True
    )
    device_eval_batch_size: int = pop_config(
        _cfg, "device_eval_batch_size", must_exist=True
    )
    max_duration: int | str = pop_config(_cfg, "max_duration", must_exist=True)
    eval_interval: int | str = pop_config(_cfg, "eval_interval", must_exist=True)
    precision: str = pop_config(_cfg, "precision", must_exist=True)
    max_seq_len: int = pop_config(_cfg, "max_seq_len", must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME", "llm")
    run_name: str = pop_config(
        _cfg, "run_name", must_exist=False, default_value=default_run_name
    )
    save_folder: str | None = pop_config(
        _cfg, "save_folder", must_exist=False, default_value=None
    )
    save_latest_filename: str = pop_config(
        _cfg,
        "save_latest_filename",
        must_exist=False,
        default_value="latest-rank{rank}.pt",
    )
    save_overwrite: bool = pop_config(
        _cfg, "save_overwrite", must_exist=False, default_value=False
    )
    save_weights_only: bool = pop_config(
        _cfg, "save_weights_only", must_exist=False, default_value=False
    )
    save_filename: str = pop_config(
        _cfg,
        "save_filename",
        must_exist=False,
        default_value="ep{epoch}-ba{batch}-rank{rank}.pt",
    )
    save_interval: str | int = pop_config(
        _cfg, "save_interval", must_exist=False, default_value="1000ba"
    )
    save_num_checkpoints_to_keep: int = pop_config(
        _cfg, "save_num_checkpoints_to_keep", must_exist=False, default_value=-1
    )

    save_ignore_keys: list[str] | None = pop_config(
        _cfg, "save_ignore_keys", must_exist=False, default_value=None
    )
    progress_bar = pop_config(
        _cfg, "progress_bar", must_exist=False, default_value=False
    )
    log_to_console: bool = pop_config(
        _cfg, "log_to_console", must_exist=False, default_value=True
    )
    python_log_level: str | None = pop_config(
        _cfg, "python_log_level", must_exist=False, default_value="debug"
    )
    console_log_interval: int | str = pop_config(
        _cfg, "console_log_interval", must_exist=False, default_value="1ba"
    )
    device_train_microbatch_size: str | int = pop_config(
        _cfg, "device_train_microbatch_size", must_exist=False, default_value="auto"
    )
    eval_subset_num_batches: int = pop_config(
        _cfg, "eval_subset_num_batches", must_exist=False, default_value=-1
    )
    eval_first: bool = pop_config(
        _cfg, "eval_first", must_exist=False, default_value=False
    )
    load_path: str = pop_config(_cfg, "load_path", must_exist=False, default_value=None)
    load_weights_only: bool = pop_config(
        _cfg, "load_weights_only", must_exist=False, default_value=False
    )
    load_strict_model_weights: bool = pop_config(
        _cfg, "load_strict_model_weights", must_exist=False, default_value=True
    )
    load_ignore_keys: list[str] | None = pop_config(
        _cfg, "load_ignore_keys", must_exist=False, default_value=None
    )
    compile_config: dict[str, Any] | None = pop_config(
        _cfg, "compile_config", must_exist=False, default_value=None
    )
    pop_config(_cfg, "metadata", must_exist=False, default_value=None, convert=True)

    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        logged_cfg.get("run_name", None) is not None
        and save_folder is not None
        and not save_overwrite
        and not save_weights_only
    ):
        autoresume_default = True

    if _cfg.get("autoresume") is None and autoresume_default:
        log(
            DEBUG,
            "As run_name, save_folder, and save_latest_filename are set,           "
            "      changing autoresume default to True...",
        )

    autoresume: bool = pop_config(
        _cfg, "autoresume", must_exist=False, default_value=autoresume_default
    )

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(_cfg, "data_local", must_exist=False)
    pop_config(_cfg, "data_remote", must_exist=False)
    pop_config(_cfg, "global_seed", must_exist=False)
    pop_config(_cfg, "global_train_batch_size", must_exist=False)
    pop_config(_cfg, "n_gpus", must_exist=False)
    pop_config(_cfg, "device_train_grad_accum", must_exist=False)

    # Warn users for unused parameters
    for key in _cfg:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            log(
                WARN,
                "Unused parameter %s found in cfg. Please check your yaml to ensure"
                " this parameter is necessary.",
                key,
            )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        if os.environ.get("LOCAL_RANK", "0") == "0":
            log(
                WARN,
                "FSDP is not applicable for single-GPU training. Reverting to DDP.",
            )
        fsdp_config = None

    # Set logging level
    # TODO: Make this through the Flower logger
    # NOTE: Logging only Rank 0 by default
    if python_log_level is not None and dist.get_global_rank() == 0:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=(
                f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d]"
                "[%(threadName)s]: %(levelname)s: %(name)s: %(message)s"
            )
        )
        logging.getLogger("llmfoundry").setLevel(python_log_level.upper())

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({"fsdp_config": fsdp_config}, merge=True)

    # Build tokenizer
    tokenizer_name = tokenizer_config["name"]
    tokenizer_kwargs = tokenizer_config.get("kwargs", {})
    tokenizer = build_tokenizer(tokenizer_name, tokenizer_kwargs)

    # Scheduler
    scheduler_name: str = scheduler_config.pop("name")
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = (
        [
            build_logger(str(name), logger_cfg)
            for name, logger_cfg in logger_configs.items()
        ]
        if logger_configs
        else []
    )

    # Profiling
    profiler: Profiler | None = None
    profiler_cfg: DictConfig | None = pop_config(
        _cfg, "profiler", must_exist=False, convert=False, default_value=None
    )
    if profiler_cfg:
        profiler_schedule_cfg: dict = pop_config(
            profiler_cfg, "schedule", must_exist=True, convert=True
        )
        profiler_schedule = cyclic_schedule(**profiler_schedule_cfg)
        # Only support json trace handler
        profiler_trace_handlers: list[TraceHandler] = []
        profiler_trace_cfg: dict | None = pop_config(
            profiler_cfg,
            "json_trace_handler",
            must_exist=False,
            default_value=None,
            convert=True,
        )
        if profiler_trace_cfg:
            profiler_trace_handlers.append(JSONTraceHandler(**profiler_trace_cfg))
        profiler = Profiler(  # type: ignore[misc]
            **profiler_cfg,  # type: ignore[reportCallIssue]
            trace_handlers=profiler_trace_handlers,
            schedule=profiler_schedule,
        )

    # Callbacks
    callbacks: list[Callback] = (
        [
            build_callback(str(name), callback_cfg)
            for name, callback_cfg in callback_configs.items()
        ]
        if callback_configs
        else []
    )

    # Algorithms
    algorithms = (
        [
            build_algorithm(str(name), algorithm_cfg)
            for name, algorithm_cfg in algorithm_configs.items()
        ]
        if algorithm_configs
        else None
    )

    # Train loader
    train_loader = None
    if train_loader_config is not None:
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            device_train_batch_size,
        )

    # Evaluators and eval loaders
    evaluators = []
    eval_loaders = []
    if eval_loader_config is not None:
        is_multi_eval = isinstance(eval_loader_config, ListConfig)
        eval_configs = eval_loader_config if is_multi_eval else [eval_loader_config]
        for eval_config in eval_configs:
            eval_dataloader = build_dataloader(
                eval_config, tokenizer, device_eval_batch_size
            )
            eval_loader = Evaluator(
                label=(
                    f"eval/{eval_config.label}"  # type: ignore[union-attr]
                    if is_multi_eval
                    else "eval"
                ),
                dataloader=eval_dataloader,
                metric_names=[],  # we will add these after model is created
            )
            eval_loaders.append(eval_loader)

    eval_gauntlet_callback = None

    if icl_tasks_config is not None:
        icl_evaluators, _, eval_gauntlet_callback = build_icl_data_and_gauntlet(
            icl_tasks_config,
            eval_gauntlet_config,
            tokenizer,
            device_eval_batch_size,
            icl_seq_len or max_seq_len,
            icl_subset_num_batches,
        )
        evaluators.extend(icl_evaluators)

    if eval_gauntlet_callback is not None:
        callbacks.append(eval_gauntlet_callback)

    # Model
    model = build_composer_model(
        name=model_config.name,
        cfg=model_config,
        tokenizer=tokenizer,
        init_context=init_context,
        master_weights_dtype=model_config.get("master_weights_dtype", None),
    )
    parameters_names = get_list_of_parameters_names(model)

    # Log number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    logged_cfg.update({"n_params": n_params})

    # Optimizer
    optimizer_name: str = optimizer_config.pop("name")
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    # Now add the eval metrics
    if eval_loader_config is not None:
        assert model.train_metrics is not None
        eval_metric_names = list(model.train_metrics.keys())
        for eval_loader in eval_loaders:
            eval_loader.metric_names = eval_metric_names
            # Put the base eval_loaders first
            evaluators.insert(0, eval_loader)

    # Build the Trainer
    trainer = Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=evaluators,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=eval_interval,
        eval_subset_num_batches=eval_subset_num_batches,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision if not isinstance(device, DeviceCPU) else None,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,
        save_folder=save_folder,
        save_ignore_keys=save_ignore_keys,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        save_metrics=True,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_strict_model_weights=load_strict_model_weights,
        load_ignore_keys=load_ignore_keys,
        autoresume=autoresume,
        python_log_level=python_log_level,
        dist_timeout=dist_timeout,
        profiler=profiler,
        compile_config=compile_config,
        device=device,
    )
    return trainer, eval_first, logged_cfg, parameters_names


def get_parameters(
    config: dict[str, Scalar],
    cfg: DictConfig,
) -> NDArrays:
    """Return the current local model parameters.

    Parameters
    ----------
    config : Config
        Configuration parameters requested by the server.
        This can be used to tell the client which parameters
        are needed along with some Scalar attributes.

    Returns
    -------
    parameters : NDArrays
        The local model parameters as a list of NumPy ndarrays.
    """
    return cast(NDArrays, get_raw_model_parameters(copy.deepcopy(cfg)))


def set_parameters_to_state(
    parameters: NDArrays,
    trainer: Trainer,
) -> None:
    """Implement how to set parameters in the case of an LLM."""
    model_parameters_dict = get_trainable_params_dict(trainer.state.model)
    params_dict = zip(model_parameters_dict.keys(), parameters, strict=True)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    trainer.state.model.load_state_dict(state_dict, strict=False)
    del state_dict


def llm_fit(
    parameters: NDArrays,
    config: dict,
    cfg: DictConfig,
    cid: int | str,
) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any]]:
    """Implement the fit step using MosaicML codebase."""
    # Retrieve the clients' states
    client_state: dict[int | str, dict[str, Any]] = ast.literal_eval(
        config["client_state"]
    )
    # Extract current client's state
    client_state_struct = ClientState(**client_state[cid])
    # Get the number of local steps done by the current client
    num_batches_trained = int(str(cfg["local_steps"]).replace("ba", ""))
    # Initialize training hyperparameters
    global_train_batch_size = int(cfg["global_train_batch_size"])
    start_time = time.time_ns()
    model_parameters = []
    n_samples_trained = 0
    train_metrics: dict[str, Scalar] = {}
    # Set the loading path
    skip_iteration = set_client_load_path(
        cfg,
        cid,
        config["server_steps_cumulative"] + num_batches_trained,
    )
    cfg.load_ignore_keys = ["*scheduler*"]  # type: ignore[union-attr]
    if config["reset_optimizer"]:
        # Ignoring the optimizer state if loading a checkpoint
        cfg.load_ignore_keys += ["*optim*"]  # type: ignore[union-attr]
        # Ignoring the optimizer state when saving a checkpoint
        cfg.save_ignore_keys = ["*optim*"]  # type: ignore[union-attr]
    # NOTE: The following, when re-loading from a checkpoint, returns a weird error
    # if not skip_iteration:
    #     # Ignoring loading the model as we need to set it from the server
    #     cfg.load_ignore_keys += ["*model*"]
    # Extract configs to build the trainer
    trainer, eval_first, _, parameters_names = _get_trainer_object(_cfg=cfg, cid=cid)

    # Create the server parameters dictionary
    server_parameters_dict = construct_parameters_dict(parameters_names, parameters)

    initial_trainer_parameters = get_parameters_from_state(
        {},
        trainer,
    )
    parameters_checker(initial_trainer_parameters, parameters, False)

    # log(DEBUG, f"Trainer config: {logged_cfg}")
    log(DEBUG, "Trainer object created.")
    train_metrics |= {"client/fit_init_time": (time.time_ns() - start_time) * 1e-9}
    # NOTE: Skipping a few steps if the checkpoint already exists
    if not skip_iteration:
        # Set the timestamp to the current time
        set_trainer_timestamp(trainer, config["server_steps_cumulative"])

        # Set the parameters
        if parameters is not None and not skip_iteration:
            # log(DEBUG, "Initializing model...")
            start_time = time.time_ns()
            set_trainer_trainable_params_dict(trainer, server_parameters_dict)

            current_trainer_parameters = get_parameters_from_state({}, trainer)
            parameters_checker(
                current_trainer_parameters, initial_trainer_parameters, False
            )
            parameters_checker(current_trainer_parameters, parameters, True)

            train_metrics |= {
                "client/fit_set_parameters_time": (time.time_ns() - start_time) * 1e-9
            }

        # Eval first if requested
        if eval_first:
            start_time = time.time_ns()
            trainer.eval()
            train_metrics |= {
                "client/fit_pre_eval_time": (time.time_ns() - start_time) * 1e-9
            }
        # log(DEBUG, "Starting training...")
        # Execute fit step for the appointed duration
        try:
            start_time = time.time_ns()
            trainer.fit(duration=0 if skip_iteration else cfg["local_steps"])
            train_metrics |= {"client/fit_time": (time.time_ns() - start_time) * 1e-9}
        except Exception as e:
            log(ERROR, "llm_fit::trainer.fit", exc_info=e, stack_info=True)
    client_state_struct.steps_done += num_batches_trained
    # Retrieve number of samples trained
    # NOTE: We assume all the clients train with the same batch size,
    # so we just consider the number of local steps
    # NOTE: Assuming that this is the correct value of local steps
    # for the client to train in this particular round and no

    n_samples_trained = num_batches_trained * global_train_batch_size

    client_state_struct.local_steps_cumulative += num_batches_trained

    # Retrieve training metrics
    train_metrics |= {
        k: v.detach().cpu().item()  # type: ignore[attr-defined]
        for k, v in trainer.state.train_metric_values.items()
    }
    log(DEBUG, f"Train metrics: {train_metrics}")
    # Retrieve model parameters
    start_time = time.time_ns()
    model_parameters = get_parameters_from_state({}, trainer)

    parameters_checker(model_parameters, initial_trainer_parameters, False)
    parameters_checker(model_parameters, parameters, False)

    train_metrics |= {
        "client/fit_get_parameters_time": (time.time_ns() - start_time) * 1e-9
    }
    # Only rank 0 collects metrics related to the pseudo gradients
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        start_time = time.time_ns()
        per_layer_sum_of_squares = [
            sum_of_squares([x - y])
            for x, y in zip(parameters, model_parameters, strict=False)
        ]

        for i, plss in enumerate(per_layer_sum_of_squares):
            train_metrics |= {
                f"client/layer/{i}/l2_norm_of_pseudo_gradient": float(np.sqrt(plss))
            }

        l2_norm_of_pseudo_gradient: float = float(
            np.sqrt(sum(per_layer_sum_of_squares))
        )

        train_metrics |= {"client/l2_norm_pseudo_gradient": l2_norm_of_pseudo_gradient}
        train_metrics |= {
            "client/fit_metrics_collection_time": (time.time_ns() - start_time) * 1e-9
        }

    # Close the trainer
    start_time = time.time_ns()
    # Close the trainer, wait for all the collaborators first
    dist.barrier()
    trainer.close()
    # NOTE: Clean up leaking shared memories
    for shm in shared_memory_list:
        SharedMemory.cleanup(shm)
        atexit.unregister(SharedMemory.cleanup)
    shared_memory_list.clear()
    # Delete the trainer
    try:
        del trainer
    except Exception as e:
        log(ERROR, "Error deleting trainer", exc_info=e, stack_info=True)
    # Clean-up garbage collector and cuda cache
    gc.collect()
    torch.cuda.empty_cache()
    # Cleaning stale shared memory
    streaming.base.util.clean_stale_shared_memory()  # type: ignore[reportAttributeAccessIssue]
    # Only rank 0 collects metrics
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        train_metrics |= {
            "client/fit_trainer_closing_time": (time.time_ns() - start_time) * 1e-9
        }
        train_metrics |= {"client_state_acc": str({cid: asdict(client_state_struct)})}

    return model_parameters, n_samples_trained, train_metrics


def llm_eval(
    parameters: NDArrays,
    config: dict,
    cfg: DictConfig,
) -> tuple[float, int, dict[str, Scalar]]:
    """Implement the fit step using MosaicML codebase."""
    start_time = time.time_ns()
    num_samples = 0
    eval_metrics: dict[str, Scalar] = {}
    # NOTE: Exclude unnecessary checkpointing and loggers for eval
    cfg.autoresume = False  # type: ignore[union-attr]
    cfg.save_folder = None  # type: ignore[union-attr]
    cfg.load_path = None  # type: ignore[union-attr]
    cfg.loggers = None  # type: ignore[union-attr]
    # Extract configs to build the trainer
    trainer, _, _, parameters_names = _get_trainer_object(
        _cfg=cfg,
        cid=0,
    )

    # Create the server parameters dictionary
    server_parameters_dict = construct_parameters_dict(parameters_names, parameters)

    initial_trainer_parameters = get_parameters_from_state({}, trainer)
    parameters_checker(initial_trainer_parameters, parameters, False)

    eval_metrics |= {"client/eval_init_time": (time.time_ns() - start_time) * 1e-9}

    # Set the parameters
    # log(DEBUG, "Initializing model...")
    start_time = time.time_ns()
    set_trainer_trainable_params_dict(trainer, server_parameters_dict)

    current_trainer_parameters = get_parameters_from_state({}, trainer)
    parameters_checker(current_trainer_parameters, initial_trainer_parameters, False)
    parameters_checker(current_trainer_parameters, parameters, True)

    gc.collect()
    torch.cuda.empty_cache()
    eval_metrics |= {
        "client/eval_set_parameters_time": (time.time_ns() - start_time) * 1e-9
    }

    # log(DEBUG, "Starting evaluation...")
    start_time = time.time_ns()
    trainer.eval()
    eval_metrics |= {"client/eval_time": (time.time_ns() - start_time) * 1e-9}
    start_time = time.time_ns()
    # Only rank 0 collects metrics
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        # Retrieve number of samples evaluated
        num_samples = trainer.state.eval_timestamp._sample.value
        # Retrieve evaluation metrics
        eval_metrics |= {
            "Val" + k: v.detach().cpu().item()  # type: ignore[attr-defined]
            for k, v in trainer.state.eval_metric_values.items()
        }
        eval_metrics |= {
            "client/eval_metrics_collection_time": (time.time_ns() - start_time) * 1e-9
        }

    # Close the trainer
    start_time = time.time_ns()
    trainer.close()
    # NOTE: Clean up leaking shared memories
    for shm in shared_memory_list:
        SharedMemory.cleanup(shm)
        atexit.unregister(SharedMemory.cleanup)
    shared_memory_list.clear()
    # Delete the trainer
    try:
        del trainer
    except Exception as e:
        log(ERROR, "Error deleting trainer", exc_info=e, stack_info=True)
    # Clean-up garbage collector and cuda cache
    gc.collect()
    torch.cuda.empty_cache()
    # Cleaning stale shared memory
    streaming.base.util.clean_stale_shared_memory()  # type: ignore[reportAttributeAccessIssue]
    if int(os.getenv("LOCAL_RANK", "")) == 0:
        eval_metrics |= {
            "client/eval_trainer_closing_time": (time.time_ns() - start_time) * 1e-9
        }

    # Return the evaluation metrics
    return 0.0, num_samples, eval_metrics
