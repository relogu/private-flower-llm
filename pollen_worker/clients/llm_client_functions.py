"""Provides the internal fucntions used by the LLM client."""

import atexit
import copy
import gc
import logging
import os
import warnings
from collections import OrderedDict
from contextlib import _GeneratorContextManager
from logging import DEBUG, ERROR, INFO, WARN, WARNING
from typing import Any

import streaming
import torch
from composer import Callback, ComposerModel, Evaluator, Trainer
from composer.devices import DeviceGPU
from composer.profiler import JSONTraceHandler, Profiler, TraceHandler, cyclic_schedule
from composer.utils import dist, reproducibility
from composer.utils.file_helpers import validate_given_remote_path
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from llmfoundry.data.dataloader import build_dataloader
from llmfoundry.models.hf.hf_causal_lm import ComposerHFCausalLM
from llmfoundry.models.hf.hf_prefix_lm import ComposerHFPrefixLM
from llmfoundry.models.hf.hf_t5 import ComposerHFT5
from llmfoundry.models.inference_api_wrapper.openai_causal_lm import (
    OpenAICausalLMEvalWrapper,
    OpenAIChatAPIEvalWrapper,
)
from llmfoundry.models.mpt.modeling_mpt import ComposerMPTCausalLM, MPTForCausalLM
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_icl_data_and_gauntlet,
    build_logger,
    build_optimizer,
    build_scheduler,
    build_tokenizer,
)
from llmfoundry.utils.config_utils import (
    pop_config,
    process_init_device,
    update_batch_size_info,
)
from omegaconf import DictConfig, ListConfig, OmegaConf
from streaming.base.shared.memory import SharedMemory, shared_memory_list
from transformers import PreTrainedTokenizerBase

from pollen_worker.utils import get_n_cpu_cores, get_n_cuda_devices, l1_norm

COMPOSER_MODEL_REGISTRY = {
    "mpt_causal_lm": ComposerMPTCausalLM,
    "hf_causal_lm": ComposerHFCausalLM,
    "hf_prefix_lm": ComposerHFPrefixLM,
    "hf_t5": ComposerHFT5,
    "openai_causal_lm": OpenAICausalLMEvalWrapper,
    "openai_chat": OpenAIChatAPIEvalWrapper,
}


def set_client_save_and_load_path(cfg: DictConfig, cid: int | str) -> DictConfig:
    """Set the save and load path given the server round and client id."""
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        cfg.save_folder = (  # type: ignore[union-attr]
            cfg.save_folder
            + f"/{cfg.run_name}"
            + "/client_"  # type: ignore[union-attr]
            + str(cid)  # type: ignore[union-attr]
        )
    return cfg


def set_client_load_path(
    cfg: DictConfig, server_round: int, local_steps: str
) -> tuple[DictConfig, bool]:
    """Set the save and load path given the server round and client id."""
    # Falg to notify whther to skip this iteration or not
    skip_iteration = False
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        try:
            log(INFO, "Looking for a checkpoint to load in %s", cfg.save_folder)
            if validate_given_remote_path(cfg.save_folder):
                n_steps_done = int(
                    int(local_steps.replace("ba", "")) * (server_round - 1)
                )
                cfg.load_path = (
                    cfg.save_folder + f"/ep0-ba{n_steps_done}-" + "rank{rank}.pt"
                )
                log(INFO, "Set checkpoint to load: %s", cfg.load_path)
            log(INFO, "Looking for the next checkpoint in %s", cfg.save_folder)
            n_steps = int(int(local_steps.replace("ba", "")) * (server_round))
            path_to_check = str(cfg.save_folder + f"/ep0-ba{n_steps}-" + "rank0.pt")
            skip_iteration = validate_given_remote_path(path_to_check)
            if skip_iteration:
                cfg.load_path = cfg.save_folder + f"/ep0-ba{n_steps}-" + "rank{rank}.pt"
                log(
                    INFO,
                    "Skipping training iteration as checkpoint %s already exists.",
                    cfg.load_path,
                )
                # NOTE: Don't re-save the checkpoint when resuming mid-round
                cfg.save_folder = None
        except Exception as e:
            log(WARNING, "The `load_path` wasn't set.", exc_info=e)
            # log(
            #     DEBUG,
            #     "Error running `os.listdir` for folder %s",
            #     self.cfg.save_folder,
            #     exc_info=e,
            #     stack_info=True,
            # )
    return cfg, skip_iteration


def set_client_wandb_logger(cfg: DictConfig, cid: int | str) -> DictConfig:
    """Set the wandb logger for the client."""
    # Set the wandb run name
    if cfg.loggers.wandb is not None:
        # Get the server run name
        run_name = cfg.loggers.wandb.init_kwargs.name
        # Add the client id to the run name
        new_run_name = run_name + f"_client_{cid}"
        server_id = cfg.loggers.wandb.init_kwargs.id
        cfg.loggers.wandb.init_kwargs.id = server_id + f"_client_{cid}"
        # Set the new run name
        cfg.loggers.wandb.init_kwargs.name = new_run_name
    return cfg


def set_all_data_paths(
    cfg: DictConfig, new_path: str, is_local: bool = True
) -> DictConfig:
    """Set the data paths for all dataloaders in the config."""
    if is_local:
        cfg.data_local = new_path
        if cfg.train_loader is not None:
            cfg.train_loader.dataset.local = new_path
        cfg.eval_loader.dataset.local = new_path
    else:
        cfg.data_remote = new_path
        if cfg.train_loader is not None:
            cfg.train_loader.dataset.remote = new_path
        cfg.eval_loader.dataset.remote = new_path
    return cfg


def set_n_workers_dataloaders(
    cfg: DictConfig,
    n_workers: int = -1,
    cap: int = 32,
) -> DictConfig:
    """Set the `n_workers` parameter for all dataloaders in the config."""
    if n_workers < 0:
        n_workers = get_n_cpu_cores()
    n_cuda_device = get_n_cuda_devices()
    if n_cuda_device > 0:
        n_workers = n_workers // n_cuda_device
    cfg.train_loader.num_workers = min(n_workers, cap)
    cfg.eval_loader.num_workers = min(n_workers, cap)
    return cfg


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

    if cfg.model.get("fc_type", "torch") == "te" or "te" in cfg.model.get(
        "ffn_config", {}
    ).get("ffn_type", "mptmlp"):
        fsdp_config = cfg.get("fsdp_config", None)
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
        torch._dynamo.config.suppress_errors = True

    if cfg.model.get("load_in_8bit", False):
        raise ValueError(
            "`load_in_8bit` is only supported for evaluation rather than training."
        )


def build_composer_model(
    model_cfg: DictConfig, tokenizer: PreTrainedTokenizerBase
) -> Any:
    """Build the Composer model gievn the config and tokenizer."""
    warnings.filterwarnings(
        action="ignore",
        message="Torchmetrics v0.9 introduced a new argument class property",
    )
    if model_cfg.name not in COMPOSER_MODEL_REGISTRY:
        raise ValueError(f"Not sure how to build model with name={model_cfg.name}")
    return COMPOSER_MODEL_REGISTRY[model_cfg.name](model_cfg, tokenizer)


def build_composer_peft_model(
    pretrained_model_name_or_path: str,
    lora_args: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> ComposerHFCausalLM:
    """Build the Composer model with Lora modules (if asked for)."""
    try:
        from peft import LoraConfig, get_peft_model  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Error importing from peft. Please verify that peft and peft utils "
            "are installed by running `pip install -e .[peft]` from `llm-foundry/`. "
            f"Error encountered: {e}"
        ) from e

    # 1) loads a hf model, 2) adds peft modules, 3) wraps it in a ComposerHFCausalLM.
    # log(INFO, "Building Lora config...")
    lora_cfg = LoraConfig(**lora_args)

    # log(INFO, "Building model from HuggingFace checkpoint...")
    model = MPTForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True
    )
    # log(INFO, "Model built!")

    # log(INFO, "Adding Lora modules...")
    model = get_peft_model(model, lora_cfg)
    # log(INFO, "Lora modules added!")

    model = ComposerHFCausalLM(model, tokenizer)

    return model


def print_trainable_parameters(model: torch.nn.Module) -> None:
    """Print the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    log(
        INFO,
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable params (%): {100 * trainable_params / all_param}",
    )


def _get_model_for_trainer(
    init_context: _GeneratorContextManager,
    tokenizer: PreTrainedTokenizerBase,
    model_config: DictConfig,
    lora_config: dict[str, Any] | None,
) -> ComposerModel:
    # Build Model
    # log(INFO, "Initializing model...")
    with init_context:
        if lora_config is not None:  # frozen model + trainable lora modules
            model: ComposerHFCausalLM = build_composer_peft_model(
                model_config.pretrained_model_name_or_path,
                lora_config["args"],
                tokenizer,
            )
            print_trainable_parameters(model)  # should not be 100%
        else:  # standard model
            model = build_composer_model(model_config, tokenizer)

        if model_config.get("master_weights_dtype") in {"bf16", "bfloat16"}:
            model = model.to(dtype=torch.bfloat16)
        elif model_config.get("master_weights_dtype") in {"f16", "float16"}:
            model = model.to(dtype=torch.float16)
        print_trainable_parameters(model)  # should not be 100%
    return model


def get_raw_model_parameters(
    _cfg: DictConfig,
) -> NDArrays:
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
    # Get LoRa config
    lora_config: dict[str, Any] | None = pop_config(
        _cfg, "lora", must_exist=False, default_value=None, convert=True
    )
    # Get model while forcing cpu to prevent any GPU allocation
    model_config.init_device = "cpu"
    model = _get_model_for_trainer(
        init_context=process_init_device(model_config, None),
        tokenizer=build_tokenizer(tokenizer_name, tokenizer_kwargs),
        model_config=model_config,
        lora_config=lora_config,
    )
    model.cpu()
    return [val.detach().to("cpu").numpy() for _, val in model.state_dict().items()]


def _get_trainer_object(
    _cfg: DictConfig,
) -> tuple[Trainer, bool, DictConfig]:
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
    # TODO: Resolve the linter suggestion here
    visible_devices = eval(os.getenv("APPOINTED_CUDA_DEVICE", "null"))  # noqa: PGH001
    if type(visible_devices) is int:
        device = DeviceGPU(device_id=int(visible_devices))
        log(DEBUG, f"Selecting device {visible_devices}, {device}")
    else:
        device = None

    # Get global and device batch size information from distributed/single node setting
    _cfg = update_batch_size_info(_cfg)
    logged_cfg.update(_cfg, merge=True)

    # Mandatory model training configs
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
    lora_config: dict[str, Any] | None = pop_config(
        _cfg, "lora", must_exist=False, default_value=None, convert=True
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
                INFO,
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
            INFO,
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
    if python_log_level is not None:
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
            **profiler_cfg,
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

    # Dataloaders
    # log(INFO, "Building train loader...")
    train_loader = None
    if train_loader_config is not None:
        train_loader = build_dataloader(
            train_loader_config,
            tokenizer,
            device_train_batch_size,
        )

    # Evaluation
    # log(INFO, "Building eval loader...")
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
            icl_seq_len if icl_seq_len else max_seq_len,
            icl_subset_num_batches,
        )
        evaluators.extend(icl_evaluators)

    if eval_gauntlet_callback is not None:
        callbacks.append(eval_gauntlet_callback)

    model = _get_model_for_trainer(
        init_context,
        tokenizer,
        model_config,
        lora_config,
    )

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
            evaluators.insert(0, eval_loader)  # Put the base eval_loaders first

    # Build the Trainer
    # log(INFO, "Building trainer...")
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
        precision=precision,
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
    return trainer, eval_first, logged_cfg


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
    return get_raw_model_parameters(copy.deepcopy(cfg))


def get_parameters_from_state(
    config: Config,
    trainer: Trainer,
) -> NDArrays:
    """Implement how to get parameters."""
    return [
        val.detach().to("cpu").numpy()
        for _, val in trainer.state.model.state_dict().items()
    ]


def set_parameters_to_state(
    parameters: NDArrays,
    trainer: Trainer,
) -> None:
    """Implement how to set parameters in the case of an LLM."""
    keys = list(trainer.state.model.state_dict().keys())
    params_dict = zip(keys, parameters, strict=False)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # NOTE: We may want to try strict=False
    trainer.state.model.load_state_dict(state_dict, strict=True)
    del state_dict


def llm_fit(
    parameters: NDArrays,
    config: dict,
    cfg: DictConfig,
) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any]]:
    """Implement the fit step using MosaicML codebase."""
    # Set the loading path
    cfg, skip_iteration = set_client_load_path(
        cfg, config["server_round"], cfg["local_steps"]
    )
    # Automatically setting the `n_workers` parameter based on CPU available
    cfg = set_n_workers_dataloaders(cfg)  # type: ignore[union-attr]
    if config["reset_optimizer"]:
        # Ignoring the optimizer state if loading a checkpoint
        cfg.load_ignore_keys = ["*optim*"]  # type: ignore[union-attr]
        # Ignoring the optimizer state when saving a checkpoint
        cfg.save_ignore_keys = ["*optim*"]  # type: ignore[union-attr]
    # Extract configs to build the trainer
    trainer, eval_first, _logged_cfg = _get_trainer_object(
        _cfg=cfg,
    )
    # log(INFO, f"Trainer config: {logged_cfg}")
    # NOTE: Skipping a few steps if the checkpoint already exists
    if not skip_iteration:
        # Set the parameters
        if parameters is not None and not skip_iteration:
            # log(INFO, "Initializing model...")
            set_parameters_to_state(parameters, trainer)
        # Eval first if requested
        if eval_first and trainer.state.timestamp.batch.value == 0:
            trainer.eval()
        # log(INFO, "Starting training...")
        # Prevent to run any evaluator
        trainer.state.evaluators = None
        # Execute fit step for the appointed duration
        try:
            trainer.fit(duration=0 if skip_iteration else cfg["local_steps"])
        except Exception as e:
            log(ERROR, "llm_fit::trainer.fit", exc_info=e, stack_info=True)
    # Retrieve number of samples trained
    # NOTE: We assume all the clients train with the same batch size,
    # so we just consider the number of local steps
    n_samples_trained = int(str(cfg["local_steps"]).replace("ba", ""))
    train_metrics: dict[str, Scalar] = {}
    # Retrieve training metrics
    train_metrics |= {
        k: v.detach().cpu().item()  # type: ignore[attr-defined]
        for k, v in trainer.state.train_metric_values.items()
    }
    log(INFO, f"Train metrics: {train_metrics}")
    # Retrieve model parameters
    model_parameters = get_parameters_from_state({}, trainer)

    # Compute the norm of the pseudo-gradient
    per_layer_norm_of_pseudo_gradient = [
        l1_norm([x - y]) for x, y in zip(parameters, model_parameters, strict=False)
    ]
    for i, plnopg in enumerate(per_layer_norm_of_pseudo_gradient):
        train_metrics |= {f"client/layer_{i}/l1_norm_of_pseudo_gradient": plnopg}
    norm_of_pseudo_gradient: float = sum(per_layer_norm_of_pseudo_gradient)
    train_metrics |= {"client/l1_norm_pseudo_gradient": norm_of_pseudo_gradient}

    # Close the trainer
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
    gc.collect()
    torch.cuda.empty_cache()

    # Cleaning stale shared memory
    streaming.base.util.clean_stale_shared_memory()

    return model_parameters, n_samples_trained, train_metrics


def llm_eval(
    parameters: NDArrays,
    config: dict,
    cfg: DictConfig,
) -> tuple[float, int, dict[str, Scalar]]:
    """Implement the fit step using MosaicML codebase."""
    # Automatically setting the `n_workers` parameter based on CPU available
    cfg = set_n_workers_dataloaders(cfg)  # type: ignore[union-attr]
    # Force llm_config params to select the centralised eval set
    cfg.train_loader = None  # type: ignore[union-attr]
    # NOTE: Trying to exclude checkpointing for eval
    cfg.autoresume = False  # type: ignore[union-attr]
    cfg.save_folder = None  # type: ignore[union-attr]
    cfg.load_path = None  # type: ignore[union-attr]
    cfg.loggers = None  # type: ignore[union-attr]
    # Extract configs to build the trainer
    trainer, _, _ = _get_trainer_object(
        _cfg=cfg,
    )
    # Set the parameters
    # log(INFO, "Initializing model...")
    set_parameters_to_state(parameters, trainer)
    gc.collect()
    torch.cuda.empty_cache()
    # log(INFO, "Starting evaluation...")
    trainer.eval()
    # Retrieve number of samples evaluated
    num_samples = trainer.state.eval_timestamp._sample.value
    # Retrieve evaluation metrics
    eval_metrics = {
        "Val" + k: v.detach().cpu().item()  # type: ignore[attr-defined]
        for k, v in trainer.state.eval_metric_values.items()
    }

    # Close the trainer
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

    gc.collect()
    torch.cuda.empty_cache()

    # Cleaning stale shared memory
    streaming.base.util.clean_stale_shared_memory()

    # Return the evaluation metrics
    return 0.0, num_samples, eval_metrics
