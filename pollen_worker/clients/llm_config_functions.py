"""Provides functionality for manipulating MosaicML configs."""

import copy
from pathlib import Path
from logging import INFO, WARN, WARNING

import torch
from composer.utils.file_helpers import validate_given_remote_path
from flwr.common.logger import log


from omegaconf import DictConfig, ListConfig


from pollen_worker.utils import (
    get_n_cpu_cores,
    get_n_cuda_devices,
)
from dataclasses import dataclass, asdict


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


def patch_dataset_config(cfg: DictConfig) -> DictConfig:
    """Patch the dataset configuration for the client."""
    dataset_config: DictConfig = cfg.shared.pop("dataset_config", None)

    if dataset_config is None:
        raise ValueError("The `dataset_config` must be provided in the shared config.")

    if dataset_config.is_federated:
        if dataset_config.is_local:
            cfg.data_local = cfg.data_local.format(dataset_config.federated.local)
        else:
            cfg.data_remote = cfg.data_remote.format(dataset_config.federated.remote)
    elif dataset_config.is_local:
        cfg.data_local = cfg.data_local.format(dataset_config.centralized.local)
    else:
        cfg.data_remote = cfg.data_remote.format(dataset_config.centralized.remote)
    return cfg


def client_fit_set_data_config(cid: int | str, cfg: DictConfig) -> DictConfig:
    """Set the client data configuration for the client.

    Parameters
    ----------
    cid : int | str
        The client id.
    cfg : DictConfig
        The configuration object.

    Returns
    -------
    DictConfig
        The updated configuration object.
    """
    cfg = patch_dataset_config(cfg)

    streams_dict_list: list[int | dict] | None = cfg.shared.pop(
        "client_streams_list", {}
    ).get("client_streams_list", None)

    client_streams = (
        streams_dict_list[int(cid)] if streams_dict_list is not None else None
    )

    if client_streams is None or isinstance(client_streams, int):
        cid_to_map_to = client_streams if client_streams is not None else cid
        if cfg.data_remote is not None:  # type: ignore[union-attr]
            # Set the appropriate path given the `client_id`
            new_remote_path = (
                str(cfg.data_remote) + f"/client_{cid_to_map_to}"  # type: ignore[union-attr]
            )
            cfg = set_all_data_paths(cfg, new_remote_path, False)
        # Tie the local path to the client_id and the run_uuid
        new_local_path = (
            str(cfg.data_local) + f"/client_{cid_to_map_to}"  # type: ignore[union-attr]
        )
        cfg = set_all_data_paths(cfg, new_local_path)
        # Execute the fit function
    else:
        actual_streams = {
            key: StreamDict(**value) for key, value in client_streams["streams"].items()
        }
        for stream in actual_streams.values():
            if cfg.data_remote is not None and stream.remote is not None:
                stream.remote = (
                    str(cfg.data_remote) + f"/{stream.remote}"
                    if stream.remote
                    else None
                )
            if stream.local is not None:
                stream.local = str(cfg.data_local) + f"/{stream.local}"

        set_all_data_paths(cfg, None, False)
        set_all_data_paths(
            cfg,
            new_path=None,
        )
        test_streams = copy.deepcopy(actual_streams)
        for stream in test_streams.values():
            stream.split = "val"

        streams_dict = {name: asdict(stream) for name, stream in actual_streams.items()}
        test_streams_dict = {
            name: asdict(stream) for name, stream in test_streams.items()
        }
        cfg.streams = streams_dict

        cfg.train_loader.dataset.streams = streams_dict
        cfg.eval_loader.dataset.streams = test_streams_dict
    return cfg


def client_evaluate_set_data_config(cid: int | str, cfg: DictConfig) -> DictConfig:
    """Set the client data configuration for eval.

    # Only supports centralized evaluation for now.

    Parameters
    ----------
    cid : int | str
        The client id.
    cfg : DictConfig
        The configuration object.

    Returns
    -------
    DictConfig
        The updated configuration object.
    """
    cfg = patch_dataset_config(cfg)
    # TODO: Implement means of controlling evaluation
    # Set the appropriate path for the (centralized) val set
    if cfg.data_remote is not None:  # type: ignore[union-attr]
        # Extracts the parent folder from the remote path
        new_remote_path = "s3:/" + str(
            Path(
                str(cfg.data_remote).replace("s3:/", "")  # type: ignore[union-attr]
            ).parent  # type: ignore[union-attr]
        )
        cfg = set_all_data_paths(cfg, new_remote_path, False)
    # Tie the local path to the client_id and the run_uuid
    new_local_path = str(cfg.data_local) + "/val"  # type: ignore[union-attr]
    cfg = set_all_data_paths(cfg, new_local_path)

    return cfg


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
    cfg: DictConfig, server_round: int, n_steps_done: int, n_steps: int
) -> tuple[DictConfig, bool]:
    """Set the save and load path given the server round and client id."""
    # Flag to notify whether to skip this iteration or not
    skip_iteration = False
    # Set the save folder specifically for this client and this run
    if cfg.save_folder is not None:  # type: ignore[union-attr]
        try:
            log(INFO, "Looking for a checkpoint to load in %s", cfg.save_folder)
            if validate_given_remote_path(cfg.save_folder):
                cfg.load_path = (
                    cfg.save_folder + f"/ep0-ba{n_steps_done}-" + "rank{rank}.pt"
                )
                log(INFO, "Set checkpoint to load: %s", cfg.load_path)
            log(INFO, "Looking for the next checkpoint in %s", cfg.save_folder)
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


def set_client_tensorboard_logger(cfg: DictConfig, cid: int | str) -> DictConfig:
    """Set the tensorboard logger for the client."""
    # Set the tensorboard run name
    if cfg.loggers.tensorboard is not None:
        # Add the client id to the parameters
        cfg.loggers.tensorboard.client_id = cid
    return cfg


def set_all_data_paths(
    cfg: DictConfig, new_path: str | None, is_local: bool = True
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
        torch._dynamo.config.suppress_errors = True  # type: ignore[reportAttributeAccessIssue]

    if cfg.model.get("load_in_8bit", False):
        raise ValueError(
            "`load_in_8bit` is only supported for evaluation rather than training."
        )
