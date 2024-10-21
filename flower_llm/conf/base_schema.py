"""Base configuration schema."""

from typing import Any
from pydantic.dataclasses import dataclass
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
from enum import StrEnum, auto


@dataclass(config={"arbitrary_types_allowed": True})
class Centralized(DictConfig):
    """Centralized configuration.

    Attributes
    ----------
    store_init_model: bool = MISSING
        Whether to store the initial model
    store_final_model: bool = MISSING
        Whether to store the final model
    stream_id: str | None = MISSING
        Stream id to pass to the data configuration
    eval_only: bool = MISSING
        Whether to only execute the evaluation
    """

    store_init_model: bool = MISSING
    store_final_model: bool = MISSING
    stream_id: str | None = MISSING
    eval_only: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class Pollen(DictConfig):
    """Pollen configuration.

    Attributes
    ----------
    placement_policy: str = MISSING
        Placement policy for the clients: "rr", "srr", "bu", "su", "lbu", "lb", "llb"
    n_nodes: int = MISSING
        Number of nodes in the cluster
    saving_path: str = MISSING
        Path to save the models
    refresh_period: int = MISSING
        Refresh period for the client workers
    fit_collaborative: bool = MISSING
        Whether to use all workers on a node to fit one client
    eval_collaborative: bool = MISSING
        Whether to use all workers on a node to evaluate one client
    cpu_only: bool = MISSING
        Whether to use only CPU
    cpu_concurrency: int = MISSING
        Number of CPU workers
    checkpoint: bool = MISSING
        Whether to checkpoint the model
    restore_run_uuid: str = MISSING
        Run UUID to restore the model
    resume_round: int | None = MISSING
        Round to resume from, None implies start anew
        negative indices are counted from the last round
    """

    placement_policy: str = MISSING
    n_nodes: int = MISSING
    saving_path: str = MISSING
    refresh_period: int = MISSING
    fit_collaborative: bool = MISSING
    eval_collaborative: bool = MISSING
    cpu_only: bool = MISSING
    cpu_concurrency: int = MISSING
    checkpoint: bool = MISSING
    restore_run_uuid: str | None = MISSING
    resume_round: int | None = MISSING


class StrategyName(StrEnum):
    """Strategy type."""

    NESTOROV = auto()
    FEDMOM = auto()
    FEDAVG = auto()
    FEDYOGI = auto()
    FEDADAM = auto()
    NESTOROV_MATRIX = auto()


class StrategyKWArgs(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """StrategyKWArgs configuration."""


@dataclass(config={"arbitrary_types_allowed": True})
class FL(DictConfig):
    """Federated learning configuration.

    Attributes
    ----------
    n_total_clients: int = MISSING
        Number of clients
    n_clients_per_round: int = MISSING
        Number of clients per round
    n_rounds: int = MISSING
        Number of rounds
    reset_optimizer: bool = MISSING
        Whether to reset the local optimizer
    n_local_epochs: int = MISSING
        Number of local epochs
    n_local_steps: int = MISSING
        Number of local steps
    """

    n_total_clients: int = MISSING
    n_clients_per_round: int = MISSING
    n_rounds: int = MISSING
    reset_optimizer: bool = MISSING
    n_local_epochs: int = MISSING
    n_local_steps: int = MISSING

    ignore_failed_rounds: bool = MISSING
    accept_failures_cnt: int = MISSING
    eval_fl: bool = MISSING

    strategy_name: StrategyName = MISSING
    strategy_kwargs: StrategyKWArgs = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class ClientConfig(DictConfig):
    """Client configuration."""

    connect_timeout: int = MISSING
    read_timeout: int = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class BackendKwargs(DictConfig):
    """Backend configuration.

    Attributes
    ----------
    client_config: DictConfig
        Configuration for the client
    """

    client_config: ClientConfig = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class S3CommConfig(DictConfig):
    """S3 communication configuration.

    Attributes
    ----------
    bucket_name: str = MISSING
        Name of the S3 bucket
    num_attempts: int = MISSING
        Number of attempts
    backend_kwargs: BackendKwargs
        Backend configuration
    """

    bucket_name: str = MISSING
    num_attempts: int = MISSING
    backend_kwargs: BackendKwargs = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class WandbSetup(DictConfig):
    """Wand setup configuration.

    Attributes
    ----------
    project: str = MISSING
        Name of the project
    group: str = MISSING
        Name of the group
    tags: list[str] = MISSING
        List of tags
    entity: str = MISSING
        Name of the entity
    mode: str = MISSING
        Mode of the run: "online", "offline"
    name: str = MISSING
        Name of the run, {Config.run_uuid}
    resume: str = MISSING
        Weather to allow resumption
    id: str = MISSING
        ID of the run, {Config.run_uuid}
    allow_val_change: bool = MISSING
        Allows changing the value of the config when resuming
    """

    project: str = MISSING
    group: str = MISSING
    tags: list[str] = MISSING
    entity: str | None = MISSING
    mode: str = MISSING
    name: str = MISSING
    resume: str = MISSING
    id: str = MISSING
    allow_val_change: bool = MISSING


@dataclass(config={"arbitrary_types_allowed": True})
class Wandb(DictConfig):
    """Wandb configuration.

    Attributes
    ----------
    setup: WandbSetup
        Wandb setup configuration
    """

    setup: WandbSetup = MISSING


class Dataset(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """Dataset configuration."""


class LLMConfig(dict[str, Any], DictConfig):  # type: ignore[reportIncompatibleMethodOverride,misc]
    """LLM configuration."""


@dataclass(config={"arbitrary_types_allowed": True})
class BaseConfig(DictConfig):
    """Base configuration.

    Attributes
    ----------
    run_uuid: str = MISSING
        Run UUID
    seed: int = MISSING
        Seed
    is_test: bool = MISSING
        Whether to run in test mode
    pretrained_model_path: str = MISSING
        Path to the pretrained model
    wte_parameters_path: str = MISSING
        Path to the model from which to take the WTE parameters
    centralized: Centralized = MISSING
        Centralized configuration
    pollen: Pollen
        Pollen configuration
    fl: FL
        Federated learning configuration
    use_shm: bool = MISSING
        Whether to use SharedMemory communication
    use_s3_comm: bool = MISSING
        Whether to use S3 communication
    s3_comm_config: S3CommConfig
        S3 communication configuration
    use_wandb: bool = MISSING
        Whether to use Wandb
    cleanup_checkpoints: bool = MISSING
        Whether to clean up all the checkpoints at the end
    wandb: Wandb
        Wandb configuration
    """

    run_uuid: str = MISSING
    seed: int = MISSING
    is_test: bool = MISSING
    pretrained_model_path: str | None = MISSING
    wte_parameters_path: str | None = MISSING
    centralized: Centralized = MISSING
    pollen: Pollen = MISSING
    fl: FL = MISSING
    use_s3_comm: bool = MISSING
    use_shm: bool = MISSING
    s3_comm_config: S3CommConfig = MISSING
    use_wandb: bool = MISSING
    cleanup_checkpoints: bool = MISSING
    wandb: Wandb = MISSING

    # NOTE: MosaicML specific, do not include in the base schema
    llm_config: LLMConfig = MISSING
    dataset: Dataset = MISSING


def register_config(name: "str") -> None:
    """Register the base configuration schema."""
    cs = ConfigStore.instance()
    cs.store(name=name, node=BaseConfig)
