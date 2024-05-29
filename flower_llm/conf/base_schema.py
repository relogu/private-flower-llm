"""Base configuration schema."""

from pydantic.dataclasses import dataclass
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore


@dataclass
class Pollen(DictConfig):
    """Pollen configuration.

    Attributes
    ----------
    placement_policy: str
        Placement policy for the clients: "rr", "srr", "bu", "su", "lbu", "lb", "llb"
    n_nodes: int
        Number of nodes in the cluster
    server_address: str
        Address of the server
    saving_path: str
        Path to save the models
    refresh_period: int
        Refresh period for the client workers
    fit_collaborative: bool
        Whether to use all workers on a node to fit one client
    eval_collaborative: bool
        Whether to use all workers on a node to evaluate one client
    cpu_only: bool
        Whether to use only CPU
    cpu_concurrency: int
        Number of CPU workers
    checkpoint: bool
        Whether to checkpoint the model
    restore_run_uuid: str
        Run UUID to restore the model
    resume_round: int
        Round to resume from
    """

    placement_policy: str
    n_nodes: int
    server_address: str
    saving_path: str
    refresh_period: int
    fit_collaborative: bool
    eval_collaborative: bool
    cpu_only: bool
    cpu_concurrency: int
    checkpoint: bool
    restore_run_uuid: str | None
    resume_round: int


@dataclass
class FL(DictConfig):
    """Federated learning configuration.

    Attributes
    ----------
    n_total_clients: int
        Number of clients
    n_clients_per_round: int
        Number of clients per round
    n_rounds: int
        Number of rounds
    reset_optimizer: bool
        Whether to reset the local optimizer
    n_local_epochs: int
        Number of local epochs
    n_local_steps: int
        Number of local steps
    rescale_global_model: bool
        Whether to rescale the norm of the global model
        to match that of the average local model
    rescale_momentum_vector: bool
        Whether to rescale the momentum vector
    server_learning_rate: float
        Learning rate of the server
    server_momentum: float
        Momentum of the server
    """

    n_total_clients: int
    n_clients_per_round: int
    n_rounds: int
    reset_optimizer: bool
    n_local_epochs: int
    n_local_steps: int
    rescale_global_model: bool
    rescale_momentum_vector: bool
    server_learning_rate: float
    server_momentum: float


@dataclass
class BackendKwargs(DictConfig):
    """Backend configuration.

    Attributes
    ----------
    client_config: DictConfig
        Configuration for the client
    """

    client_config: DictConfig


@dataclass
class S3CommConfig(DictConfig):
    """S3 communication configuration.

    Attributes
    ----------
    bucket_name: str
        Name of the S3 bucket
    num_attempts: int
        Number of attempts
    backend_kwargs: BackendKwargs
        Backend configuration
    """

    bucket_name: str
    num_attempts: int
    backend_kwargs: BackendKwargs


@dataclass
class WandbSetup(DictConfig):
    """Wand setup configuration.

    Attributes
    ----------
    project: str
        Name of the project
    group: str
        Name of the group
    tags: List[str]
        List of tags
    entity: str
        Name of the entity
    mode: str
        Mode of the run: "online", "offline"
    name: str
        Name of the run, {Config.run_uuid}
    resume: str
        Weather to allow resumption
    id: str
        ID of the run, {Config.run_uuid}
    allow_val_change: bool
    """

    project: str
    group: str
    tags: list[str]
    entity: str | None
    mode: str
    name: str
    resume: str
    id: str
    allow_val_change: bool
    start_eval: int


@dataclass
class Wandb(DictConfig):
    """Wandb configuration.

    Attributes
    ----------
    setup: WandbSetup
        Wandb setup configuration
    """

    setup: WandbSetup


@dataclass
class BaseConfig(DictConfig):
    """Base configuration.

    Attributes
    ----------
    run_uuid: str
        Run UUID
    seed: int
        Seed
    is_test: bool
        Whether to run in test mode
    pretrained_model_path: str
        Path to the pretrained model
    store_init_model: bool
        Whether to store the initial model
    store_final_model: bool
        Whether to store the final model
    pollen: Pollen
        Pollen configuration
    fl: FL
        Federated learning configuration
    defaults: List[str]
        List of default configurations
    use_s3_comm: bool
        Whether to use S3 communication
    s3_comm_config: S3CommConfig
        S3 communication configuration
    use_wandb: bool
        Whether to use Wandb
    wandb: Wandb
        Wandb configuration
    """

    run_uuid: str
    seed: int
    is_test: bool
    pretrained_model_path: str | None
    store_init_model: bool
    store_final_model: bool
    pollen: Pollen
    fl: FL
    defaults: list[str]
    use_s3_comm: bool
    s3_comm_config: S3CommConfig
    use_wandb: bool
    wandb: Wandb

    # NOTE: MosaicML specific, do not include in the base schema
    llm_config: DictConfig
    dataset: DictConfig


def register_config(name: "str") -> None:
    """Register the base configuration schema."""
    cs = ConfigStore.instance()
    cs.store(name=name, node=BaseConfig)
