"""Implementation of the Flower's ServerApp for orchestrating federate learning."""

import copy
from logging import DEBUG, INFO
import os
import timeit
from typing import cast
import random
import time
import warnings


from flower_llm.server.broadcast_utils import broadcast_parameters_to_nodes
from flower_llm.server.evaluate_utils import evaluate_round
from flower_llm.server.fit_utils import fit_round
from flower_llm.server.init_utils import (
    get_centralized_run_parameters,
    initialize_round,
    resume_from_round,
)
from flower_llm.server.s3_utils import (
    delete_clients_checkpoints,
    delete_rounds,
    import_checkpoints,
    upload_server_checkpoint,
)
from flower_llm.server.server_util import (
    wait_for_nodes_to_connect,
)
import wandb
import flwr as fl
from flwr.common import (
    Context,
)
from flwr.common.typing import ConfigsRecordValues
from flwr.common.logger import log, update_console_handler
from flwr.server import Driver
from omegaconf import OmegaConf
from composer.loggers import RemoteUploaderDownloader

from flower_llm.conf.base_schema import BaseConfig
from flower_llm.strategy.dispatcher import dispatch_strategy
from flower_llm.strategy.fedadam import FedAdam
from flower_llm.strategy.fedmom import FedMom
from flower_llm.strategy.fednestorov import FedNesterov
from flower_llm.strategy.fedyogi import FedYogi
from flower_llm.utils import (
    create_remote_up_down,
    wandb_init,
)


# Fix the logger
update_console_handler(level=DEBUG, colored=False, timestamps=True)
# Filter user warning from configuration of MPT
warnings.filterwarnings(
    action="ignore",
    category=UserWarning,
    message=("If not using a Prefix Language Model*"),
    append=True,
)
# TODO: These don't work -- not sure why
# Filter deprecation warning from pkg_resources
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("Deprecated call to *"),
    append=True,
)
warnings.filterwarnings(
    action="ignore",
    category=DeprecationWarning,
    message=("pkg_resources is deprecated*"),
    append=True,
)

# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    """Implement the main function for the Flower ServerApp.

    Parameters
    ----------
    driver : fl.server.Driver
        The driver object for the server.
    context : fl.common.Context
        The context object for the server.
    """
    start_up_time = timeit.default_timer()
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("POLLEN_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
    # Load the configuration from the config file
    cfg = cast(BaseConfig, OmegaConf.load(save_path + "/config.yaml"))
    log(INFO, "Initializing Pollen Server")

    # Get FL setting parameters from the config file
    n_total_clients = cfg.fl.n_total_clients
    n_clients_per_round = cfg.fl.n_clients_per_round
    num_rounds = cfg.fl.n_rounds
    # Instantiate a PRNG
    rng = random.Random(cfg.seed)
    # Get Pollen parameters
    n_nodes = cfg.pollen.n_nodes
    # TODO: Exclude Pollen assignments implementation for now
    # placement_policy = cfg.pollen.placement_policy

    def pollen_fit_config(
        server_round: int, client_id: str | int
    ) -> dict[str, ConfigsRecordValues]:
        return {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.global_train_batch_size,
            "n_local_steps": cfg.fl.n_local_steps,
            "n_local_epochs": cfg.fl.n_local_epochs,
            "collaborative": cfg.pollen.fit_collaborative,
            "reset_checkpoint": cfg.fl.reset_checkpoint,
            "reset_optimizer": cfg.fl.reset_optimizer,
            "reset_dataset_state": cfg.fl.reset_dataset_state,
            "reset_timestamp": cfg.fl.reset_timestamp,
            "use_unigram_metrics": cfg.fl.use_unigram_metrics,
            "resize_vocab": str(cfg.fl.resize_vocab),
            "s3_comm_config": str(
                OmegaConf.to_container(cfg.s3_comm_config, resolve=True)
            ),
            "random_layers": str(cfg.fl.random_layers),
            "random_init_freq": str(cfg.fl.random_init_freq),
            "personalized_layers": str(cfg.fl.personalized_layers),
        }

    def pollen_evaluate_config(
        server_round: int, client_id: str | int
    ) -> dict[str, ConfigsRecordValues]:
        return {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.device_eval_batch_size,
            "collaborative": cfg.pollen.eval_collaborative,
            "use_unigram_metrics": cfg.fl.use_unigram_metrics,
            "resize_vocab": str(cfg.fl.resize_vocab),
            "s3_comm_config": str(
                OmegaConf.to_container(cfg.s3_comm_config, resolve=True)
            ),
        }

    strategy = dispatch_strategy(
        cfg,
    )
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    with wandb_init(  # type: ignore[union-attr,misc]
        cfg.use_wandb,
        **cfg.wandb.setup,  # type: ignore[reportCallIssue]
        settings=wandb.Settings(start_method="thread"),  # type: ignore[arg-type]
        config=wandb_config,  # type: ignore[arg-type]
    ) as _:
        # Create RemoteUploaderDownloader
        # TODO: We may want to have this as a function or a more dynamical object
        # that can change the bucket it's referring to
        remote_up_down: RemoteUploaderDownloader | None = None
        if cfg.pollen.checkpoint or cfg.use_s3_comm:
            remote_up_down = create_remote_up_down(
                bucket_name=cfg.s3_comm_config.bucket_name,
                prefix=f"{cfg.run_uuid}/server",
                run_uuid=cfg.run_uuid,
                num_attempts=cfg.s3_comm_config.num_attempts,
                client_config=OmegaConf.to_container(
                    cfg.s3_comm_config.backend_kwargs.client_config
                ),  # type: ignore[reportArgumentType, arg-type]
            )

        # Import another experiment checkpoints for restoration
        if cfg.pollen.restore_run_uuid is not None:
            assert (
                remote_up_down is not None
            ), "Cannot restore without a RemoteUploaderDownloader object"
            import_checkpoints(cfg=cfg, remote_up_down=remote_up_down)

        # Resume experiment from a previously saved checkpoint
        if cfg.pollen.resume_round is not None:
            assert (
                cfg.pollen.checkpoint is not None
            ), "Cannot resume if `cfg.pollen.checkpoint` is None"
            assert (
                remote_up_down is not None
            ), "Cannot resume without a RemoteUploaderDownloader object"
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = resume_from_round(cfg, remote_up_down)
            # Loop over the PRNG to get to the correct round
            for _ in range(start_round):
                sampled_clients = rng.sample(
                    range(n_total_clients), n_clients_per_round
                )
            sampled_clients = []
        elif cfg.pollen.restore_cent_run_uuid is not None:
            assert (
                remote_up_down is not None
            ), "Cannot restore without a RemoteUploaderDownloader object"
            parameters = get_centralized_run_parameters(copy.deepcopy(cfg))
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = initialize_round(cfg, remote_up_down, parameters=parameters)
        else:
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
                second_momentum_vector,
            ) = initialize_round(cfg, remote_up_down)

        # NOTE: The strategy needs to hold a copy of the initial parameters, but we want
        # it to free the reference it holds as an attribute. Then, similarly to the
        # `initialize_parameters()` of FedAvg, we nullify such attribute
        if strategy.initial_parameters:
            strategy.initial_parameters = None
        # NOTE: Since we initialized the strategy object before creating the parameters,
        # we must assign to the strategy attributes the parameters we got from the
        # initialization
        if isinstance(strategy, FedNesterov | FedMom | FedYogi | FedAdam):
            strategy.parameters = parameters
        if isinstance(strategy, FedNesterov | FedMom | FedYogi | FedAdam):
            assert momentum_vector is not None, "Momentum vector must be initialized"
            strategy.momentum_vector = momentum_vector
        else:
            momentum_vector = None
        if isinstance(strategy, FedYogi | FedAdam):
            assert (
                second_momentum_vector is not None
            ), "Second momentum vector must be initialized"
            strategy.second_momentum_vector = second_momentum_vector
        else:
            second_momentum_vector = None

        # Wait for the minimum number of nodes to connect
        wait_for_nodes_to_connect(driver, n_nodes)

        log(
            INFO,
            "Start-up time for the server is %s",
            timeit.default_timer() - start_up_time,
        )
        # Run federated learning for number of rounds
        log(INFO, "FL starting from round %s", start_round + 1)
        start_time = timeit.default_timer()

        # Broadcast model parameters to all NodeManagers
        broadcast_time = time.time_ns()
        all_node_ids = driver.get_node_ids()
        broadcast_parameters_to_nodes(
            driver=driver,
            parameters=parameters,
            node_ids=all_node_ids,
            current_round=start_round,
            remote_uploader_downloader=remote_up_down,
            use_s3_comm=cfg.use_s3_comm,
            use_shm=cfg.use_shm,
        )
        history.add_metrics_centralized(
            server_round=start_round + 1,
            metrics={
                "server/broadcast_pre_time": (time.time_ns() - broadcast_time) * 1e-9
            },
        )
        if cfg.fl.eval_fl is not None:
            # Launch the evaluate process for the starting round
            sampled_clients = [0]
            history = evaluate_round(
                driver=driver,
                sampled_clients=sampled_clients,
                evaluate_config_fn=pollen_evaluate_config,
                all_node_ids=all_node_ids,
                current_round=start_round,
                client_state=client_state,
                server_steps_cumulative=server_steps_cumulative,
                cfg=cfg,
                strategy=strategy,
                history=history,
            )
        # Nullify assignments
        sampled_clients = []

        # Federated learning loop
        for current_round in range(start_round + 1, num_rounds + 1):
            start_round_time = time.time_ns()
            log(DEBUG, f"Commencing server round {current_round}")

            # Check NodeManagers health
            first_check_nm_time = time.time_ns()
            all_node_ids = driver.get_node_ids()
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/first_check_nm_time": (time.time_ns() - first_check_nm_time)
                    * 1e-9
                },
            )

            # List of sampled Client IDs in this round
            sampled_clients = rng.sample(range(n_total_clients), n_clients_per_round)
            log(DEBUG, f"Sampled {len(sampled_clients)} Client IDs: {sampled_clients}")

            # Launch the federated fit process
            (
                parameters,
                client_state,
                server_steps_cumulative,
                history,
            ) = fit_round(
                sampled_clients=sampled_clients,
                all_node_ids=all_node_ids,
                driver=driver,
                fit_config_fn=pollen_fit_config,
                current_round=current_round,
                client_state=client_state,
                server_steps_cumulative=server_steps_cumulative,
                cfg=cfg,
                strategy=strategy,
                remote_up_down=remote_up_down,
                history=history,
                parameters=parameters,
            )
            # Nullify sampled clients
            sampled_clients = []

            # Broadcast model parameters to all NodeManagers
            broadcast_time = time.time_ns()
            broadcast_parameters_to_nodes(
                driver=driver,
                parameters=parameters,
                node_ids=all_node_ids,
                current_round=current_round,
                remote_uploader_downloader=remote_up_down,
                use_s3_comm=cfg.use_s3_comm,
                use_shm=cfg.use_shm,
            )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/broadcast_post_time": (time.time_ns() - broadcast_time)
                    * 1e-9
                },
            )

            # Check for changes in connected NodeManagers
            second_check_nm_time = time.time_ns()
            all_node_ids = driver.get_node_ids()
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/second_check_nm_time": (
                        time.time_ns() - second_check_nm_time
                    )
                    * 1e-9
                },
            )
            if cfg.fl.eval_fl is not None and current_round % cfg.fl.eval_fl == 0:
                # Launch the evaluate process
                sampled_clients = [0]
                history = evaluate_round(
                    driver=driver,
                    sampled_clients=sampled_clients,
                    evaluate_config_fn=pollen_evaluate_config,
                    all_node_ids=all_node_ids,
                    current_round=current_round,
                    client_state=client_state,
                    server_steps_cumulative=server_steps_cumulative,
                    cfg=cfg,
                    strategy=strategy,
                    history=history,
                )
            # Nullify assignments
            sampled_clients = []

            # Save the checkpoint to S3 Object Store
            if cfg.pollen.checkpoint or cfg.use_s3_comm:
                assert (
                    remote_up_down is not None
                ), "Cannot checkpoint without a RemoteUploaderDownloader object"
                upload_server_checkpoint(
                    parameters=parameters,
                    history=history,
                    current_round=current_round,
                    current_time_elapsed=time_offset,
                    server_steps_cumulative=server_steps_cumulative,
                    momentum_vector=momentum_vector,
                    second_momentum_vector=second_momentum_vector,
                    client_state=client_state,
                    remote_up_down=remote_up_down,
                )

            # Log the time taken for the round
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/round_time": (time.time_ns() - start_round_time) * 1e-9
                },
            )
            # Remove old clients checkpoints from the S3 Object Store
            delete_clients_checkpoints(
                run_uuid_path=f"s3://checkpoints/{cfg.run_uuid}",
            )
            # Remove old server checkpoints from the S3 Object Store
            delete_rounds(
                run_uuid_path=f"s3://checkpoints/{cfg.run_uuid}",
                state_keys=(
                    "state.bin",
                    "current_server_parameters",
                    "current_momentum_vector",
                ),
            )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time + time_offset
        log(INFO, "FL finished in %s", elapsed)

        log(DEBUG, "app_fit: losses_distributed %s", str(history.losses_distributed))
        log(
            DEBUG,
            "app_fit: metrics_distributed_fit %s",
            str(history.metrics_distributed_fit),
        )
        log(DEBUG, "app_fit: metrics_distributed %s", str(history.metrics_distributed))
        log(DEBUG, "app_fit: losses_centralized %s", str(history.losses_centralized))
        log(DEBUG, "app_fit: metrics_centralized %s", str(history.metrics_centralized))

        # Clean up checkpoints if asked to
        if cfg.cleanup_checkpoints:
            # Remove old clients checkpoints from the S3 Object Store
            delete_clients_checkpoints(
                run_uuid_path=f"s3://checkpoints/{cfg.run_uuid}",
                end_idx=None,
            )
            # Remove old server checkpoints from the S3 Object Store
            delete_rounds(
                run_uuid_path=f"s3://checkpoints/{cfg.run_uuid}",
                state_keys=(
                    "state.bin",
                    "current_server_parameters",
                    "current_momentum_vector",
                ),
                end_idx=None,
            )
