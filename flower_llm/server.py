"""TODO:."""

from logging import DEBUG, INFO
import os
import sys
import timeit
from typing import cast
import random
import time
import warnings


import numpy as np

from flower_llm.server_util import (
    broadcast_parameters_to_nodes,
    handle_evaluate_replies,
    message_collaborative,
    message_independent,
    handle_fit_replies,
    get_rr_assignment_function,
    import_checkpoints,
    initialize_round,
    resume_from_round,
    upload_server_checkpoint,
    wait_for_nodes_to_connect,
)
from flower_llm.strategy.aggregation import weighted_average
import wandb
import flwr as fl
from flwr.common import (
    Context,
    ndarrays_to_parameters,
    MessageType,
    Scalar,
)
from flwr.common.logger import log, update_console_handler
from flwr.server import Driver
from omegaconf import OmegaConf
from composer.loggers import RemoteUploaderDownloader

from flower_llm.conf.base_schema import BaseConfig
from flower_llm.strategy.dispatcher import dispatch_strategy
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
    """TODO."""
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

    def pollen_fit_config(server_round: int, client_id: str | int) -> dict[str, Scalar]:
        return {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.global_train_batch_size,
            "n_local_steps": cfg.fl.n_local_steps,
            "n_local_epochs": cfg.fl.n_local_epochs,
            "collaborative": cfg.pollen.fit_collaborative,
            "reset_optimizer": cfg.fl.reset_optimizer,
        }

    def pollen_evaluate_config(
        server_round: int, client_id: str | int
    ) -> dict[str, Scalar]:
        return {
            "cid": client_id,
            "server_round": server_round,
            "batch_size": cfg.llm_config.device_eval_batch_size,
            "collaborative": cfg.pollen.eval_collaborative,
        }

    strategy = dispatch_strategy(
        cfg,
        # NOTE: We put a fake array as it will be touched on again later
        initial_parameters=ndarrays_to_parameters([np.array([[0.0], [0.0]])]),
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        # These are not really important anymore with this new server
        fraction_fit=sys.float_info.min,
        fraction_evaluate=sys.float_info.min,
        min_fit_clients=n_clients_per_round,
        min_available_clients=n_clients_per_round,
        min_evaluate_clients=1,
        accept_failures=False,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
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
            ) = resume_from_round(cfg, remote_up_down)
        else:
            (
                parameters,
                history,
                start_round,
                time_offset,
                server_steps_cumulative,
                client_state,
                momentum_vector,
            ) = initialize_round(cfg, remote_up_down)

        # TODO: Lorenzo Reconcile initialization of parameters and momentum vector
        # with what the Strategy does,
        # NOTE: Calling the `get_initial_parameters` method to for consistently
        # freeing up the memory allocated for the initial parameters in strategy.
        # parameters = get_initial_parameters(strategy)

        # TODO: Fix this
        if start_round == 0:
            log(INFO, "Evaluating initial parameters")
            res = strategy.evaluate(0, parameters=parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])

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
        for current_round in range(start_round + 1, num_rounds + 1):
            start_round_time = time.time_ns()
            log(DEBUG, f"Commencing server round {current_round}")

            first_check_nm_time = time.time_ns()

            # Check NodeManagers health
            all_node_ids = driver.get_node_ids()

            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/first_check_nm_time": (time.time_ns() - first_check_nm_time)
                    * 1e-9
                },
            )

            broadcast_time = time.time_ns()
            # Broadcast model parameters to all NodeManagers
            broadcast_parameters_to_nodes(
                driver=driver,
                parameters=parameters,
                node_ids=all_node_ids,
                current_round=current_round,
            )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/broadcast_time": (time.time_ns() - broadcast_time) * 1e-9
                },
            )

            # List of sampled Client IDs in this round
            sampled_clients: list[int] = []
            sampled_clients = rng.sample(range(n_total_clients), n_clients_per_round)
            log(DEBUG, f"Sampled {len(sampled_clients)} Client IDs: {sampled_clients}")

            # Make an assignment function that round-robin divides the clients
            # NOTE: extend to add other types
            rr_assignment = get_rr_assignment_function(sampled_clients, all_node_ids)

            fit_round_time = time.time_ns()
            fit_replies = (
                message_collaborative(
                    driver=driver,
                    message_type=MessageType.TRAIN,
                    sampled_clients=sampled_clients,
                    gen_instructions=pollen_fit_config,
                    all_node_ids=all_node_ids,
                    current_round=current_round,
                )
                if cfg.pollen.fit_collaborative
                else message_independent(
                    driver=driver,
                    message_type=MessageType.TRAIN,
                    gen_instructions=pollen_fit_config,
                    all_node_ids=all_node_ids,
                    current_round=current_round,
                    assignment_function=rr_assignment,
                )
            )

            res_fit = handle_fit_replies(
                cfg,
                fit_replies,
                strategy,
                current_round,
                remote_up_down,
                client_state,
                server_steps_cumulative,
            )
            if res_fit is not None:
                (
                    parameters_aggregated,
                    fit_metrics,
                    (_raw_metrics, _failures),
                ) = res_fit
                parameters = (
                    parameters_aggregated
                    if parameters_aggregated is not None
                    else parameters
                )
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/fit_round_time": (time.time_ns() - fit_round_time) * 1e-9
                },
            )

            # TODO: Distributed evaluation

            evaluate_time = time.time_ns()
            res_cen = strategy.evaluate(current_round, parameters=parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/evaluate_time": (time.time_ns() - evaluate_time) * 1e-9
                },
            )

            # Check for changes in connected NodeManagers
            second_check_nm_time = time.time_ns()
            # Check NodeManagers health
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

            # Evaluate model on a sample of available clients
            evaluate_round_time = time.time_ns()

            eval_replies = (
                message_collaborative(
                    driver=driver,
                    message_type=MessageType.EVALUATE,
                    sampled_clients=sampled_clients,
                    gen_instructions=pollen_evaluate_config,
                    all_node_ids=all_node_ids,
                    current_round=current_round,
                )
                if cfg.pollen.eval_collaborative
                else message_independent(
                    driver=driver,
                    message_type=MessageType.EVALUATE,
                    gen_instructions=pollen_evaluate_config,
                    all_node_ids=all_node_ids,
                    current_round=current_round,
                    assignment_function=rr_assignment,
                )
            )

            res_fed = handle_evaluate_replies(
                cfg, eval_replies, strategy, current_round
            )
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/evaluate_round_time": (time.time_ns() - evaluate_round_time)
                    * 1e-9
                },
            )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/round_time": (time.time_ns() - start_round_time) * 1e-9
                },
            )

            # Save the checkpoint to S3 Object Store (w/ model parameters)
            if cfg.pollen.checkpoint or cfg.use_s3_comm:
                assert (
                    remote_up_down is not None
                ), "Cannot checkpoint without a RemoteUploaderDownloader object"
                upload_server_checkpoint(
                    parameters=parameters,
                    history=history,
                    current_round=start_round,
                    current_time_elapsed=time_offset,
                    server_steps_cumulative=server_steps_cumulative,
                    momentum_vector=momentum_vector,
                    client_state=client_state,
                    remote_up_down=remote_up_down,
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
