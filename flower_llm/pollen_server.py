"""Pollen server."""

import ast
import concurrent.futures
from dataclasses import asdict, dataclass
import pickle
import sys
import time
import timeit
from collections.abc import Callable, Generator
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
from typing import Any, Literal, cast
from flower_llm.conf.base_schema import S3CommConfig

import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
from flwr.client import Client
from flwr.client.numpy_client import NumPyClient
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import GetPropertiesIns, Properties
from flwr.server import Server
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import _handle_finished_future_after_evaluate  # noqa: PLC2701
from flwr.server.server import evaluate_client, fit_client
from flwr.server.strategy import FedAvg
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path

from flower_llm.clients.empty_virtual_client import EmptyVirtualClient
from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
)
from flower_llm.placements import (
    get_placement_fn,
    get_pollen_models,
    skim_clients_training_stats,
)
from flower_llm.pollen_client_manager import PollenClientManager
from flower_llm.resources_manager import Node
from flower_llm.strategy.rs_nesterov import FedNesterov
from flower_llm.utils import (
    IntentionalClientDropoutError,
    download_file_from_s3,
    dump_model_parameters_to_file,
    get_table_from_pyarrow_buffer,
    load_model_parameters_from_file,
    obtain_sorted_runs,
    upload_file_to_s3,
    ClientState,
)

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[tuple[ClientProxy, FitRes] | BaseException],
]
EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[tuple[ClientProxy, EvaluateRes] | BaseException],
]
ReconnectResultsAndFailures = tuple[
    list[tuple[ClientProxy, DisconnectRes]],
    list[tuple[ClientProxy, DisconnectRes] | BaseException],
]

GetPropResultsAndFailures = tuple[
    list[tuple[ClientProxy, Node]],
    list[tuple[ClientProxy, Node] | BaseException],
]

ClientLike = Client | NumPyClient


class TooManyFailuresError(Exception):
    """Exception raised when a client is dropped out of the tree."""


@dataclass
class AuxiliaryFitClientsResults:
    """Dataclass for the auxiliary results of the fit_clients function."""

    pollen_models: dict[str, Any] | None
    correction_tables: dict[str, pa.Table] | None
    fit_pollen_models_time: float


class PollenServer(Server):
    """Flower server."""

    def __init__(
        self,
        *,
        run_uuid: str,
        client_manager: PollenClientManager,
        cids: dict[str | int, int],
        client_fn: Callable[[int], ClientLike],
        strategy: FedAvg | None = None,
        placement_policy: str = "rr",
        saving_path: Path | None = None,
        history: History | None = None,
        num_nodes: int = 1,
        accept_failures_cnt: (
            int | None
        ) = 0,  # how many failures to accept, None for infinite
        ignore_failed_rounds: bool = False,
        print_failures: bool = True,
        print_intentional_failures: bool = True,
        use_s3_comm: bool = False,
        s3_comm_config: S3CommConfig | None = None,
        checkpoint: bool = False,
        resume_round: int | None = None,
        restore_run_uuid_and_steps_per_round: tuple[str, int] | None = None,
    ) -> None:
        self.start_up_time = timeit.default_timer()
        self._client_manager: PollenClientManager = client_manager  # type: ignore[reportIncompatibleVariableOverride]
        self.cids = cids
        self.client_fn = client_fn
        self.placement_policy = placement_policy
        self.placement_fn = get_placement_fn(self.placement_policy)
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: FedAvg = strategy if strategy is not None else FedAvg()
        _check_strategy_for_pollen(self.strategy)
        self.on_fit_config: Callable[[int], dict[str, Scalar]] = (
            conf_fn
            if (conf_fn := self.strategy.on_fit_config_fn) is not None
            else lambda _: {}
        )
        self.on_evaluate_config: Callable[[int], dict[str, Scalar]] = (
            conf_fn
            if (conf_fn := self.strategy.on_evaluate_config_fn) is not None
            else lambda _: {}
        )
        self.max_workers: int | None = None
        self.nodes_dict: dict[str, tuple[ClientProxy, Node]] = {}
        self.saving_path = saving_path
        self.clients_training_stats: pa.Table | None = None
        self.history = history
        self.num_nodes = num_nodes
        self.pollen_models: dict[str, Any] | None = None
        self.correction_tables: dict[str, pa.Table] | None = None
        self.accept_failures_cnt = accept_failures_cnt
        self.ignore_failed_rounds = ignore_failed_rounds
        self.print_failures = print_failures
        self.print_intentional_failures = print_intentional_failures
        self.use_s3_comm = use_s3_comm
        self.s3_comm_config = s3_comm_config
        self.checkpoint = checkpoint
        self.resume_round = resume_round
        self.restore_run_uuid_and_step = restore_run_uuid_and_steps_per_round
        self.run_uuid = run_uuid

        self.client_state: dict[str | int, ClientState] = {}
        self.server_steps_cumulative = 0

        if self.checkpoint or self.use_s3_comm:
            bucket_uri = f"s3://{self.s3_comm_config.bucket_name}"  # type: ignore[union-attr]
            self.remote_up_down = RemoteUploaderDownloader(
                # TODO: Don't hardcode
                bucket_uri=bucket_uri,
                backend_kwargs={
                    "bucket": self.s3_comm_config.bucket_name,  # type: ignore[union-attr]
                    "prefix": f"{self.run_uuid}/server",  # Don't touch
                    "region_name": None,  # Not necessary
                    "endpoint_url": None,  # Will be read from env var
                    "aws_access_key_id": None,  # Will be read from config file
                    "aws_secret_access_key": None,  # Will be read from config file
                    "aws_session_token": None,  # Will be automatically generated
                    "client_config": OmegaConf.to_container(
                        self.s3_comm_config.backend_kwargs.client_config  # type: ignore[union-attr]
                    ),  # And using defaults
                    "transfer_config": None,  # Using defaults
                },
                file_path_format_string="{remote_file_name}",  # Don't touch
                # TODO: Think about this in relation with the checkpointing
                num_concurrent_uploads=1,
                upload_staging_folder=None,  # Don't touch, it's /tmp by default
                use_procs=True,  # Don't touch
                num_attempts=self.s3_comm_config.num_attempts,  # type: ignore[union-attr]
            )
            self.remote_up_down.init(run_name=self.run_uuid)

    def set_max_workers(self, max_workers: int | None) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(  # type: ignore[override]
        self,
        strategy: FedAvg,
    ) -> None:
        """Replace server strategy."""
        self.strategy = strategy  # type: ignore[reportIncompatibleVariableOverride]

    def client_manager(self) -> PollenClientManager:
        """Return PollenClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: float | None) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        log(INFO, "Initializing Pollen simulation")

        if (
            (self.checkpoint or self.use_s3_comm)
            and self.restore_run_uuid_and_step is not None
            and self.resume_round is not None
        ):
            self.import_checkpoints()

        if self.checkpoint and self.resume_round is not None:
            history, start_round, time_offset = self.resume_from_round(timeout)
        else:
            history, start_round, time_offset = self.initialize_round(timeout)

        # NOTE: Register VirtualClients to the PollenClientManager
        self._client_manager.clients = {
            str(i): cast(ClientProxy, EmptyVirtualClient(cid=str(k)))
            for i, (k, _) in enumerate(self.cids.items())
        }

        # Waiting for at least one node to connect
        log(INFO, "Waiting for at least %s node(s) to connect", self.num_nodes)
        self._client_manager.wait_for_node_managers(self.num_nodes)

        log(
            INFO,
            "Start-up time for the server is %s",
            timeit.default_timer() - self.start_up_time,
        )
        # Run federated learning for num_rounds
        log(INFO, "FL starting from round %s", start_round + 1)
        start_time = timeit.default_timer()
        for current_round in range(start_round + 1, num_rounds + 1):
            start_round_time = time.time_ns()
            # Check for changes in connected NodeManagers
            first_check_nm_time = time.time_ns()
            self.check_node_managers(server_round=current_round)
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/first_check_nm_time": (time.time_ns() - first_check_nm_time)
                    * 1e-9
                },
            )

            # Train model and replace previous global model
            fit_round_time = time.time_ns()
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
            history.add_metrics_centralized(
                server_round=current_round,
                metrics={
                    "server/fit_round_time": (time.time_ns() - fit_round_time) * 1e-9
                },
            )

            # Push the global model to S3 Object Store (but not the server state)
            if self.checkpoint or self.use_s3_comm:
                log(DEBUG, "Dump server parameters to disk")
                dump_model_parameters_to_file(
                    Path.cwd() / "current_server_parameters.npz",
                    parameters_to_ndarrays(self.parameters),
                )
                log(DEBUG, "Push parameters to S3 Object Store")
                upload_file_to_s3(
                    self.remote_up_down,
                    f"{current_round}/current_server_parameters.npz",
                    Path.cwd() / "current_server_parameters.npz",
                )

            # Evaluate model using strategy implementation
            evaluate_time = time.time_ns()
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
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
            self.check_node_managers(server_round=current_round)
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
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
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

            # Save the checkpoint to S3 Object Store (w/o the global parameters)
            if self.checkpoint or self.use_s3_comm:
                log(DEBUG, "Create server state (server_round, history, time_offset)")
                current_server_state = {
                    "server_round": current_round,
                    "history": history,
                    "time_offset": time_offset,
                    "client_state": str(
                        {k: asdict(v) for k, v in self.client_state.items()}
                    ),
                    "server_steps_cumulative": self.server_steps_cumulative,
                }
                log(DEBUG, "Dump server state to disk")
                with open(Path.cwd() / "current_server_state.bin", "wb") as f:
                    pickle.dump(current_server_state, f)
                log(DEBUG, "Push server state to S3")
                upload_file_to_s3(
                    self.remote_up_down,
                    f"{current_round}/state.bin",
                    Path.cwd() / "current_server_state.bin",
                )
                if (
                    isinstance(self.strategy, FedNesterov)
                    and self.strategy.momentum_vector is not None
                ):
                    log(DEBUG, "Dump momentum vector to disk")
                    dump_model_parameters_to_file(
                        Path.cwd() / "current_momentum_vector.npz",
                        self.strategy.momentum_vector,
                    )
                    log(DEBUG, "Push momentum vector to S3 Object Store")
                    upload_file_to_s3(
                        self.remote_up_down,
                        f"{current_round}/current_momentum_vector.npz",
                        Path.cwd() / "current_momentum_vector.npz",
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time + time_offset
        log(INFO, "FL finished in %s", elapsed)
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: float | None,
    ) -> tuple[float | None, dict[str, Scalar], EvaluateResultsAndFailures] | None:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Translate `client_instruction` to `node_instructions`
        node_assignments: list[tuple[ClientProxy, dict[str, str]]] = self.placement_fn(
            sampled_virtual_cids=[
                (int(client.cid), self.cids[int(client.cid)])
                for client, _ in client_instructions
            ],
            nodes_dict=self.nodes_dict,
            batch_size=self.on_evaluate_config(server_round)["batch_size"],
            cids=self.cids,
            pollen_models=self.pollen_models,
            clients_stats=self.clients_training_stats,
            correction_tables=self.correction_tables,
            verbose=False,
        )
        log(
            DEBUG,
            "Node assignments for evaluate_round %s: %s",
            server_round,
            node_assignments,
        )
        node_instructions = []
        for client_proxy, device_assignment in node_assignments:
            # Skip if the device_assignment is empty
            if len(device_assignment) > 0:
                # Get the `fit_config` for the virtual clients
                node_evaluate_config = self.on_evaluate_config(server_round)
                # NOTE: This key is used only when the training policy of workers
                # is not `sequential`, and for setting the `num_workers` parameter
                # in the `DataLoader`
                if "server_round" not in node_evaluate_config:
                    node_evaluate_config["server_round"] = server_round
                if "n_workers" not in node_evaluate_config:
                    node_evaluate_config["n_workers"] = 1

                # Assign `cids` to NodeManagers' devices
                node_evaluate_config.update(device_assignment)

                # Append instruction
                if self.use_s3_comm:
                    node_instructions.append(
                        (
                            client_proxy,
                            EvaluateIns(
                                # NOTE: We must pass a real NDArrays object,
                                # Flower crashes otherwise
                                ndarrays_to_parameters([np.array([[0.0], [0.0]])]),
                                node_evaluate_config,
                            ),
                        )
                    )
                else:
                    node_instructions.append(
                        (
                            client_proxy,
                            EvaluateIns(self.parameters, node_evaluate_config),
                        )
                    )

        log(
            DEBUG,
            "Node instructions for evaluate_round %s: %s",
            server_round,
            [(c_p, ins.config) for c_p, ins in node_instructions],
        )

        log(
            DEBUG,
            "evaluate_round %s: sending instructions to %s NodeManagers",
            server_round,
            len(node_instructions),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = pollen_evaluate_clients(
            node_instructions=node_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            server_round=server_round,
        )
        if len(failures) > 0:
            log(
                ERROR,
                "evaluate_round %s: there are %s failures: %s",
                server_round,
                len(failures),
                failures,
            )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: tuple[
            float | None,
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(  # type: ignore[override]
        self,
        server_round: int,
        timeout: float | None,
    ) -> (
        None
        | tuple[
            Parameters | None,
            dict[str, Scalar],
            tuple[
                list[tuple[ClientProxy, dict[str, Scalar], Status, int]],
                list[tuple[ClientProxy, FitRes] | BaseException],
                list[BaseException],
            ],
        ]
    ):
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        # Add the desired client state to the client instructions for all clients
        converted_client_state = str(
            {k: asdict(v) for k, v in self.client_state.items()}
        )

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Translate `client_instruction` to `node_instructions`
        node_assignments: list[tuple[ClientProxy, dict[str, str]]] = self.placement_fn(
            sampled_virtual_cids=[
                (int(client.cid), self.cids[int(client.cid)])
                for client, _ in client_instructions
            ],
            nodes_dict=self.nodes_dict,
            batch_size=self.on_fit_config(server_round)["batch_size"],
            cids=self.cids,
            pollen_models=self.pollen_models,
            clients_stats=self.clients_training_stats,
            correction_tables=self.correction_tables,
            verbose=False,
        )
        log(
            DEBUG,
            "Node assignments for fit_round %s: %s",
            server_round,
            node_assignments,
        )
        node_instructions = []
        for client_proxy, device_assignment in node_assignments:
            # Get the `fit_config` for the virtual clients
            node_fit_config = self.on_fit_config(server_round)
            node_fit_config["client_state"] = converted_client_state

            # NOTE: This key is used only when the training policy of workers
            # is not `sequential`, and for setting the `num_workers` parameter
            # in the `DataLoader`
            if "server_round" not in node_fit_config:
                node_fit_config["server_round"] = server_round
            if "n_workers" not in node_fit_config:
                node_fit_config["n_workers"] = 1
            if "server_steps_cumulative" not in node_fit_config:
                node_fit_config["server_steps_cumulative"] = (
                    self.server_steps_cumulative
                )

            # Assign `cids` to NodeManagers' devices
            node_fit_config.update(device_assignment)

            # Append instruction
            if self.use_s3_comm:
                node_instructions.append(
                    (
                        client_proxy,
                        FitIns(
                            # NOTE: We must pass a real NDArrays object,
                            # Flower crashes otherwise
                            ndarrays_to_parameters([np.array([[0.0], [0.0]])]),
                            node_fit_config,
                        ),
                    )
                )
            else:
                node_instructions.append(
                    (
                        client_proxy,
                        FitIns(self.parameters, node_fit_config),
                    )
                )

        log(
            DEBUG,
            "Node instructions for fit_round %s: %s",
            server_round,
            [(c_p, ins.config) for c_p, ins in node_instructions],
        )

        log(
            DEBUG,
            "fit_round %s: sending instructions to %s NodeManagers",
            server_round,
            len(node_instructions),
        )

        # Using a generator limits us in failure/metrics accumulation
        # The output params are not used in the aggregation
        # They are merely populated by the processing of the generator
        failures: list[tuple[ClientProxy, FitRes] | BaseException] = []

        # Accumulate IntentionalClientDropoutError exceptions
        intentional_failures: list[BaseException] = []

        metrics_accumulator: list[
            tuple[ClientProxy, dict[str, Scalar], Status, int]
        ] = []
        auxiliary_fit_results = AuxiliaryFitClientsResults(
            pollen_models=None,
            correction_tables=None,
            fit_pollen_models_time=0.0,
        )
        results_futures = pollen_fit_clients(
            node_instructions=node_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            clients_stats=self.clients_training_stats,
            batch_size=int(self.on_fit_config(server_round)["batch_size"]),
            cids=self.cids,
            placement_policy=self.placement_policy,
            auxiliary_fit_results=auxiliary_fit_results,
        )
        results_and_failures = (
            _handle_finished_future_after_fit_async(future)
            for future in results_futures
        )

        handle_success_and_failure = get_handle_success_and_failure(
            metrics_accumulator,
            failures,
            intentional_failures,
            accept_failures_cnt=self.accept_failures_cnt,
        )

        results_and_failures = (
            handle_success_and_failure(result) for result in results_and_failures
        )

        results = (result for success, result in results_and_failures if success)
        complete_results = cast(
            Generator[tuple[ClientProxy, FitRes], None, None], results
        )

        # If applicable, pull the client parameters from S3 Object Store
        if self.use_s3_comm:
            self.remote_up_down._check_workers()
            complete_results = (
                replace_clients_updates_with_remote(
                    self.remote_up_down, server_round, result
                )
                for result in complete_results
            )

        try:
            # Aggregate training results
            parameters_aggregated, metrics_aggregated = self.strategy.aggregate_fit(
                server_round,
                cast(list[tuple[ClientProxy, FitRes]], complete_results),
                failures,
            )

            # Collect statistics that Pollen uses from the FitRes of the NodeManagers
            received_clients_training_stats = []
            for metrics in metrics_accumulator:
                tmp_clients_training_stats = metrics[1].pop("stats", None)
                if tmp_clients_training_stats is not None:
                    received_clients_training_stats.append(
                        get_table_from_pyarrow_buffer(
                            cast(pa.Buffer, tmp_clients_training_stats)
                        )
                    )
            # Aggregate the metrics
            # NOTE: This bypasses any metrics aggregation in the aggregate_fit of the
            # strategy because the metrics are empty there
            if self.strategy.fit_metrics_aggregation_fn:
                fit_metrics = [
                    (num_examples, metrics)
                    for _, metrics, _, num_examples in metrics_accumulator
                ]
                client_state_accumulator: dict[str | int, dict[str, Any]] = {}
                for _, inner_metrics in fit_metrics:
                    acc: dict[str | int, dict[str, Any]] = ast.literal_eval(
                        cast(str, inner_metrics["client_state_acc"])
                    )
                    client_state_accumulator |= acc
                # NOTE: When using partial participation
                # We need to accumulate the keys of the old state
                # and the new state
                self.client_state |= {
                    k: ClientState(**v) for k, v in client_state_accumulator.items()
                }
                # NOTE: Update the server steps cumulative by adding to the previous
                # value the maximum number of local steps done across the clients sample
                # in this round
                max_steps = max(
                    *[
                        _client_state.steps_done
                        for _client_state in self.client_state.values()
                    ],
                    0,
                )
                self.server_steps_cumulative += max_steps
                # NOTE: Reset the steps_done for all clients to zero as it is just meant
                # to be an ephemeral record of the local steps done during one federated
                # round
                for client_state in self.client_state.values():
                    client_state.steps_done = 0

                metrics_aggregated |= self.strategy.fit_metrics_aggregation_fn(
                    fit_metrics
                )

            elif server_round == 1:  # Only log this warning once
                log(WARNING, "No fit_metrics_aggregation_fn provided")
        except TooManyFailuresError as e:
            if self.ignore_failed_rounds:
                log(
                    ERROR,
                    """Ignoring failed round %s: %s,
                    there are %s failures: %s,
                    there are %s intentional failures: %s""",
                    server_round,
                    e,
                    len(failures),
                    failures if self.print_failures else [],
                    len(intentional_failures),
                    intentional_failures if self.print_intentional_failures else [],
                )
                return None
            else:
                raise

        # Collect the new statistics and append to the global statistics
        if self.clients_training_stats is None:
            # Create the table from scratch
            self.clients_training_stats = pa.concat_tables(
                received_clients_training_stats
            )
        else:
            # Skim the `clients_training_stats` to keep just the average per n_samples
            self.clients_training_stats = skim_clients_training_stats(
                self.clients_training_stats
            )
            # Append to the global statistics
            self.clients_training_stats = pa.concat_tables(
                [self.clients_training_stats] + received_clients_training_stats
            )

        # NOTE: We must assign this only after we can the first iter over the results
        # generator
        self.correction_tables = auxiliary_fit_results.correction_tables
        self.pollen_models = auxiliary_fit_results.pollen_models

        log(
            DEBUG,
            """fit_round %s received %s results, %s
            failures, and %s intentional failures
            Using async inplace aggregation.
            These failures are: %s,
            the intentional failures are: %s""",
            server_round,
            len(metrics_accumulator),
            len(failures),
            len(intentional_failures),
            failures if self.print_failures else [],
            intentional_failures if self.print_intentional_failures else [],
        )
        return (
            parameters_aggregated,
            metrics_aggregated,
            (metrics_accumulator, failures, intentional_failures),
        )

    def check_node_managers(
        self,
        server_round: int,
    ) -> None:
        """Quick check on the availability of the NodeManagers."""
        while self._client_manager.num_available_node_managers() < self.num_nodes:
            log(
                INFO,
                "Waiting for %s nodes to connect",
                self.num_nodes - self._client_manager.num_available_node_managers(),
            )
            time.sleep(5)
            # Check for changes in connected NodeManagers
            dropped, new = _check_connected_node_managers(
                old_connected_node_managers_cid=[k for k, _ in self.nodes_dict.items()],
                new_connected_node_managers_cid=[
                    k for k, _ in self._client_manager.node_managers.items()
                ],
            )
            if len(dropped) > 0:
                # Handle dropped NodeManagers
                [self.nodes_dict.pop(k) for k in dropped]
            if len(new) > 0:
                # Handle newly added NodeManagers
                results, failures = get_nodes_properties(
                    node_managers={
                        k: self._client_manager.node_managers[k] for k in new
                    },
                    max_workers=self.max_workers,
                    server_round=server_round,
                )
                log(
                    INFO,
                    "Get nodes properties: there are %s results and %s failures",
                    len(results),
                    len(failures),
                )
                new_nodes_dict = {
                    client_proxy.cid: (client_proxy, node)
                    for client_proxy, node in results
                }
                self.nodes_dict.update(new_nodes_dict)
        results, failures = get_nodes_properties(
            node_managers=self._client_manager.node_managers,
            max_workers=self.max_workers,
            server_round=server_round,
        )
        log(
            INFO,
            "Get nodes properties: there are %s results and %s failures",
            len(results),
            len(failures),
        )
        for failure in failures:
            if isinstance(failure, BaseException):
                log(ERROR, "Failure in getting nodes properties.", exc_info=failure)
        # This is a dictionary of the form {"node_id": Node}
        self.nodes_dict = {
            client_proxy.cid: (client_proxy, node) for client_proxy, node in results
        }
        # TODO: Clean-up stats?

    def initialize_round(self, timeout: float | None) -> tuple[History, int, float]:
        """Initialize the server for a new round.

        Parameters
        ----------
            timeout (float | None): The timeout for the communication with the clients.

        Returns
        -------
            tuple[History, int, float]: History object, the start round, time offset.
        """
        start_round: int = 0
        time_offset: float = 0.0
        history = self.history if self.history is not None else History()
        self.client_state = {cid: ClientState(0) for cid in self.cids}
        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters(timeout=timeout, server_round=0)
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        # Initialize client_state_dict
        # Save the checkpoint to S3 Object Store (w/ model parameters)
        if self.checkpoint or self.use_s3_comm:
            log(DEBUG, "Create server state (server_round, history, time_offset)")
            current_server_state = {
                "server_round": start_round,
                "history": history,
                "time_offset": time_offset,
                "client_state": str(
                    {k: asdict(v) for k, v in self.client_state.items()}
                ),
                "server_steps_cumulative": self.server_steps_cumulative,
            }
            log(DEBUG, "Dump server state to disk")
            with open(Path.cwd() / "current_server_state.bin", "wb") as f:
                pickle.dump(current_server_state, f)
            log(DEBUG, "Push server state to S3")
            upload_file_to_s3(
                self.remote_up_down,
                f"{start_round}/state.bin",
                Path.cwd() / "current_server_state.bin",
            )
            if (
                isinstance(self.strategy, FedNesterov)
                and self.strategy.momentum_vector is not None
            ):
                log(DEBUG, "Dump momentum vector to disk")
                dump_model_parameters_to_file(
                    Path.cwd() / "current_momentum_vector.npz",
                    self.strategy.momentum_vector,
                )
                log(DEBUG, "Push momentum vector to S3 Object Store")
                upload_file_to_s3(
                    self.remote_up_down,
                    f"{start_round}/current_momentum_vector.npz",
                    Path.cwd() / "current_momentum_vector.npz",
                )
            log(DEBUG, "Dump server parameters to disk")
            dump_model_parameters_to_file(
                Path.cwd() / "current_server_parameters.npz",
                parameters_to_ndarrays(self.parameters),
            )
            log(DEBUG, "Push parameters to S3 Object Store")
            upload_file_to_s3(
                self.remote_up_down,
                f"{start_round}/current_server_parameters.npz",
                Path.cwd() / "current_server_parameters.npz",
            )
        return history, start_round, time_offset

    def resume_from_round(self, timeout: float | None) -> tuple[History, int, float]:
        """Resume the server from a given round.

        Returns
        -------
            tuple[History, float]: The history object and the time offset.
        """
        time_offset: float = 0.0
        start_round: int = 0
        assert self.resume_round is not None
        try:
            server_path = (
                f"s3://{self.s3_comm_config.bucket_name}/"  # type: ignore[union-attr,reportOptionalMemberAccess]
                f"{self.run_uuid}/server/"
            )
            if self.resume_round < 0:
                log(INFO, "Negative round number %s", self.resume_round)
                server_round_indices = obtain_sorted_runs(server_path)

                log(INFO, "Found server round indices %s", server_round_indices)
                if (
                    not server_round_indices
                    or (self.resume_round + server_round_indices[-1] + 1) < 0
                ):
                    log(
                        INFO,
                        "No checkpoints found in %s. Starting training from scratch.",
                        server_path,
                    )
                    return self.initialize_round(timeout)

                self.resume_round += server_round_indices[-1] + 1
                log(INFO, "Resuming from round %s", self.resume_round)

            log(INFO, "Resuming from checkpoint")
            # Check whether the server parameters exist
            file_found = False
            remote_file_name_no_ext = (
                server_path + f"{self.resume_round}/current_server_parameters"
            )
            while not file_found:
                file_found = validate_given_remote_path(
                    remote_file_name_no_ext + ".bin"
                ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
                time.sleep(0.5)
            # Set the file names depending on the extension found
            remote_file_name = (
                f"{self.resume_round}/current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else f"{self.resume_round}/current_server_parameters.npz"
            )
            local_file_name = (
                Path.cwd() / "current_server_parameters.bin"
                if validate_given_remote_path(remote_file_name_no_ext + ".bin")
                else Path.cwd() / "current_server_parameters.npz"
            )
            log(DEBUG, "Pull server parameters from S3 Object Store")
            # Download the parameters
            download_file_from_s3(
                self.remote_up_down, remote_file_name, local_file_name
            )
            log(DEBUG, "Read server parameters from disk")
            checkpoint_parameters = load_model_parameters_from_file(local_file_name)
            self.parameters = ndarrays_to_parameters(checkpoint_parameters)
            if isinstance(self.strategy, FedNesterov):
                self.strategy.ndarray_parameters = checkpoint_parameters
            log(DEBUG, "Pull server state from S3 Object Store")
            # Download the server state from S3 Object Store
            download_file_from_s3(
                self.remote_up_down,
                f"{self.resume_round}/state.bin",
                str(Path.cwd() / "current_server_state.bin"),
            )
            log(DEBUG, "Read server state from disk")
            with open(Path.cwd() / "current_server_state.bin", "rb") as f:
                server_state = pickle.load(f)
            start_round = server_state["server_round"]
            assert (
                start_round == self.resume_round
            ), "Server round mismatch with checkpoint"
            history: History = server_state["history"]
            if "client_state" in server_state:
                saved_client_state: dict[str | int, dict[str, Any]] = ast.literal_eval(
                    server_state["client_state"]
                )
            else:
                saved_client_state = {
                    cid: {"local_steps_cumulative": int(500 * self.resume_round)}
                    for cid in self.cids
                }
            self.client_state = {
                k: ClientState(**v) for k, v in saved_client_state.items()
            }

            if "time_offset" in server_state:
                time_offset = server_state["time_offset"]
            if "server_steps_cumulative" in server_state:
                self.server_steps_cumulative = server_state["server_steps_cumulative"]
            else:
                # Make it back compatible with the previous versions
                self.server_steps_cumulative = max(
                    *[
                        _client_state.local_steps_cumulative
                        for _client_state in self.client_state.values()
                    ],
                    0,
                )
            if isinstance(self.strategy, FedNesterov):
                if "momentum" in server_state:
                    log(DEBUG, "Get momentum vector from server state")
                    self.strategy.momentum_vector = server_state["momentum"]
                else:
                    log(DEBUG, "Pull momentum from S3 Object Store")
                    # Set the file names depending on the extension found
                    remote_file_name = (
                        f"{self.resume_round}/current_momentum_vector.npz"
                    )
                    local_file_name = Path.cwd() / "current_momentum_vector.npz"
                    # Download the parameters
                    download_file_from_s3(
                        self.remote_up_down, remote_file_name, local_file_name
                    )
                    self.strategy.momentum_vector = load_model_parameters_from_file(
                        local_file_name
                    )
            log(DEBUG, "Server state has been read from disk")
        except Exception as e:
            log(ERROR, "Failed to resume from checkpoint: %s", e)
            sys.exit(1)
        return history, start_round, time_offset

    def import_checkpoints(self) -> None:
        """Import checkpoints from a previous run."""
        # Import previous checkpoints if asked to

        assert (
            self.restore_run_uuid_and_step is not None and self.resume_round is not None
        )

        restore_run_uuid, steps_per_round = self.restore_run_uuid_and_step
        if self.resume_round < 0:
            server_path = (
                f"s3://{self.s3_comm_config.bucket_name}/"  # type: ignore[union-attr,reportOptionalMemberAccess]
                f"{restore_run_uuid}/server/"
            )

            log(INFO, "Negative round number %s", self.resume_round)
            server_round_indices = obtain_sorted_runs(server_path)

            log(INFO, "Found server round indices %s", server_round_indices)
            if (
                not server_round_indices
                or (self.resume_round + server_round_indices[-1] + 1) < 0
            ):
                log(
                    INFO,
                    "No checkpoints found in %s. Starting training from scratch.",
                    server_path,
                )
                self.resume_round = None
                return

            self.resume_round += server_round_indices[-1] + 1
            log(INFO, "Resuming from round %s", self.resume_round)

        copy_old_checkpoints_to_new_run(
            remote_up_down=self.remote_up_down,
            bucket_uri=f"s3://{self.s3_comm_config.bucket_name}",  # type: ignore[union-attr]
            run_uuid=self.run_uuid,
            restore_run_uuid=restore_run_uuid,
            restore_run_round=self.resume_round,
            restore_run_step=self.resume_round * steps_per_round,
            n_total_clients=len(self.cids),
        )


# NEW FUNCTIONS #######################


def pollen_evaluate_clients(
    node_instructions: list[tuple[ClientProxy, EvaluateIns]],
    max_workers: int | None,
    timeout: float | None,
    server_round: int,
) -> EvaluateResultsAndFailures:
    """Evaluate parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(evaluate_client, client_proxy, ins, timeout, server_round)
            for client_proxy, ins in node_instructions
        }
        # TODO: Implement Pollen's model for the eval assignment
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return results, failures


def pollen_fit_clients(
    node_instructions: list[tuple[ClientProxy, FitIns]],
    max_workers: int | None,
    timeout: float | None,
    auxiliary_fit_results: AuxiliaryFitClientsResults,
    cids: dict[str | int, int],
    clients_stats: pa.Table | None = None,
    batch_size: int = 1,
    placement_policy: str = "rr",
) -> Generator[concurrent.futures.Future[tuple[ClientProxy, FitRes]], Any, None]:
    """Refine parameters concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout, 0)
            for client_proxy, ins in node_instructions
        }
        # Fit the Pollen models and collect their parameters and correction tables in
        # the `auxiliary_fit_results` variables
        start_time = timeit.default_timer()
        auxiliary_fit_results.pollen_models, auxiliary_fit_results.correction_tables = (
            get_pollen_models(
                placement_policy=placement_policy,
                clients_stats=clients_stats,
                batch_size=batch_size,
                cids=cids,
                server_round=int(node_instructions[0][1].config["server_round"]),
            )
        )
        auxiliary_fit_results.fit_pollen_models_time = (
            timeit.default_timer() - start_time
        )
        # Constructing the generator of training results
        while submitted_fs:
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in finished_fs:
                submitted_fs.remove(future)
                yield future


def get_nodes_properties(
    node_managers: dict[str, ClientProxy],
    max_workers: int | None,
    server_round: int,
    timeout: float | None = None,
) -> GetPropResultsAndFailures:
    """Get the properties of all nodes in the cluster."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(get_properties_client, client_proxy, timeout, server_round)
            for node_id, client_proxy in node_managers.items()
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=timeout,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, Node]] = []
    failures: list[tuple[ClientProxy, Node] | BaseException] = []
    for future in finished_fs:
        _handle_finished_future_after_get_properties(
            future=future, results=results, failures=failures
        )
    return results, failures


def get_properties_client(
    client: ClientProxy, timeout: float | None, server_round: int
) -> tuple[ClientProxy, Node]:
    """Get properties from a Node."""
    ins = GetPropertiesIns(config={})
    node_properties_res = client.get_properties(
        ins=ins, timeout=timeout, group_id=server_round
    )
    node_properties: Properties = node_properties_res.properties
    # log(
    #     DEBUG,
    #     "node properties received from %s: %s",
    #     client,
    #     node_properties,
    # )
    return client, Node.from_str(str(node_properties["node"]))


def _handle_finished_future_after_fit_async(
    future: concurrent.futures.Future,
) -> (
    tuple[Literal[True], tuple[ClientProxy, FitRes]]
    | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException]
):
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        return (False, failure)

    # Successfully received a result from a client
    result: tuple[ClientProxy, FitRes] = future.result()
    _, res = result
    # Check result status code
    if res.status.code == Code.OK:
        return (True, result)

    # Not successful, client returned a result where the status code is not OK
    return (False, result)


def replace_clients_updates_with_remote(
    remote_uploader_downloader: RemoteUploaderDownloader,
    current_round: int,
    client_result: tuple[ClientProxy, FitRes],
) -> tuple[ClientProxy, FitRes]:
    """Replace the parameters in the FitRes with the ones from S3 Object Store."""
    proxy, fit_res = client_result
    endpoint_id: Any
    if "endpoint_id" in fit_res.metrics:
        endpoint_id = fit_res.metrics["endpoint_id"]
        del fit_res.metrics["endpoint_id"]
        if not isinstance(endpoint_id, str):
            raise TypeError("endpoint_id is not a string")
    else:
        raise ValueError("endpoint_id is not present in fit_res")
    # Check whether the server has uploaded the parameters
    file_found = False
    remote_file_name_no_ext = (
        f"s3://{remote_uploader_downloader.remote_bucket_name}/"  # type: ignore[union-attr]
        f"{remote_uploader_downloader.backend_kwargs['prefix']}/"
        f"{current_round}/{endpoint_id}/parameters"
    )

    log(
        DEBUG,
        "Wait for NodeManager %s parameters to be in S3 Object Store at %s",
        endpoint_id,
        remote_file_name_no_ext,
    )

    while not file_found:
        file_found = validate_given_remote_path(
            remote_file_name_no_ext + ".bin"
        ) or validate_given_remote_path(remote_file_name_no_ext + ".npz")
        time.sleep(0.5)

    # Set the file names depending on the extension found
    remote_file_name = (
        f"{current_round}/{endpoint_id}/parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else f"{current_round}/{endpoint_id}/parameters.npz"
    )
    local_file_name = (
        Path.cwd() / f"{endpoint_id}_current_server_parameters.bin"
        if validate_given_remote_path(remote_file_name_no_ext + ".bin")
        else Path.cwd() / f"{endpoint_id}_current_server_parameters.npz"
    )
    log(
        DEBUG,
        "Pull Node %s parameters from S3 Object Store: %s -> %s",
        endpoint_id,
        remote_file_name,
        local_file_name,
    )
    download_file_from_s3(remote_uploader_downloader, remote_file_name, local_file_name)
    log(DEBUG, "Read server parameters from disk")
    fit_res.parameters = ndarrays_to_parameters(
        load_model_parameters_from_file(local_file_name)
    )

    log(
        DEBUG,
        "Node %s parameters have been read from disk and assigned to fit_res",
        endpoint_id,
    )

    return (proxy, fit_res)


def get_handle_success_and_failure(
    metrics_accumulator: list[tuple[ClientProxy, dict[str, Scalar], Status, int]],
    failures: list[tuple[ClientProxy, FitRes] | BaseException],
    intentional_failures: list[BaseException],
    accept_failures_cnt: int | None,
) -> Callable[
    [
        tuple[Literal[True], tuple[ClientProxy, FitRes]]
        | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException]
    ],
    tuple[Literal[True], tuple[ClientProxy, FitRes]]
    | tuple[Literal[False], tuple[ClientProxy, FitRes] | BaseException],
]:
    """Closure to generate a function which handles client success and failure.

    The function distinguishes between intentional and unintentional failures.
    It enforces constraints on the number of unintentional failures.
    It stores results and failures in the respective lists.

    Parameters
    ----------
    metrics_accumulator : List[Tuple[ClientProxy, Dict[str, Scalar], Status, int]]
        The list where the metrics are accumulated.
    failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
        The list where the failures are accumulated.
    intentional_failures : List[BaseException]
        The list where the intentional failures are accumulated.
    accept_failures_cnt : int | None
        The maximum number of unintentional failures to accept.

    Returns
    -------
    handle_success_and_failure : Callable[
        [
            Tuple[Literal[True], Tuple[ClientProxy, FitRes]]
            | Tuple[Literal[False], Tuple[ClientProxy, FitRes] | BaseException]
        ],
        Tuple[Literal[True], Tuple[ClientProxy, FitRes]]
        | Tuple[Literal[False], Tuple[ClientProxy, FitRes] | BaseException],
    ]
        The function which handles client success and failure while saving the outputs.
    """

    def handle_success_and_failure(
        result: (
            tuple[Literal[True], tuple[ClientProxy, FitRes]]
            | tuple[Literal[False], (tuple[ClientProxy, FitRes] | BaseException)]
        ),
    ) -> (
        tuple[Literal[True], tuple[ClientProxy, FitRes]]
        | tuple[Literal[False], (tuple[ClientProxy, FitRes] | BaseException)]
    ):
        cnt_failures = 0

        match result:
            case (True, res):
                cast_res = cast(tuple[ClientProxy, FitRes], res)
                client_proxy, fit_res = cast_res
                metrics_accumulator.append(
                    (
                        client_proxy,
                        fit_res.metrics,
                        fit_res.status,
                        fit_res.num_examples,
                    )
                )
                return (True, cast_res)
            case (False, res) if isinstance(res, IntentionalClientDropoutError):
                intentional_failures.append(res)
                return (False, res)
            case (False, res):
                cast_failure_res = cast(tuple[ClientProxy, FitRes] | BaseException, res)
                if isinstance(cast_failure_res, BaseException):
                    log(ERROR, "Unintentional failure.", exc_info=cast_failure_res)
                cnt_failures += 1
                if (
                    accept_failures_cnt is not None
                    and cnt_failures > accept_failures_cnt
                ):
                    raise TooManyFailuresError(
                        f"""Unintentional failures passed
                        the maximum: {accept_failures_cnt}"""
                    )
                failures.append(cast_failure_res)
                return (False, cast_failure_res)
        return result

    return handle_success_and_failure


def _handle_finished_future_after_get_properties(
    future: concurrent.futures.Future,
    results: list[tuple[ClientProxy, Node]],
    failures: list[tuple[ClientProxy, Node] | BaseException],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, Node] = future.result()
    results.append(result)


def _check_strategy_for_pollen(
    strategy: FedAvg,
) -> bool:
    if strategy.on_fit_config_fn is None:
        log(
            ERROR,
            "The strategy, %s, passed to the `PollenServer` doesn't"
            " have a proper `on_fit_config_fn` attribute. The user"
            " must define such method as type `Callable[[int], Dict]`"
            "Currently, `on_fit_config_fn` is %s.",
            strategy,
            strategy.on_fit_config_fn,
        )
        sys.exit(0)
    if "batch_size" not in strategy.on_fit_config_fn(0) or not isinstance(
        strategy.on_fit_config_fn(0)["batch_size"], int
    ):
        log(
            ERROR,
            "The `on_fit_config_fn` function of the strategy passed"
            " to the `PollenServer` must have a proper `batch_size`"
            " key with an `int` value. The call"
            " `strategy.on_fit_config_fn(0)` returned %s instead",
            strategy.on_fit_config_fn(0),
        )
        sys.exit(0)
    if strategy.on_evaluate_config_fn is None:
        log(
            ERROR,
            "The strategy, %s, passed to the `PollenServer` doesn't"
            " have a proper `on_evaluate_config_fn` attribute. The user"
            " must define such method as type `Callable[[int], Dict]`"
            "Currently, `on_evaluate_config_fn` is %s.",
            strategy,
            strategy.on_evaluate_config_fn,
        )
        sys.exit(0)
    if "batch_size" not in strategy.on_evaluate_config_fn(0) or not isinstance(
        strategy.on_evaluate_config_fn(0)["batch_size"], int
    ):
        log(
            ERROR,
            "The `on_evaluate_config_fn` function of the strategy passed"
            " to the `PollenServer` must have a proper `batch_size`"
            " key with an `int` value. The call"
            " `strategy.on_evaluate_config_fn(0)` returned %s instead",
            strategy.on_evaluate_config_fn(0),
        )
        sys.exit(0)
    return True


def _check_connected_node_managers(
    old_connected_node_managers_cid: list[str],
    new_connected_node_managers_cid: list[str],
) -> tuple[list[str], list[str]]:
    dropped: list[str] = []
    new: list[str] = []
    for old_cid in old_connected_node_managers_cid:
        if old_cid not in set(new_connected_node_managers_cid):
            dropped.append(old_cid)
    for new_cid in new_connected_node_managers_cid:
        if new_cid not in set(old_connected_node_managers_cid):
            new.append(new_cid)
    return dropped, new
