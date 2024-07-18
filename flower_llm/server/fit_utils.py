"""Utility functions for fit tasks on main server loop in flwr next."""

import ast
from collections.abc import Callable, Generator
from copy import deepcopy
import copy
from dataclasses import asdict
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from typing import Any, cast
import time

from flower_llm.clients.llm_client_functions import (
    copy_old_checkpoints_to_new_run,
    get_raw_model_parameters,
)
from flower_llm.pollen_server import TooManyFailuresError
from flower_llm.server.s3_utils import replace_clients_updates_with_remote
from flower_llm.utils import (
    ClientState,
    download_file_from_s3,
    dump_model_parameters_to_file,
    load_model_parameters_from_file,
    obtain_sorted_runs,
    upload_file_to_s3,
)
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
    Parameters,
    log,
    Message,
    MessageType,
    DEFAULT_TTL,
    ConfigsRecord,
    RecordSet,
    Scalar,
    FitIns,
    FitRes,
    EvaluateRes,
    Status,
    Code,
)

from flwr.common.recordset_compat import parameters_to_parametersrecord
from flwr.common.recordset_compat import (
    fitins_to_recordset,
    recordset_to_fitres,
    recordset_to_evaluateres,
)
from flwr.server import Driver, History
from flwr.server.strategy import FedAvg
from omegaconf import OmegaConf
from composer.loggers import RemoteUploaderDownloader
from composer.utils.file_helpers import validate_given_remote_path


from flower_llm.conf.base_schema import BaseConfig


def handle_fit_replies(
    cfg: BaseConfig,
    replies: Generator[Message, None, None],
    strategy: FedAvg,
    current_round: int,
    remote_up_down: RemoteUploaderDownloader | None,
    client_state: dict[str | int, ClientState],
    server_steps_cumulative: int,
) -> (
    None
    | tuple[
        Parameters | None,
        dict[str, Scalar],
        tuple[list[tuple[dict[str, Scalar], Status, int]], list[FitRes | None]],
    ]
):
    """Perform a single round of federated averaging."""
    all_fitres = (
        recordset_to_fitres(msg.content, keep_input=True) if msg.has_content() else None
        for msg in replies
    )

    results_and_failures = (
        (
            (fit_res.status.code == Code.OK, fit_res)
            if fit_res is not None
            else (False, fit_res)
        )
        for fit_res in all_fitres
    )

    # Using a generator limits us in failure/metrics accumulation
    # The output params are not used in the aggregation
    # They are merely populated by the processing of the generator
    failures: list[FitRes | None] = []

    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]] = []

    handle_success_and_failure = get_handle_success_and_failure_fit(
        metrics_accumulator,
        failures,
        accept_failures_cnt=cfg.fl.accept_failures_cnt,
    )
    handled_results_and_failures = (
        handle_success_and_failure(result)  # type: ignore[arg-type]
        for result in results_and_failures
    )

    results = (result for success, result in handled_results_and_failures if success)

    complete_results = results

    if cfg.use_s3_comm and remote_up_down is not None:
        remote_up_down._check_workers()
        complete_results = (
            replace_clients_updates_with_remote(
                remote_up_down,
                current_round,
                result,
            )
            for result in results
            if result is not None
        )

    parameters_aggregated = None
    metrics_aggregated: dict = {}

    try:
        # Aggregate training results
        parameters_aggregated, _ = strategy.aggregate_fit(
            current_round,
            ((None, fit_res) for fit_res in complete_results),  # type: ignore[reportArgumentType, arg-type]
            failures,  # type: ignore[reportArgumentType, arg-type]
        )

        # Collect statistics that Pollen uses from the FitRes of the NodeManagers

        fit_metrics = [
            (num_examples, metrics) for metrics, _, num_examples in metrics_accumulator
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
        client_state |= {
            k: ClientState(**v) for k, v in client_state_accumulator.items()
        }
        # NOTE: Update the server steps cumulative by adding to the previous
        # value the maximum number of local steps done across the clients sample
        # in this round
        max_steps = max(
            *[_client_state.steps_done for _client_state in client_state.values()],
            0,
        )
        server_steps_cumulative += max_steps
        # NOTE: Reset the steps_done for all clients to zero as it is just meant
        # to be an ephemeral record of the local steps done during one federated
        # round
        for c_state in client_state.values():
            c_state.steps_done = 0

        # Aggregate the metrics
        # NOTE: This bypasses any metrics aggregation in the aggregate_fit of the
        # strategy because the metrics are empty there
        if strategy.fit_metrics_aggregation_fn:
            metrics_aggregated |= strategy.fit_metrics_aggregation_fn(fit_metrics)

        elif current_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
    except TooManyFailuresError as e:
        if cfg.fl.ignore_failed_rounds:
            log(
                ERROR,
                """Ignoring failed round %s: %s,
                there are %s failures: %s""",
                current_round,
                e,
                len(failures),
                failures,
            )
        else:
            raise
    return parameters_aggregated, metrics_aggregated, (metrics_accumulator, failures)


def get_handle_success_and_failure_fit(
    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]],
    fit_failures: list[FitRes | None],
    accept_failures_cnt: int | None,
) -> Callable[
    [tuple[bool, FitRes] | tuple[bool, None]],
    tuple[bool, FitRes] | tuple[bool, None],
]:
    """Closure to generate a function which handles client success and failure.

    The function distinguishes between intentional and unintentional failures.
    It enforces constraints on the number of unintentional failures.
    It stores results and failures in the respective lists.

    Parameters
    ----------
    metrics_accumulator : List[Tuple[ClientProxy, Dict[str, Scalar], Status, int]]
        The list where the metrics are accumulated.
    failures : List[Union[FitRes, BaseException]]
        The list where the failures are accumulated.
    accept_failures_cnt : int | None
        The maximum number of unintentional failures to accept.

    Returns
    -------
    handle_success_and_failure : Callable[
        [
            Tuple[bool, FitRes]
            | Tuple[bool, FitRes | BaseException]
        ],
        Tuple[bool, FitRes]
        | Tuple[bool, FitRes | BaseException],
    ]
        The function which handles client success and failure while saving the outputs.
    """

    def handle_success_and_failure_fit(
        result: tuple[bool, FitRes] | tuple[bool, None],
    ) -> tuple[bool, FitRes] | tuple[bool, None]:
        cnt_failures = 0

        match result:
            case (True, res):
                fit_res = cast(FitRes, res)
                metrics_accumulator.append((
                    fit_res.metrics,
                    fit_res.status,
                    fit_res.num_examples,
                ))
                return (True, fit_res)
            case (False, res):
                cnt_failures += 1
                if (
                    accept_failures_cnt is not None
                    and cnt_failures > accept_failures_cnt
                ):
                    raise TooManyFailuresError(
                        f"""Unintentional failures passed
                        the maximum: {accept_failures_cnt}"""
                    )
                fit_failures.append(res)  # type: ignore[arg-type]
                return (False, res) if res is not None else (False, None)  # type: ignore[return-value]
        return result

    return handle_success_and_failure_fit
