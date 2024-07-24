"""Utility functions for fit tasks on main server loop in flwr next."""

import ast
from collections.abc import Callable, Generator
from logging import ERROR, WARNING
from typing import Any, cast

from flower_llm.server.s3_utils import (
    replace_parameters_in_recordset_with_remote,
)
from flower_llm.server.server_util import TooManyFailuresError
from flower_llm.utils import (
    ClientState,
)
from flwr.common import (
    Parameters,
    log,
    Message,
    Scalar,
    FitRes,
    Status,
    Code,
)

from flwr.common.recordset_compat import (
    recordset_to_fitres,
)
from flwr.server.strategy import FedAvg
from composer.loggers import RemoteUploaderDownloader


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
    """Handle fit replies from clients.

    Parameters
    ----------
    cfg : BaseConfig
        The configuration object.
    replies : Generator[Message, None, None]
        The generator of messages from clients.
    strategy : FedAvg
        The strategy object.
    current_round : int
        The current round number.
    remote_up_down : RemoteUploaderDownloader | None
        The object to handle remote communication.
    client_state : dict[str | int, ClientState]
        The dictionary of client states.
    server_steps_cumulative : int
        The cumulative number of server steps.

    Returns
    -------
    None | Tuple[Parameters | None, Dict[str, Scalar],
        Tuple[List[Tuple[Dict[str, Scalar], Status, int]], List[FitRes | None]]]
        The aggregated parameters, the aggregated metrics, and the metrics and failures.
    """
    # Translate message with fake parameters with parameters downloaded from the S3
    processed_msgs = (
        (
            replace_parameters_in_recordset_with_remote(
                remote_uploader_downloader=remote_up_down,
                incoming_message=msg,
                use_s3_comm=cfg.use_s3_comm,
                msg_str="fitres",
            )
            if msg.has_content()
            else msg
        )
        for msg in replies
    )

    # Translate Messages to FitRes
    status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Unexpected empty content")
    error_fitres = FitRes(
        status=status,
        # parameters=ndarrays_to_parameters([np.array([[0.0], [0.0]])]),
        parameters=Parameters(tensors=[], tensor_type="empty"),
        metrics={},
        num_examples=1,
    )
    all_fitres = (
        (
            recordset_to_fitres(msg.content, keep_input=True)
            if msg.has_content()
            else error_fitres
        )
        for msg in processed_msgs
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

    parameters_aggregated = None
    metrics_aggregated: dict = {}

    try:
        # Aggregate training results
        parameters_aggregated, _ = strategy.aggregate_fit(
            current_round,
            ((None, fit_res) for fit_res in results),  # type: ignore[reportArgumentType, arg-type]
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
                metrics_accumulator.append(
                    (
                        fit_res.metrics,
                        fit_res.status,
                        fit_res.num_examples,
                    )
                )
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
