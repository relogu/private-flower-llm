"""Utility functions for evaluate tasks on main server loop in flwr next."""

from collections.abc import Callable, Generator
from logging import DEBUG, ERROR
from typing import cast

from flower_llm.pollen_server import TooManyFailuresError
from flwr.common import (
    log,
    Message,
    Scalar,
    EvaluateRes,
    Status,
    Code,
)

from flwr.common.recordset_compat import (
    recordset_to_evaluateres,
)
from flwr.server.strategy import FedAvg


from flower_llm.conf.base_schema import BaseConfig


def handle_evaluate_replies(
    cfg: BaseConfig,
    replies: Generator[Message, None, None],
    strategy: FedAvg,
    current_round: int,
) -> (
    None
    | tuple[
        float | None,
        dict[str, Scalar],
        tuple[list[tuple[None, EvaluateRes | None]], list[EvaluateRes | None]],
    ]
):
    all_eval_res = (
        recordset_to_evaluateres(msg.content) if msg.has_content() else None
        for msg in replies
    )

    results_and_failures = (
        (
            (eval_res.status.code == Code.OK, eval_res)
            if eval_res is not None
            else (False, eval_res)
        )
        for eval_res in all_eval_res
    )

    # Using a generator limits us in failure/metrics accumulation
    # The output params are not used in the aggregation
    # They are merely populated by the processing of the generator
    failures: list[EvaluateRes | None] = []

    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]] = []

    handle_success_and_failure = get_handle_success_and_failure_evaluate(
        metrics_accumulator,
        failures,
        accept_failures_cnt=cfg.fl.accept_failures_cnt,
    )
    results_and_failures = (
        handle_success_and_failure(result)  # type: ignore[arg-type]
        for result in results_and_failures
    )

    results = (result for success, result in results_and_failures if success)
    completed_results = [(None, fit_res) for fit_res in results]

    aggregated_result: tuple[
        float | None,
        dict[str, Scalar],
    ] = strategy.aggregate_evaluate(
        current_round,
        completed_results,  # type: ignore[reportArgumentType,arg-type]
        failures,  # type: ignore[reportArgumentType,arg-type]
    )

    if len(failures) > 0:
        log(
            ERROR,
            "evaluate_round %s: there are %s failures: %s",
            current_round,
            len(failures),
            failures,
        )
    log(
        DEBUG,
        "evaluate_round %s received %s results and %s failures",
        current_round,
        len(completed_results),
        len(failures),
    )

    loss_aggregated, metrics_aggregated = aggregated_result
    return loss_aggregated, metrics_aggregated, (completed_results, failures)


def get_handle_success_and_failure_evaluate(
    metrics_accumulator: list[tuple[dict[str, Scalar], Status, int]],
    evaluate_failures: list[EvaluateRes | None],
    accept_failures_cnt: int | None,
) -> Callable[
    [tuple[bool, EvaluateRes] | tuple[bool, None]],
    tuple[bool, EvaluateRes] | tuple[bool, None],
]:
    """Closure to generate a function which handles client success and failure.

    The function distinguishes between intentional and unintentional failures.
    It enforces constraints on the number of unintentional failures.
    It stores results and failures in the respective lists.

    Parameters
    ----------
    metrics_accumulator : List[Tuple[ClientProxy, Dict[str, Scalar], Status, int]]
        The list where the metrics are accumulated.
    failures : List[Union[EvaluateRes, BaseException]]
        The list where the failures are accumulated.
    accept_failures_cnt : int | None
        The maximum number of unintentional failures to accept.

    Returns
    -------
    handle_success_and_failure : Callable[
        [
            Tuple[bool, EvaluateRes]
            | Tuple[bool, EvaluateRes | BaseException]
        ],
        Tuple[bool, EvaluateRes]
        | Tuple[bool, EvaluateRes | BaseException],
    ]
        The function which handles client success and failure while saving the outputs.
    """

    def handle_success_and_failure_evaluate(
        result: tuple[bool, EvaluateRes] | tuple[bool, None],
    ) -> tuple[bool, EvaluateRes] | tuple[bool, None]:
        cnt_failures = 0

        match result:
            case (True, res):
                fit_res = cast(EvaluateRes, res)
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
                evaluate_failures.append(res)  # type: ignore[arg-type]
                return (False, res) if res is not None else (False, None)  # type: ignore[return-value]
        return result

    return handle_success_and_failure_evaluate
