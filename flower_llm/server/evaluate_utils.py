"""Utility functions for evaluate tasks on main server loop in flwr next."""

from collections.abc import Callable, Generator
from logging import DEBUG, ERROR
import time
from typing import cast

from flwr.common import (
    log,
    Message,
    Scalar,
    EvaluateRes,
    Status,
    Code,
    MessageType,
)
from flwr.common.typing import ConfigsRecordValues

from flwr.common.recordset_compat import (
    recordset_to_evaluateres,
)
from flwr.server import Driver, History
from flwr.server.strategy import FedAvg


from flower_llm.conf.base_schema import BaseConfig
from flower_llm.server.server_util import (
    TooManyFailuresError,
    message_collaborative,
    message_independent,
)
from flower_llm.utils import ClientState


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
    """Process evaluation replies from nodes, aggregates the results, and logs failures.

    This function iterates over a generator of replies from clients, extracts evaluation
    results, and handles successes and failures. It aggregates successful evaluation
    results using a specified federated averaging strategy and logs any failures
    encountered during the process. The function returns the aggregated loss, aggregated
    metrics, and a tuple containing lists of completed results and failures.

    Parameters
    ----------
    cfg : BaseConfig
        Configuration settings for the federated learning process, including failure
        tolerance.
    replies : Generator[Message, None, None]
        A generator of messages from clients, each potentially containing an evaluation
        result.
    strategy : FedAvg
        The federated averaging strategy to use for aggregating evaluation results.
    current_round : int
        The current round of the federated learning process.

    Returns
    -------
    (
        None |
        tuple[
            float | None,
            dict[str, Scalar],
            tuple[list[tuple[None, EvaluateRes | None]], list[EvaluateRes | None]]
        ]
    )
        A tuple containing the aggregated loss (or None if not applicable), a dictionary
        of aggregated metrics, and a tuple of two lists: one for completed results (with
        placeholders for successes) and one for failures. Returns None if the operation
        cannot be completed.

    Raises
    ------
    - This function does not explicitly raise exceptions but depends on the strategy's
        `aggregate_evaluate` method and the handling of messages for potential
        exceptions.

    Notes
    -----
    - The function first transforms the replies into evaluation results, marking each as
        a success or failure.
    - It then separates successes from failures, aggregates the successful results using
        the provided strategy, and logs any failures.
    - The aggregation of results is based on the current round's data and the strategy's
        aggregation method, which typically involves computing the mean loss and metrics
        across all successful evaluations.
    - Failures are logged with an error level, including the round number and details of
        the failures.
    - The function is designed to work within a federated learning framework, assuming
        the existence of `BaseConfig`, `Message`, `FedAvg`, `EvaluateRes`, `Scalar`, and
        logging utilities.
    """
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
                evaluate_failures.append(res)  # type: ignore[arg-type]
                return (False, res) if res is not None else (False, None)  # type: ignore[return-value]
        return result

    return handle_success_and_failure_evaluate


def evaluate_round(
    driver: Driver,
    sampled_clients: list[int] | list[str],
    evaluate_config_fn: Callable[[int, int | str], dict[str, ConfigsRecordValues]],
    all_node_ids: list[int],
    current_round: int,
    client_state: dict[str | int, ClientState],
    server_steps_cumulative: int,
    cfg: BaseConfig,
    strategy: FedAvg,
    history: History,
) -> History:
    """Execute a round of federated evaluation.

    Parameters
    ----------
        driver (Driver): The driver responsible for communication with clients.
        sampled_clients (list[int] | list[str]): List of client IDs selected for this
            round.
        evaluate_config_fn (Callable[[int, int | str], dict[str, ConfigsRecordValues]]):
            Function to generate evaluation configuration for each client.
        all_node_ids (list[int]): List of all node IDs available for assignment.
        current_round (int): The current round number of federated evaluation.
        client_state (dict[str | int, ClientState]): Dictionary maintaining the state of
            each client.
        server_steps_cumulative (int): Cumulative number of server steps.
        cfg (BaseConfig): Configuration object containing various settings.
        strategy (FedAvg): Federated averaging strategy for aggregating client updates.
        history (History): Object to record the history of metrics and events.

    Returns
    -------
        History: Updated history object with evaluation metrics and events.
    """
    # Evaluate model on a sample of available clients
    evaluate_round_time = time.time_ns()

    # NOTE: We are re-using the nodes attached for doing centralized evaluation
    # at the moment
    def _rr_assignment(x: int) -> list[int]:
        """Implement a dummy round-robin assignment function for evaluation.

        Parameters
        ----------
            x (int): Node ID.

        Returns
        -------
            list[int]: List containing a single node ID (0).
        """
        return [0]

    eval_replies = (
        # Send on client-message at a time
        message_collaborative(
            driver=driver,
            message_type=MessageType.EVALUATE,
            sampled_clients=sampled_clients,
            gen_ins_function=evaluate_config_fn,
            all_node_ids=all_node_ids,
            current_round=current_round,
            msg_str="evaluateins",
            client_state=client_state,
            server_steps_cumulative=server_steps_cumulative,
        )
        if cfg.pollen.eval_collaborative
        # Send one-shot assignment of multiple clients
        else message_independent(
            driver=driver,
            message_type=MessageType.EVALUATE,
            gen_ins_function=evaluate_config_fn,
            all_node_ids=all_node_ids,
            current_round=current_round,
            assignment_function=_rr_assignment,
            msg_str="evaluateins",
            client_state=client_state,
            server_steps_cumulative=server_steps_cumulative,
        )
    )
    res_fed = handle_evaluate_replies(cfg, eval_replies, strategy, current_round)
    if res_fed is not None:
        loss_fed, evaluate_metrics_fed, _ = res_fed
        if loss_fed is not None:
            history.add_loss_distributed(server_round=current_round, loss=loss_fed)
            history.add_metrics_distributed(
                server_round=current_round, metrics=evaluate_metrics_fed
            )
    history.add_metrics_centralized(
        server_round=current_round,
        metrics={
            "server/evaluate_round_time": (time.time_ns() - evaluate_round_time) * 1e-9
        },
    )
    return history
