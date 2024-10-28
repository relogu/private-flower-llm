"""The placement policies used in the Pollen paper.

A placement policy assigns clients to devices according to a certain strategy. The
official strategy representing Pollen is the learning-based placement. However, we also
provide other strategies as baselines. Round-robin should be considered the default
baseline.
"""

import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from heapq import heappop, heappush
import itertools
from collections.abc import Iterator
from copy import copy
from inspect import signature
from logging import DEBUG, ERROR
from math import floor, log10
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from numpy.typing import NDArray
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

from flower_llm.resources_manager import Node
import operator

INVALID_ARGUMENTS_GET_PLACEMENT_FN = """
The `policy` passed to `get_placement_fn` is unknown.
The known values are:
 - `rr` for round robin placement;
 - `srr` for sorted round robin placement;
 - `bu` for batch uniform placement;
 - `lb` for Pollen's learning-based placement;
 - `llb` for Parrot's learning-based placement;
 - `su` for sample uniform placement;
 - `lbu` for logarithm batch uniform placement.
"""


@dataclass(order=True)
class WorkerAssignment:
    """Dataclass to store the worker assignment."""

    model: Any = field(compare=False)
    cids: list[int] = field(compare=False)
    load: float
    node: str = field(compare=False)
    device: str = field(compare=False)
    worker_id: str = field(compare=False)
    is_removed: bool = False

    def __hash__(self) -> int:
        """Hash by the worker_id."""
        return hash(self.worker_id)


def add_worker_assignment(
    priority_queue: list[tuple[float, int, WorkerAssignment]],
    entry_finder: dict[WorkerAssignment, tuple[float, int, WorkerAssignment]],
    counter: Iterator,
    worker_assignment: WorkerAssignment,
    load: float = 0.0,
) -> None:
    """Add a new WorkerAssignment or update its priority."""
    if worker_assignment in entry_finder:
        remove_task(entry_finder, worker_assignment)
    count = next(counter)
    entry = (load, count, worker_assignment)
    entry_finder[worker_assignment] = entry
    heappush(priority_queue, entry)


def remove_task(
    entry_finder: dict[WorkerAssignment, tuple[float, int, WorkerAssignment]],
    worker_assignment: WorkerAssignment,
) -> None:
    """Mark an existing task as REMOVED.  Raise KeyError if not found."""
    _, _, worker_assignment = entry_finder.pop(worker_assignment)
    worker_assignment.is_removed = True


def pop_worker_assignment(
    priority_queue: list[tuple[float, int, WorkerAssignment]],
    entry_finder: dict[WorkerAssignment, tuple[float, int, WorkerAssignment]],
) -> WorkerAssignment:
    """Remove and return the lowest priority task. Raise KeyError if empty."""
    while priority_queue:
        _, _, worker_assignment = heappop(priority_queue)
        if not worker_assignment.is_removed:
            del entry_finder[worker_assignment]
            return worker_assignment
    raise KeyError("pop from an empty priority queue")


def get_placement_fn(policy: str = "rr") -> Callable:
    """Wrap the placement functions. Return them by code.

    Args:
        policy (str, optional): chosen placement policy. Defaults to "rr".

    Returns
    -------
        Callable: placement function.
    """
    if policy == "rr":
        return round_robin_placement
    elif policy == "srr":
        return sorted_round_robin_placement
    elif policy == "bu":
        return batches_placement
    elif policy == "lb":
        return pollen_learning_based_placement
    elif policy == "llb":
        return parrot_learning_based_placement
    elif policy == "su":
        return samples_placement
    elif policy == "lbu":
        return log_batches_placement
    else:
        log(ERROR, INVALID_ARGUMENTS_GET_PLACEMENT_FN)
        sys.exit()


def pollen_learning_based_placement(
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Return the clients' placements according to the Pollen's placement.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    return learning_based_placement(
        fns=[_pollen_function, _jacobian_pollen_function],
        **kwargs,
        # fns=[_linear, _jacobian_linear], **kwargs
    )


def parrot_learning_based_placement(
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Return the clients' placements according to the Parrot's placement.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    return learning_based_placement(
        fns=[_linear, _jacobian_linear], is_parrot=True, **kwargs
    )


def get_pollen_models(
    cids: dict[str | int, int],
    placement_policy: str = "rr",
    batch_size: int = 1,
    # Columns here are: server_round, node, n_samples, delta, device
    clients_stats: pa.Table | None = None,
    server_round: int = 1,
) -> tuple[dict[str, Any] | None, dict[str, pa.Table] | None]:
    """Train models for the given placement policy using the provided clients' stats.

    Args:
        placement_policy (str): The placement policy to use for training the models.
        clients_stats (Optional[pa.Table]): The clients' statistics to use for training
            the models.
        batch_size (int): The batch size to use for training the models.
        cids (Optional[Dict[int, int]]): The client IDs to use for training the models.

    Returns
    -------
        Optional[Dict[str, Any]]: A dictionary containing the trained models.
    """
    if (placement_policy in {"lb", "llb"}) and clients_stats is not None:
        # Set up the functions to use for training the models
        fns: list[
            Callable[[Any, Any, Any], Any] | Callable[[Any, Any, Any, Any], Any]
        ] = (
            # [_pollen_function, _jacobian_pollen_function]
            [_linear, _jacobian_linear]
            if placement_policy == "lb"
            else [_linear, _jacobian_linear]
        )
        # Split clients_stats table into a list of tables, one per client
        splitted_clients_stats: dict[str, pa.Table] = split_clients_training_table(
            clients_stats
        )
        # Create correction tables using the last server round
        correction_tables: dict[str, pa.Table] = {}
        for node_device, _client_stats in splitted_clients_stats.items():
            correction_tables[node_device] = _client_stats.group_by(
                ["n_samples"]
            ).aggregate([("delta", "mean")])
        # Train models
        try:
            pollen_models: dict[str, Any] = sequential_train_models(
                fns, splitted_clients_stats
            )
        except Exception as e:
            log(ERROR, "Failed to train models.", exc_info=e, stack_info=True)
            return None, None
        # Order models from the fastest to the slowest according to the prediction
        # This is a dictionary {'model_name': (trained_model)}
        pollen_models = dict(
            sorted(
                pollen_models.items(),
                key=lambda item: _predict_single_client(
                    model=item[1],
                    fn=fns[0],
                    # NOTE: Choosing an arbitrary number of samples for ordering the
                    # devices by speed
                    # n_samples=batch_size**2,
                    n_samples=3 * batch_size,
                ),
            )
        )
        # # Get models' scores
        # current_scores: dict[str, float] = sequential_get_models_scores(
        #     fns[0], pollen_models, splitted_clients_stats
        # )
        # # Log scores and return trained models
        # log(DEBUG, "Pollen-MLStrategy :: models' scores %s", current_scores)
        return pollen_models, correction_tables
    else:
        return None, None


def learning_based_placement(
    fns: list[Callable],
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    batch_size: int,
    pollen_models: dict[str, Any] | None = None,
    correction_tables: dict[str, pa.Table] | None = None,
    is_parrot: bool = False,
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement generic learning-based placement strategy.

    Args:
        fns (List[Callable]): fit functions and its jacobian.
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        batch_size (int): batch size of ALL clients.
        cids (Union[Dict[str, int], Dict[int, int]]): mapping between cids and
        number of samples.
        clients_stats (pa.Table, optional): collected clients' stats. Defaults to None.
        gpu_stats (pa.Table, optional):  collected GPUs' stats. Defaults to None.
        is_parrot (bool, optional): flag for parrot. Defaults to False.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    if pollen_models is None:  # or gpu_stats is None:
        log(
            DEBUG,
            "Pollen-MLStrategy :: no models have been provided, using RR placement",
        )
        return round_robin_placement(sampled_virtual_cids, nodes_dict)
    else:
        start_time = time.time()
        # Sorting by n samples -- batch size (increasing order b/c popright is faster)
        # This is a list of tuples (cid, list of samples)
        sampled_virtual_cids = sorted(
            sampled_virtual_cids,
            key=operator.itemgetter(1),  # // batch_size,
            reverse=False,
        )
        # Getting nodes a simpler node dict
        simple_node_dict = {node.name: node for k, (c_p, node) in nodes_dict.items()}

        # Creating priority queue, entry finder and counter
        priority_queue: list[tuple[float, int, WorkerAssignment]] = []
        entry_finder: dict[WorkerAssignment, tuple[float, int, WorkerAssignment]] = {}
        counter = cast(Iterator, itertools.count())

        # Create the priority queue of WorkerAssignments
        for model_name, model_params in pollen_models.items():
            node_name = model_name.split("_")[0]
            device_name = model_name.split("_")[1]
            current_node = simple_node_dict[node_name]
            if is_parrot:
                # Parrot uses one worker per device
                tmp_worker_assignment = WorkerAssignment(
                    model=model_params,
                    cids=[],
                    load=0.0,
                    node=node_name,
                    device=device_name,
                    worker_id=node_name + "_" + device_name,
                )
                add_worker_assignment(
                    priority_queue,
                    entry_finder,
                    counter,
                    tmp_worker_assignment,
                )
            else:
                assert current_node.device_info is not None
                concurrency = current_node.device_info[device_name].concurrency
                # Pollen uses `concurrency` workers per device
                for i in range(concurrency):
                    tmp_worker_assignment = WorkerAssignment(
                        model=model_params,
                        cids=[],
                        load=0.0,
                        node=node_name,
                        device=device_name,
                        worker_id=node_name + "_" + device_name + "_" + str(i),
                    )
                    add_worker_assignment(
                        priority_queue,
                        entry_finder,
                        counter,
                        tmp_worker_assignment,
                    )
        # Assign clients to the workers depending on the priority queue
        load_memory: dict[str, dict[int, float]] = {}
        while sampled_virtual_cids:
            # Extract the first element of the list
            virtual_cid, n_samples = sampled_virtual_cids.pop()
            # Pop the WorkerAssignment with the lowest load
            worker_assignment = pop_worker_assignment(priority_queue, entry_finder)
            # Assign client to the least loaded device
            worker_assignment.cids.append(virtual_cid)
            # Get device load
            if (
                f"{worker_assignment.node}_{worker_assignment.device}" in load_memory
                and n_samples
                in load_memory[f"{worker_assignment.node}_{worker_assignment.device}"]
            ):
                load = load_memory[
                    f"{worker_assignment.node}_{worker_assignment.device}"
                ][n_samples]
            else:
                load = _predict_single_client(
                    model=worker_assignment.model,
                    fn=fns[0],
                    n_samples=n_samples,
                )
                # Apply correction table
                load = apply_correction_table(
                    load,
                    correction_tables,
                    f"{worker_assignment.node}_{worker_assignment.device}",
                    n_samples,
                )
                if (
                    f"{worker_assignment.node}_{worker_assignment.device}"
                    not in load_memory
                ):
                    load_memory[
                        f"{worker_assignment.node}_{worker_assignment.device}"
                    ] = {}
                load_memory[f"{worker_assignment.node}_{worker_assignment.device}"][
                    n_samples
                ] = load
            # Update the load of the WorkerAssignment
            worker_assignment.load += load
            # Add the WorkerAssignment back to the priority queue
            add_worker_assignment(
                priority_queue,
                entry_finder,
                counter,
                worker_assignment,
                worker_assignment.load,
            )
        log(
            DEBUG,
            "Pollen-MLStrategy :: estimated loads %s",
            [w[0] for w in priority_queue],
        )
        # Merge workers assignments
        devices_assignment: dict[str, list[list[int]]] = defaultdict(list)
        for _, _, worker_assignment in priority_queue:
            devices_assignment[
                f"{worker_assignment.node}_{worker_assignment.device}"
            ].append(worker_assignment.cids)
        # Build node assignments
        node_assignments = []
        for client_proxy, node in nodes_dict.values():
            devices_assignment_node = {
                node_dev_name.split("_")[1]: str(list_of_cids)
                for node_dev_name, list_of_cids in devices_assignment.items()
                if node_dev_name.split("_")[0] == node.name
            }
            node_assignments.append((client_proxy, devices_assignment_node))
        log(
            DEBUG,
            f"Pollen-MLStrategy :: placement took {time.time() - start_time} seconds",
        )
        if verbose:
            log(
                DEBUG,
                "Pollen-MLStrategy placement :: tuple(node, device assignments) %s",
                node_assignments,
            )
        return node_assignments


def round_robin_placement(
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement Round-Robin placement strategy.

    Args:
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    # Extract cids
    cids: NDArray[np.int16] = np.array([x[0] for x in sampled_virtual_cids])
    # Creates equal clients splits amongst workers
    # in an ordered fashion by index, [1,2,3] split by two -> [1],[2,3].
    # (the remainder is assigned to the last worker)
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
            if node.device_info is not None
        ]
    )
    log(DEBUG, f"Round Robin (RR) placement :: n_total_workers {n_total_workers}")
    splits = np.array_split(cids, n_total_workers)
    # Init the device assignment and the return value
    device_assignment: dict[str, list[list[int]]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items(), strict=False
        ):
            # Loop over devices in the current node
            assert node.device_info is not None
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        device_assignment[device_id].append(current_split.tolist())
    # Covert list of int to string``
    node_assignments: list[tuple[ClientProxy, dict[str, str]]] = [
        (
            c_p,
            {k: str(v) for k, v in device_assignment.items()},
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Round Robin (RR) placement :: tuple(node, device assignments) %s",
            node_assignments,
        )
    return node_assignments


def sorted_round_robin_placement(
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement Sorted Round-Robin placement strategy.

    Args:
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    # Sorting by size (decreasing order)
    sorted_sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=operator.itemgetter(1),
        reverse=True,
    )
    # Extract cids
    cids: NDArray[np.int16] = np.array([x[0] for x in sorted_sampled_virtual_cids])
    # Creates equal clients splits amongst workers by index
    # (the remainder is assigned to the first workers)
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
            if node.device_info is not None
        ]
    )
    splits = [
        cids[np.arange(i, len(cids), n_total_workers)] for i in range(n_total_workers)
    ]
    log(DEBUG, f"Round Robin (RR) placement :: n_total_workers {n_total_workers}")
    # Init the device assignment and the return value
    device_assignment: dict[str, list[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items(), strict=False
        ):
            # Loop over devices in the current node
            assert node.device_info is not None
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        device_assignment[device_id].append(current_split.tolist())
    # Covert list of int to string
    node_assignments: list[tuple[ClientProxy, dict[str, str]]] = [
        (
            c_p,
            {k: str(v) for k, v in device_assignment.items()},
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Sorted Round Robin (SRR) placement :: tuple(node, device assignments) %s",
            node_assignments,
        )
    return node_assignments


def samples_placement(
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement placement strategy based on the number of samples.

    Args:
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    # Sorting by size (decreasing order)
    sorted_sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=operator.itemgetter(1),
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
            if node.device_info is not None
        ]
    )
    # Assign the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    # Assign the remaining clients iteratively to the least loaded worker
    for virtual_cid, num_samples in sorted_sampled_virtual_cids[n_total_workers:]:
        sums = [sum(x[1] for x in list_cids) for list_cids in tmp_splits]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: dict[str, list[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items(), strict=False
        ):
            # Loop over devices in the current node
            assert node.device_info is not None
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        device_assignment[device_id].append(current_split.tolist())
    # Covert list of int to string
    node_assignments: list[tuple[ClientProxy, dict[str, str]]] = [
        (
            c_p,
            {k: str(v) for k, v in device_assignment.items()},
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Number of samples placement :: dict(node, node_assignments) %s",
            node_assignments,
        )
    return node_assignments


def batches_placement(
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    batch_size: int,
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement placement strategy based on the number of batches.

    Args:
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=operator.itemgetter(1),
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
            if node.device_info is not None
        ]
    )
    # Assign the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    for virtual_cid, num_samples in sampled_virtual_cids[n_total_workers:]:
        sums = [
            sum(floor(x[1] / batch_size) for x in list_cids) for list_cids in tmp_splits
        ]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: dict[str, list[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items(), strict=False
        ):
            # Loop over devices in the current node
            assert node.device_info is not None
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        device_assignment[device_id].append(current_split.tolist())
    # Covert list of int to string
    node_assignments: list[tuple[ClientProxy, dict[str, str]]] = [
        (
            c_p,
            {k: str(v) for k, v in device_assignment.items()},
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Number of batches placement :: dict(node, node_assignments) %s",
            node_assignments,
        )
    return node_assignments


def log_batches_placement(
    sampled_virtual_cids: list[tuple[int, int]],
    nodes_dict: dict[str, tuple[ClientProxy, Node]],
    batch_size: int,
    verbose: bool = False,
    **kwargs: Any,
) -> list[tuple[ClientProxy, dict[str, str]]]:
    """Implement placement strategy based on the log of the number of batches.

    Args:
        sampled_virtual_cids (List[Tuple[int, int]]): sampled virtual cids.
        nodes_dict (Dict[str, Tuple[ClientProxy, Node]]): dictionary of nodes'
        resources.
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=operator.itemgetter(1),
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
            if node.device_info is not None
        ]
    )
    # Assign the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    for virtual_cid, num_samples in sampled_virtual_cids[n_total_workers:]:
        sums = [
            sum(log10(floor(x[1] / batch_size)) for x in list_cids)
            for list_cids in tmp_splits
        ]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: dict[str, list[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items(), strict=False
        ):
            # Loop over devices in the current node
            assert node.device_info is not None
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        device_assignment[device_id].append(current_split.tolist())
    # Covert list of int to string
    node_assignments: list[tuple[ClientProxy, dict[str, str]]] = [
        (
            c_p,
            {k: str(v) for k, v in device_assignment.items()},
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Logarithm of num of batches placement :: dict(node, node_assignments) %s",
            node_assignments,
        )
    return node_assignments


def _logarithm(x, a, b) -> Any:  # noqa: ANN001
    y = a + b * np.log(x)
    return y


def _pollen_function(x, a, b, c) -> Any:  # noqa: ANN001
    y = a + b * np.log(x) + c * x
    return y


def _linear(x, a, b) -> Any:  # noqa: ANN001
    y = a + b * x
    return y


def _jacobian_logarithm(x, a, b) -> NDArray[Any]:  # noqa: ANN001
    grad_a = np.ones_like(x)
    grad_b = np.log(x)
    return np.hstack((grad_a.reshape(-1, 1), grad_b.reshape(-1, 1)))


def _jacobian_pollen_function(x, a, b, c) -> NDArray[Any]:  # noqa: ANN001
    grad_a = np.ones_like(x)
    grad_b = np.log(x)
    return np.hstack((grad_a.reshape(-1, 1), grad_b.reshape(-1, 1), x.reshape(-1, 1)))


def _jacobian_linear(x, a, b) -> NDArray:  # noqa: ANN001
    grad_a = np.ones_like(x)
    return np.hstack((grad_a.reshape(-1, 1), x.reshape(-1, 1)))


def _predict_single_client(model: tuple[Any, Any], fn: Callable, n_samples: int) -> Any:
    parameters, _ = model
    return fn(n_samples, *parameters)


def apply_correction_table(
    load: float,
    correction_tables: dict[str, pa.Table] | None,
    device_name: str,
    n_samples: int,
) -> float:
    """Apply the correction table to the given load."""
    # Filter out -inf predictions, if any
    if load == -np.inf:
        load = 1e-8
    if correction_tables is not None:
        # NOTE: A correction table has columns: n_samples, delta_mean
        correction: pa.Table = correction_tables[device_name].filter(
            pc.field("n_samples") == pc.scalar(n_samples)
        )
        if correction.num_rows > 0:
            load_correction = float(correction.column("delta_mean").to_numpy()[0])
            # load = (load + load_correction) / 2
            load = load_correction
    return load


def skim_clients_training_stats(
    clients_training_stats: pa.Table,
) -> pa.Table:
    """Skim the clients' training stats keeping just the average per n_samples."""
    # Initialize the list of skimmed tables
    table_list: list[pa.Table] = []
    # Split the table into a list of tables, one per GPU in each node
    # Columns here are: server_round, node, n_samples, delta, device
    for table in split_clients_training_table(clients_training_stats).values():
        # Extract constant information: node, device
        node = table.column("node").to_numpy()[0]
        device = table.column("device").to_numpy()[0]
        # Make up the server round
        server_round = 0
        # Drop unused columns for aggregation
        table = table.drop_columns(["node", "device", "server_round"])  # type: ignore[attr-defined, reportAttributeAccessIssue]
        # Aggregate the table by n_samples and average over the delta
        table = table.group_by(["n_samples"]).aggregate(
            [("delta", "mean")]  # , ("delta", "stddev")]
        )
        # Rename column `delta_mean` to `delta`
        table = table.rename_columns(["n_samples", "delta"])  # type: ignore[attr-defined]
        # Add constant columns back
        table = add_constant_column_to_clients_stats_table(table, "device", device)
        table = add_constant_column_to_clients_stats_table(table, "node", node)
        table = add_constant_column_to_clients_stats_table(
            table, "server_round", server_round
        )
        # Append the table to the list
        table_list.append(table)
    return pa.concat_tables(table_list)


def add_constant_column_to_clients_stats_table(
    input_table: pa.Table,
    column_name: str,
    constant_value: Any,
) -> pa.Table:
    """Add a constant column to the given Table."""
    return input_table.add_column(
        0,
        column_name,
        cast(pa.Array, pa.array([constant_value] * input_table.num_rows)),
    )


def split_clients_training_table(input_table: pa.Table) -> dict[str, pa.Table]:
    """Split the training table into a list of tables, one per GPU.

    Args:
        input_table (pa.Table): the training table.

    Returns
    -------
        List[pa.Table]: a list of tables, one per GPU.
    """
    # Get the list of unique node names
    unique_node_names = np.unique(input_table.column("node").to_numpy())
    # Get the list of unique GPU names
    unique_gpu_names = np.unique(input_table.column("device").to_numpy())
    # Create cross product iterator
    iterator = ((a, b) for a in unique_node_names for b in unique_gpu_names)
    # Create a list of tables, one per GPU
    output = {
        f"{a}_{b}": input_table.filter(pc.field("node") == pc.scalar(a)).filter(
            pc.field("device") == pc.scalar(b)
        )
        for a, b in iterator
    }
    # Remove None values
    output = {k: v for k, v in output.items() if v.num_rows != 0}
    # Return the cleaned list of tables
    return output


def sequential_train_models(
    fns: list[Callable], clients_stats: dict[str, pa.Table]
) -> dict[str, Any]:
    """Train the models sequentially."""
    # Init return dict
    ret = {}
    # Loop over model names
    for k, v in clients_stats.items():
        # v = v.group_by(["n_samples"]).aggregate(
        #     [("delta", "mean"), ("delta", "stddev")]
        # )
        ret.update(_train_model(fns, k, v))
    return ret


def _train_model(
    fns: list[Callable], model_name: str, data: pa.Table
) -> dict[str, Any]:
    # x = data.column("n_batches").to_numpy()
    x = data.column("n_samples").to_numpy()
    delta = data.column("delta").to_numpy()
    # delta = data.column("delta_mean").to_numpy()
    # delta_stddev = data.column("delta_stddev").to_numpy()
    # delta_stddev = delta_stddev.clip(min=1e-9)
    # delta_stddev = [1 / a for a in x]
    p0 = [1e-9] + [1.0] * (len(signature(fns[0]).parameters) - 2)
    bounds = (0.0, np.inf)
    return {
        model_name: curve_fit(
            f=fns[0],
            xdata=x,
            ydata=delta,
            # sigma=delta_stddev,
            # absolute_sigma=True,
            p0=p0,
            bounds=bounds,
            jac=fns[1],
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            # loss="arctan",
            check_finite=True,
            nan_policy="omit",
            # max_nfev=10000,
        )
    }


def sequential_get_models_scores(
    fn: Callable, models: dict[str, Any], clients_stats: dict[str, pa.Table]
) -> dict[str, float]:
    """Return the scores of the model computed sequentially."""
    # Init return dict
    ret = {}
    # Loop over model names
    for k, model in models.items():
        v = clients_stats[k]
        ret.update(_get_model_score(fn, k, model, v))
    return ret


def _get_model_score(
    fn: Callable, model_name: str, model: Any, data: pa.Table
) -> dict[str, float]:
    parameters, _ = model
    x = data.column("n_samples").to_numpy()
    delta = data.column("delta").to_numpy()
    plt.scatter(x, delta, label="data", alpha=0.5, marker=".", c="r")
    plt.scatter(x, fn(x, *parameters), label="model", alpha=0.5, marker=".", c="b")
    plt.savefig(f"{model_name}.png")
    plt.close()
    return {model_name: np.sum(np.abs(delta - fn(x, *parameters)))}
