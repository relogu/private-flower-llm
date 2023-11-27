"""The placement policies used in the Pollen paper.

A placement policy asssigns clients to devices according to a certain strategy. The
official strategy representing Pollen is the learning-based placement. However, we also
provide other strategies as baselines. Round-robin should be considered the default
baseline.
"""
import sys
import time
from collections import defaultdict
from copy import copy
from inspect import signature
from logging import DEBUG, ERROR
from math import floor, log10
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.compute as pc
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from numpy.typing import NDArray
from scipy.optimize import curve_fit

from pollen_worker.resources_manager import Node

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
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
    """Return the cliets' placements according to the Pollen's learning-based placement.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    return learning_based_placement(
        fns=[_pollen_function, _jacobian_pollen_function], **kwargs
    )


def parrot_learning_based_placement(
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
    """Return the cliets' placements according to the Parrot's learning-based placement.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    return learning_based_placement(
        fns=[_linear, _jacobian_linear], is_parrot=True, **kwargs
    )


def get_pollen_models(
    cids: Dict[Union[str, int], int],
    placement_policy: str = "rr",
    batch_size: int = 1,
    clients_stats: Optional[pa.Table] = None,
    server_round: int = 1,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, pa.Table]]]:
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
    if (
        placement_policy == "lb" or placement_policy == "llb"
    ) and clients_stats is not None:
        # Set up the functions to use for training the models
        fns: List[
            Union[Callable[[Any, Any, Any], Any], Callable[[Any, Any, Any, Any], Any]]
        ] = (
            [_pollen_function, _jacobian_pollen_function]
            if placement_policy == "lb"
            else [_linear, _jacobian_linear]
        )
        # Add n_batches column to clients_stats table
        clients_stats = add_n_batches_column_to_clients_stats_table(
            clients_stats, batch_size, cids
        )
        # Split clients_stats table into a list of tables, one per client
        splitted_clients_stats: Dict[str, pa.Table] = split_clients_training_table(
            clients_stats
        )
        # Create correction tables
        correction_tables: Dict[str, pa.Table] = {}
        for client_id, _client_stats in splitted_clients_stats.items():
            filtered_client_stats = _client_stats.filter(
                pc.field("server_round") == pc.scalar(server_round - 1),
                null_selection_behavior="emit_null",
            )
            y1 = np.array(filtered_client_stats.column("end_time").flatten())
            y0 = np.array(filtered_client_stats.column("start_time").flatten())
            ctt = (y1 - y0) * 1e-9
            filtered_client_stats = filtered_client_stats.add_column(
                0,
                "ctt",
                cast(pa.Array, pa.array(np.array(ctt).flatten())),
            )
            correction_tables[client_id] = filtered_client_stats.group_by(
                ["n_batches"]
            ).aggregate([("ctt", "mean")])
        # Train models
        pollen_models: Dict[str, Any] = sequential_train_models(
            fns, splitted_clients_stats
        )
        # Order models from the fastest to the slowest according to the prediction
        # This is a dictionary {'model_name': (trained_model)}
        pollen_models = dict(
            sorted(
                pollen_models.items(),
                key=lambda item: _predict_single_client(
                    model=item[1],
                    fn=fns[0],
                    # n_samples=batch_size**2,
                    n_samples=3 * batch_size,
                    batch_size=batch_size,
                ),
            )
        )
        # Get models' scores
        current_scores: Dict[str, float] = sequential_get_models_scores(
            fns[0], pollen_models, splitted_clients_stats
        )
        # Log scores and return trained models
        log(DEBUG, "Pollen-MLStrategy :: models' scores %s", current_scores)
        return pollen_models, correction_tables
    else:
        return None, None


def learning_based_placement(
    fns: List[Callable],
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    batch_size: int,
    pollen_models: Optional[Dict[str, Any]] = None,
    correction_tables: Optional[Dict[str, pa.Table]] = None,
    is_parrot: bool = False,
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
    if pollen_models is None:  #  or gpu_stats is None:
        log(
            DEBUG,
            "Pollen-MLStrategy :: no models have been provided, using RR placement",
        )
        return round_robin_placement(sampled_virtual_cids, nodes_dict)
    else:
        start_time = time.time()
        # Sorting by batch size (decreasing order)
        # This is a list of tuples (cid, list of samples)
        sampled_virtual_cids = sorted(
            sampled_virtual_cids,
            key=lambda x: x[1] // batch_size,
            reverse=True,
        )
        # Getting nodes a simpler node dict
        simple_node_dict = {node.name: node for k, (c_p, node) in nodes_dict.items()}

        # Init the device assignment and the return value
        workers_assignments = []
        for model_name, model_params in pollen_models.items():
            if is_parrot:
                # Parrot uses one worker per device
                workers_assignments.append(
                    [
                        model_params,  # Model parameters
                        [],  # List of cids
                        0.0,  # Device load
                        model_name.split("_")[0],  # Node name
                        model_name.split("_")[1],  # Device name
                    ]
                )
            else:
                concurrency = (
                    simple_node_dict[model_name.split("_")[0]]
                    .device_info[model_name.split("_")[1]]
                    .concurrency
                )
                for _ in range(concurrency):
                    # Pollen uses `concurrency` workers per device
                    workers_assignments.append(
                        [
                            model_params,  # Model parameters
                            [],  # List of cids
                            0.0,  # Device load
                            model_name.split("_")[0],  # Node name
                            model_name.split("_")[1],  # Device name
                        ]
                    )
        ## Assignment
        # Assign initially at least one client per worker
        for worker in workers_assignments:
            # Extract the first element of the list
            virtual_cid, num_samples = sampled_virtual_cids.pop(0)
            # Assign client to the current worker
            worker[1].append(virtual_cid)
            # Get device load
            load = _predict_single_client(
                model=worker[0],
                fn=fns[0],
                n_samples=num_samples,
                batch_size=batch_size,
            )
            if correction_tables is not None:
                correction = correction_tables[f"{worker[3]}_{worker[4]}"].filter(
                    pc.field("n_batches") == pc.scalar(num_samples // batch_size)
                )
                if correction.num_rows > 0:
                    correction = correction.column("ctt_mean").to_numpy()[0]
                    # load = (load + correction) / 2
                    load = load
            worker[2] += load
        # Assing all the rest
        while len(sampled_virtual_cids) > 0:
            # Sort devices by load (increasing order)
            workers_assignments = sorted(
                workers_assignments,
                key=lambda x: x[2],
            )
            # Extract the first element of the list
            virtual_cid, num_samples = sampled_virtual_cids.pop(0)
            # Assign client to the least loaded device
            workers_assignments[0][1].append(virtual_cid)
            # Get device load
            load = _predict_single_client(
                model=workers_assignments[0][0],
                fn=fns[0],
                n_samples=num_samples,
                batch_size=batch_size,
            )
            if correction_tables is not None:
                correction = correction_tables[f"{worker[3]}_{worker[4]}"].filter(
                    pc.field("n_batches") == pc.scalar(num_samples // batch_size)
                )
                if correction.num_rows > 0:
                    correction = correction.column("ctt_mean").to_numpy()[0]
                    load = (load + correction) / 2
                    # load = correction
            workers_assignments[0][2] += load
        # log(
        #     DEBUG,
        #     "Pollen-MLStrategy :: estimated loads %s",
        #     [w[2] for w in workers_assignments],
        # )
        # Merge workers assignments
        devices_assignment: Dict[str, List[int]] = defaultdict(list)
        for worker in workers_assignments:
            # log(
            #     DEBUG,
            #     f"Pollen-MLStrategy :: worker assignments {worker}",
            # )
            _, list_of_cids, _, node_name, gpu_name = worker
            devices_assignment[f"{node_name}_{gpu_name}"].extend(list_of_cids)
        # Build node assignments
        node_assignments = []
        for _, (client_proxy, node) in nodes_dict.items():
            devices_assignment_node = {
                node_dev_name.split("_")[1]: _convert_list_of_int_to_string(
                    list_of_cids
                )
                for node_dev_name, list_of_cids in devices_assignment.items()
                if node_dev_name.split("_")[0] == node.name
            }
            node_assignments.append((client_proxy, devices_assignment_node))
        log(
            DEBUG,
            f"Pollen-MLStrategy :: placement took {time.time()-start_time} seconds",
        )
        if verbose:
            log(
                DEBUG,
                "Pollen-MLStrategy placement :: tuple(node, device assignements) %s",
                node_assignments,
            )
        return node_assignments


def round_robin_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
    # in an ordered fashon by index, [1,2,3] split by two -> [1],[2,3].
    # (the remainder is assigned to the last worker)
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
        ]
    )
    log(DEBUG, f"Round Robin (RR) placement :: n_total_workers {n_total_workers}")
    splits = np.array_split(cids, n_total_workers)
    # Init the device assignment and the return value
    device_assignment: Dict[str, List[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        for c in current_split:
                            device_assignment[device_id].append(c)
    # Covert list of int to string
    node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Round Robin (RR) placement :: tuple(node, device assignements) %s",
            node_assignments,
        )
    return node_assignments


def sorted_round_robin_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
        key=lambda x: x[1],
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
        ]
    )
    splits = [
        cids[np.arange(i, len(cids), n_total_workers)] for i in range(n_total_workers)
    ]
    log(DEBUG, f"Round Robin (RR) placement :: n_total_workers {n_total_workers}")
    # Init the device assignment and the return value
    device_assignment: Dict[str, List[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        for c in current_split:
                            device_assignment[device_id].append(c)
    # Covert list of int to string
    node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
        )
        for c_p, device_assignment in tmp_node_assignments
    ]
    if verbose:
        log(
            DEBUG,
            "Sorted Round Robin (SRR) placement :: tuple(node, device assignements) %s",
            node_assignments,
        )
    return node_assignments


def samples_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
        key=lambda x: x[1],
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
        ]
    )
    # Assing the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    # Assign the remaining clients iteratively to the least loaded worker
    for virtual_cid, num_samples in sorted_sampled_virtual_cids[n_total_workers:]:
        sums = [sum([x[1] for x in list_cids]) for list_cids in tmp_splits]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: Dict[str, List[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        for c in current_split:
                            device_assignment[device_id].append(c)
    # Covert list of int to string
    node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
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
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
        key=lambda x: x[1],
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
        ]
    )
    # Assing the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    for virtual_cid, num_samples in sampled_virtual_cids[n_total_workers:]:
        sums = [
            sum([floor(x[1] / batch_size) for x in list_cids])
            for list_cids in tmp_splits
        ]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: Dict[str, List[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        for c in current_split:
                            device_assignment[device_id].append(c)
    # Covert list of int to string
    node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
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
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> List[Tuple[ClientProxy, Dict[str, str]]]:
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
        key=lambda x: x[1],
        reverse=True,
    )
    # Get the total number of workers
    n_total_workers = np.sum(
        [
            np.sum([device.concurrency for _, device in node.device_info.items()])
            for _, (_, node) in nodes_dict.items()
        ]
    )
    # Assing the first `n_total_workers` clients to the workers
    tmp_splits = [[c] for c in sampled_virtual_cids[:n_total_workers]]
    for virtual_cid, num_samples in sampled_virtual_cids[n_total_workers:]:
        sums = [
            sum([log10(floor(x[1] / batch_size)) for x in list_cids])
            for list_cids in tmp_splits
        ]
        min_worker = np.argmin(sums)
        tmp_splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in tmp_splits]
    # Init the device assignment and the return value
    device_assignment: Dict[str, List[int]] = defaultdict(list)
    tmp_node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            tmp_node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        for c in current_split:
                            device_assignment[device_id].append(c)
    # Covert list of int to string
    node_assignments: List[Tuple[ClientProxy, Dict[str, str]]] = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
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


def _logarithm(x, A, B):
    y = A + B * np.log(x)
    return y


def _pollen_function(x, A, B, C):
    y = A + B * np.log(x) + C * x
    return y


def _linear(x, A, B):
    y = A + B * x
    return y


def _jacobian_logarithm(x, A, B):
    dA = np.ones_like(x)
    dB = np.log(x)
    return np.hstack((dA.reshape(-1, 1), dB.reshape(-1, 1)))


def _jacobian_pollen_function(x, A, B, C):
    dA = np.ones_like(x)
    dB = np.log(x)
    return np.hstack((dA.reshape(-1, 1), dB.reshape(-1, 1), x.reshape(-1, 1)))


def _jacobian_linear(x, A, B):
    dA = np.ones_like(x)
    return np.hstack((dA.reshape(-1, 1), x.reshape(-1, 1)))


def _predict_single_client(model, fn: Callable, n_samples: int, batch_size: int):
    parameters, covariance = model
    return fn(n_samples // batch_size, *parameters)


def _convert_list_of_int_to_string(list_of_int: List[int]) -> str:
    return ",".join([str(i) for i in list_of_int])


def add_n_batches_column_to_clients_stats_table(
    input: pa.Table,
    batch_size: int,
    cids: Dict[Union[str, int], int],
) -> pa.Table:
    """Add a `num_batches` column to the given Table."""
    return input.add_column(
        0,
        "n_batches",
        cast(
            pa.Array,
            pa.array([cids[int(cid.as_py())] // batch_size for cid in input["cid"]]),
        ),
    )


def split_clients_training_table(input: pa.Table) -> Dict[str, pa.Table]:
    """Split the training table into a list of tables, one per GPU.

    Args:
        input (pa.Table): the training table.

    Returns
    -------
        List[pa.Table]: a list of tables, one per GPU.
    """
    # Get the list of unique node names
    unique_node_names = np.unique(input.column("node").to_numpy())
    # Get the list of unique GPU names
    unique_gpu_names = np.unique(input.column("gpu").to_numpy())
    # Create cross product iterator
    iterator = ((a, b) for a in unique_node_names for b in unique_gpu_names)
    # Create a list of tables, one per GPU
    output = {
        f"{a}_{b}": input.filter(pc.field("node") == pc.scalar(a)).filter(
            pc.field("gpu") == pc.scalar(b)
        )
        for a, b in iterator
    }
    # Remove None values
    output = {k: v for k, v in output.items() if v.num_rows != 0}
    # Return the cleaned list of tables
    return output


def parallel_train_models(
    fns: List[Callable], clients_stats: Dict[str, pa.Table]
) -> Dict[str, Any]:
    """Train the models in parallel."""
    # Set up the parallelisation
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus
    n_jobs = min(n_jobs, len(clients_stats))
    pool = Pool(n_jobs)
    # Execute the pool
    pool_outputs = pool.starmap(
        _train_model, [[fns, k, v] for k, v in clients_stats.items()]
    )
    pool.close()
    pool.join()
    # Return the results from the pool
    return {k: v for result in pool_outputs for k, v in result.items()}


def sequential_train_models(
    fns: List[Callable], clients_stats: Dict[str, pa.Table]
) -> Dict[str, Any]:
    """Train the models sequentially."""
    # Init return dict
    ret = {}
    # Loop over model names
    for k, v in clients_stats.items():
        ret.update(_train_model(fns, k, v))
    return ret


def _train_model(
    fns: List[Callable], model_name: str, data: pa.Table
) -> Dict[str, Any]:
    x = data.column("n_batches").to_numpy()
    y1 = data.column("end_time").to_numpy()
    y0 = data.column("start_time").to_numpy()
    delta = (y1 - y0) * 1e-9
    p0 = [0.0] * (len(signature(fns[0]).parameters) - 1)
    bounds = (0.0, np.inf)
    return {
        model_name: curve_fit(
            f=fns[0],
            xdata=x,
            ydata=delta,
            p0=p0,
            bounds=bounds,
            jac=fns[1],
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
            loss="arctan",
        )
    }


def parallel_get_models_scores(
    fn: Callable, models: Dict[str, Any], clients_stats: Dict[str, pa.Table]
) -> Dict[str, float]:
    """Return the scores of the model computed in parallel."""
    # Set up the parallelisation
    n_jobs = 100
    try:
        cpus = len(psutil.Process().cpu_affinity())  # type: ignore
    except AttributeError:
        cpus = psutil.cpu_count()
    if n_jobs > cpus:
        n_jobs = cpus
    n_jobs = min(n_jobs, len(models))
    pool = Pool(n_jobs)
    # Execute the pool
    pool_outputs = pool.starmap(
        _get_model_score, [[fn, k, models[k], clients_stats[k]] for k in models.keys()]
    )
    pool.close()
    pool.join()
    # Return the results from the pool
    return {k: v for result in pool_outputs for k, v in result.items()}


def sequential_get_models_scores(
    fn: Callable, models: Dict[str, Any], clients_stats: Dict[str, pa.Table]
) -> Dict[str, float]:
    """Return the scores of the model computed sequentially."""
    # Init return dict
    ret = {}
    # Loop over model names
    for k in models.keys():
        ret.update(_get_model_score(fn, k, models[k], clients_stats[k]))
    return ret


def _get_model_score(
    fn: Callable, model_name: str, model: Any, data: pa.Table
) -> Dict[str, float]:
    parameters, _ = model
    x = data.column("n_batches").to_numpy()
    y1 = data.column("end_time").to_numpy()
    y0 = data.column("start_time").to_numpy()
    delta = (y1 - y0) * 1e-9
    # return {model_name: np.sum(np.abs(delta - fn(x, *parameters)))}
    return {model_name: (np.square(delta - fn(x, *parameters))).mean()}
