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
from logging import DEBUG, ERROR
from math import floor, log10
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Tuple, Union

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
    return learning_based_placement(fns=[_linear, _jacobian_linear], **kwargs)


def learning_based_placement(
    fns: List[Callable],
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Tuple[ClientProxy, Node]],
    batch_size: int,
    cids: Union[Dict[str, int], Dict[int, int]],
    clients_stats: pa.Table = None,
    gpu_stats: pa.Table = None,
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
        verbose (bool, optional): flag for logger. Defaults to False.

    Returns
    -------
        List[Tuple[ClientProxy, Dict[str, str]]]: a list of tuples
        (client_proxy, device_assignment).
    """
    if clients_stats is None:  #  or gpu_stats is None:
        return round_robin_placement(sampled_virtual_cids, nodes_dict)
    else:
        start_time = time.time()
        ## Prepare data
        # Add n_batches column to clients_stats table
        # t_0 = time.time()
        clients_stats = add_n_batches_column_to_clients_stats_table(
            clients_stats, batch_size, cids
        )
        # log(
        #     DEBUG,
        #     "Pollen-MLStrategy :: add batches to table took %s seconds",
        #     time.time()-t_0
        # )
        # TODO: Come up with a procedure when a new NodeManager appears after round 1
        # TODO: Deal with dropped NodeManagers
        # Split clients_stats table into a list of tables, one per client
        # t_0 = time.time()
        clients_stats: Dict[str, pa.Table] = split_clients_training_table(clients_stats)
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: splitting tables took {time.time()-t_0} seconds",
        # )
        # Train models
        # t_0 = time.time()
        # trained_models: Dict[str, Any] = parallel_train_models(fns, clients_stats)
        trained_models: Dict[str, Any] = sequential_train_models(fns, clients_stats)
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: training models took {time.time()-t_0} seconds",
        # )
        # Get models' scores
        # t_0 = time.time()
        # current_scores: Dict[str, float] = parallel_get_models_scores(
        #     fns, trained_models, clients_stats
        # )
        current_scores: Dict[str, float] = sequential_get_models_scores(
            fns[0], trained_models, clients_stats
        )
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: getting scores took {time.time()-t_0} seconds",
        # )
        # Check scores
        log(DEBUG, "Pollen-MLStrategy :: models' scores %s", current_scores)

        # TODO: Estimate the threshold to fall back to RR
        # if max([abs(score) for score in current_scores]) > 10e-1:
        #     return round_robin_placement(sampled_virtual_cids, nodes_dict)
        # TODO: Adaptive discard of the old data

        # Sorting by batch size (decreasing order)
        # This is a list of tuples (cid, list of samples)
        # t_0 = time.time()
        sampled_virtual_cids = sorted(
            sampled_virtual_cids,
            key=lambda x: x[1] // batch_size,
            reverse=True,
        )
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: sorting clients took {time.time()-t_0} seconds",
        # )
        # t_0 = time.time()
        # Order models from the fastest to the slowest according to the prediction
        # This is a dictionary {'model_name': (trained_model)}
        trained_models = dict(
            sorted(
                trained_models.items(),
                key=lambda item: _predict_single_client(
                    model=item[1],
                    fn=fns[0],
                    n_samples=sampled_virtual_cids[0][1],
                    batch_size=batch_size,
                ),
            )
        )
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: sorting devices took {time.time()-t_0} seconds",
        # )

        # Init the device assignment and the return value
        # t_0 = time.time()
        devices_assignment = [
            [
                v,  # Model parameters
                [],  # List of cids
                0.0,  # Device load
                k.split("_")[0],  # Node name
                k.split("_")[1],  # Device name
            ]
            for k, v in trained_models.items()
        ]
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: init assignments took {time.time()-t_0} seconds",
        # )

        # Assignment
        # t_0 = time.time()
        while len(sampled_virtual_cids) > 0:
            # Extract the first element of the list
            virtual_cid, num_samples = sampled_virtual_cids.pop(0)
            # Assign client to the least loaded device
            devices_assignment[0][1].append(virtual_cid)
            # Get device load
            load = _predict_single_client(
                model=devices_assignment[0][0],
                fn=fns[0],
                n_samples=num_samples,
                batch_size=batch_size,
            )
            devices_assignment[0][2] += load
            # Sort devices by load (increasing order)
            devices_assignment = sorted(
                devices_assignment,
                key=lambda x: x[2],
            )
        # log(
        #     DEBUG,
        #     f"Pollen-MLStrategy :: assignment took {time.time()-t_0} seconds",
        # )

        # Build node assignments
        # t_0 = time.time()
        node_assignments = []
        for _, (client_proxy, node) in nodes_dict.items():
            devices_assignment_node = {
                dev_name: _convert_list_of_int_to_string(list_of_cids)
                for _, list_of_cids, _, node_dev_name, dev_name in devices_assignment
                if node_dev_name == node.name
            }
            node_assignments.append((client_proxy, devices_assignment_node))
        # log(
        #     DEBUG,
        #     "Pollen-MLStrategy :: building node assignments %s seconds",
        #     time.time()-t_0,
        # )
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
    sampled_virtual_cids: NDArray[np.int16] = np.array(
        [x[0] for x in sampled_virtual_cids]
    )
    # Creates equal clients splits amongst workers
    # in an ordered fashon by index, [1,2,3] split by two -> [1],[2,3].
    # (the remainder is assigned to the last worker)
    n_total_workers = np.sum(
        [
            [device.concurrency for _, device in node.device_info.items()]
            for _, (_, node) in nodes_dict.items()
        ]
    )
    log(DEBUG, f"Round Robin (RR) placement :: n_total_workers {n_total_workers}")
    splits = np.array_split(sampled_virtual_cids, n_total_workers)
    # Init the device assignment and the return value
    device_assignment = defaultdict(list)
    node_assignments = [
        (client_proxy, copy(device_assignment))
        for _, (client_proxy, _) in nodes_dict.items()
    ]
    # Loop over the splits created
    while len(splits) > 0:
        # Loop over nodes
        for (_c_p, device_assignment), (_, (_, node)) in zip(
            node_assignments, nodes_dict.items()
        ):
            # Loop over devices in the current node
            for device_id, device in node.device_info.items():
                for _ in range(device.concurrency):
                    current_split = splits.pop(0)
                    if len(current_split) > 0:
                        [device_assignment[device_id].append(c) for c in current_split]
    # Covert list of int to string
    node_assignments = [
        (
            c_p,
            {
                k: _convert_list_of_int_to_string(v)
                for k, v in device_assignment.items()
            },
        )
        for c_p, device_assignment in node_assignments
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
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Extract cids
    sampled_virtual_cids: NDArray[np.int16] = np.array(
        [x[0] for x in sampled_virtual_cids]
    )
    # Creates equal clients splits amongst workers by index
    # (the remainder is assigned to the first workers)
    splits = [
        sampled_virtual_cids[np.arange(i, len(sampled_virtual_cids), len(nodes_dict))]
        for i in range(len(nodes_dict))
    ]
    node_assignments = []
    for _, (client_proxy, node) in nodes_dict.items():
        device_assignment: Dict[str, str] = {}
        for device_id, device in node.device_info.items():
            current_split = splits.pop(0)
            log(DEBUG, f"current_split {current_split}")
            if len(current_split) > 0:
                device_assignment[device_id] = ",".join(
                    [
                        ",".join([str(cid) for cid in current_split])
                        for _ in range(device.concurrency)
                    ]
                )
        node_assignments.append((client_proxy, device_assignment))
    if verbose:
        log(
            DEBUG,
            "Sorted Round Robin (SRR) placement :: tuple(node, device assignements) %s",
            node_assignments,
        )
    return node_assignments


def samples_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
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
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[: len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict) :]:
        sums = [sum([x[1] for x in list_cids]) for list_cids in splits]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in splits]
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(
            DEBUG, f"Number of samples placement :: dict(worker_id, [cids]) {placement}"
        )
    return lists_cids


def batches_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
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
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[: len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict) :]:
        sums = [
            sum([floor(x[1] / batch_size) for x in list_cids]) for list_cids in splits
        ]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids]) for list_cids in splits]
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(
            DEBUG, f"Number of batches placement :: dict(worker_id, [cids]) {placement}"
        )
    return lists_cids


def log_batches_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
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
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[: len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict) :]:
        sums = [
            sum([log10(floor(x[1] / batch_size)) for x in list_cids])
            for list_cids in splits
        ]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(
            DEBUG,
            "Logarithm of number of batches placement :: dict(worker_id, [cids]) %s",
            placement,
        )
    return lists_cids


def _pollen_function(x, A, B):
    y = A + B * np.log(x)
    return y


def _linear(x, A, B):
    y = A + B * x
    return y


def _jacobian_pollen_function(x, A, B):
    dA = np.ones_like(x)
    dB = np.log(x)
    return np.hstack((dA.reshape(-1, 1), dB.reshape(-1, 1)))


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
    cids: Union[Dict[str, int], Dict[int, int]],
) -> pa.Table:
    """Add a `num_batches` column to the given Table."""
    return input.add_column(
        0,
        "n_batches",
        pa.array([cids[int(cid.as_py())] // batch_size for cid in input["cid"]]),
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
    # NOTE: This might be unnecessary with the defaults in the `filter` function
    # Remove None values
    output = {k: v for k, v in output.items() if v is not None}
    # log(
    #     DEBUG,
    #     "split_clients_training_table after checking for Nones :: output %s",
    #     output,
    # )
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
    return {
        model_name: curve_fit(
            f=fns[0],
            xdata=x,
            ydata=delta,
            p0=[0.0, 0.0],
            jac=fns[1],
            ftol=1e-3,
            xtol=1e-3,
            gtol=1e-3,
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
    return {model_name: np.sum((delta - fn(x, *parameters)))}
