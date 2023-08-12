import sys
from collections import defaultdict
from logging import DEBUG, ERROR
from math import floor, log10
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flwr.common.logger import log
from scipy.optimize import curve_fit
from pollen_utils import (
    get_clients_dataframe_from_dict,
    get_ctt_dataframe_from_pickle,
    merge_ctt_size,
    invert_many_to_one_dictionary,
)
from resources_manager import Node

INVALID_ARGUMENTS_GET_PLACEMENT_FN = """
The `policy` passed to `get_placement_fn` is unknown.
The known values are:
 - `rr` for round robin placement;
 - `srr` for sorted round robin placement;
 - `bu` for batch uniform placement;
 - `lb` for learning-based placement;
 - `su` for sample uniform placement;
 - `lbu` for logarithm batch uniform placement.
"""

def get_placement_fn(policy: str = 'rr'):
    if policy == 'rr':
        return round_robin_placement
    elif policy == 'srr':
        return sorted_round_robin_placement
    elif policy == 'bu':
        return batches_placement
    elif policy == 'lb':
        return learning_based_placement
    elif policy == 'su':
        return samples_placement
    elif policy == 'lbu':
        return log_batches_placement
    else:
        log(ERROR, INVALID_ARGUMENTS_GET_PLACEMENT_FN)
        sys.exit()


def learning_based_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    server_round: int,
    batch_size: int,
    cids: Dict[str, int],
    worker_to_resource: Dict[str, Tuple[str, str, int, int]],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    if server_round == 1:
        return round_robin_placement(
            sampled_virtual_cids, nodes_dict)
    else:
        current_scores = []
        map_workers_models = {
            k: f'{node}_{gpu_name}'
            for k, (node, gpu_name, gpu_id, concurrency)
            in worker_to_resource.items()
        }
        map_models_workers = invert_many_to_one_dictionary(
            map_workers_models
        )
        for model_name, worker_ids in map_models_workers.items():
            try:
                # TODO: Load data, re-think this procedure to make it more
                # efficient and eventually force PollenWorkers to send
                # through the network the statistics round by round.
                # This is a dictionary {'model_name': (x_train, y_train)}
                data = get_train_data(
                    path_save_stats=self.path_save_stats,
                    cid_samples_dict=cids,
                    worker_ids=[self.map_workers_address_id[w_id] for w_id in worker_ids],
                    # TODO: Reset data for the model given a condition
                    reset=False,
                    )
                # Fit data
                # This is a dictionary {'model_name': (trained_model)}
                try:
                    trained_models[model_name] = get_trained_model(
                        data=data)
                    # Get scores
                    current_scores.append(
                        get_model_score(
                            trained_models[model_name],
                            data))
                except RuntimeError as e:
                    log(ERROR, f'Exception in getting trained model: {e}')
                    trained_models = {}
                    current_scores.append(10)
            except RuntimeError as e:
                log(ERROR, f'Exception in getting data: {e}')
                trained_models = {}
                current_scores.append(10)
        # Check scores
        log(
            DEBUG,'PollenStrategy::poly3 : models\' scores %s',
            current_scores
        )
        # TODO/FIXME: Estimate the threshold better
        if max([abs(score) for score in current_scores]) > 10e-1:
            lists_cids = round_robin_placement(
                sampled_virtual_cids, nodes_dict)
        else:
            # Sorting by batch size (decreasing order)
            # This is a list of tuples (cid, list of samples)
            sampled_virtual_cids = sorted(
                sampled_virtual_cids,
                key=lambda x: x[1]/batch_size,
                reverse=True,
            )
            # Order models from the fastest to the slowest according to the prediction
            # This is a dictionary {'model_name': (trained_model)}
            trained_models = {k: v for k, v in sorted(
                trained_models.items(),
                key=lambda item: predict_single_client(
                    model=item[1],
                    n_samples=sampled_virtual_cids[0][1],
                    batch_size=batch_size,
                ))}
            # Initial assignment
            lists_cids = defaultdict(list)
            for model_name, predictor in trained_models.items():
                for worker in map_models_workers[model_name]:
                    cid, n_samples = sampled_virtual_cids.pop(0)
                    lists_cids[worker].append((
                        cid,
                        n_samples,
                        predict_single_client(
                            model=predictor,
                            n_samples=n_samples,
                            batch_size=batch_size,
                        )
                    ))
            # Placement
            for virtual_cid, num_samples in sampled_virtual_cids:
                # Loop over workers to get the less loaded one
                min_w_id, load = None, None
                for w_id, worker in nodes_dict.items():
                    current_load = sum([c[2]
                                        for c in lists_cids[w_id]])
                    if min_w_id is None:
                        min_w_id, load = w_id, current_load
                    elif current_load < load:
                        min_w_id, load = w_id, current_load

                lists_cids[min_w_id].append((
                    virtual_cid,
                    num_samples,
                    predict_single_client(
                        model=trained_models[map_workers_models[min_w_id]],
                        n_samples=n_samples,
                        batch_size=batch_size,
                    )
                ))
            # Results of the placement
            if verbose:
                placement = {w_id: (sum([x[2] for x in load]), len(load), sum([x[1] for x in load]), [x[0] for x in load]) for w_id, load in lists_cids.items()}
                log(DEBUG,
                    'PollenStrategy :: poly3 placement'
                    ' dict(w_id: (sum_of_ctt, n_clients, sum_of_batches, [cids])) %s',
                    placement)
            return {w_id: [x[0] for x in load] for w_id, load in lists_cids.items()}


def round_robin_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    # Extract cids
    sampled_virtual_cids = np.array(
        [x[0] for x in sampled_virtual_cids])
    # Creates equal clients splits amongst workers
    # (the remainder is assigned to the last worker)
    lists_cids = defaultdict(list)
    splits = np.array_split(sampled_virtual_cids, len(nodes_dict))
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(DEBUG,
            f'Round Robin (RR) placement :: dict(worker_id, [cids]) {placement}')
    return lists_cids


def sorted_round_robin_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Extract cids
    sampled_virtual_cids = np.array(
        [x[0] for x in sampled_virtual_cids])
    # Creates equal clients splits amongst workers by index
    # (the remainder is assigned to the first workers)
    splits = [sampled_virtual_cids[np.arange(
        i, len(sampled_virtual_cids), len(nodes_dict))] for i in range(len(nodes_dict))]
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(DEBUG,
            f'Sorted Round Robin (SRR) placement :: dict(worker_id, [cids]) {placement}')
    return lists_cids


def samples_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[:len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict):]:
        sums = [sum([x[1] for x in list_cids])
                for list_cids in splits]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids])
              for list_cids in splits]
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(DEBUG,
            f'Number of samples placement :: dict(worker_id, [cids]) {placement}')
    return lists_cids


def batches_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[:len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict):]:
        sums = [sum([floor(x[1]/batch_size)
                    for x in list_cids]) for list_cids in splits]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    splits = [np.array([x[0] for x in list_cids])
              for list_cids in splits]
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(DEBUG,
            f'Number of batches placement :: dict(worker_id, [cids]) {placement}')
    return lists_cids


def log_batches_placement(
    sampled_virtual_cids: List[Tuple[int, int]],
    nodes_dict: Dict[str, Node],
    batch_size: int,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, List[int]]:
    # Sorting by size (decreasing order)
    sampled_virtual_cids = sorted(
        sampled_virtual_cids,
        key=lambda x: x[1],
        reverse=True,
    )
    # Assing the first `len(nodes_dict)` clients to the workers
    splits = [[c] for c in sampled_virtual_cids[:len(nodes_dict)]]
    for virtual_cid, num_samples in sampled_virtual_cids[len(nodes_dict):]:
        sums = [sum([log10(floor(x[1]/batch_size))
                    for x in list_cids]) for list_cids in splits]
        min_worker = np.argmin(sums)
        splits[min_worker].append((virtual_cid, num_samples))
    lists_cids = defaultdict(list)
    for worker_dict, split in zip(nodes_dict.items(), splits):
        lists_cids[worker_dict[0]] = split
    if verbose:
        placement = [(k, v) for k, v in lists_cids.items()]
        log(DEBUG,
            f'Logarithm of number of batches placement :: dict(worker_id, [cids]) {placement}')
    return lists_cids


def get_train_data(
    path_save_stats: Path,
    cid_samples_dict: Dict[int, int],
    worker_ids: List[int],
    reset: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    dfs = []
    file_list = [path_save_stats.parent/f'stats_worker{i}' for i in worker_ids]
    for file in file_list:
        df = get_ctt_dataframe_from_pickle(file)
        dfs.append(df)
    df = pd.concat(dfs)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    _, x_train, y_train = merge_ctt_size(
        ctt_df=df,
        clients_df=get_clients_dataframe_from_dict(cid_samples_dict)
    )
    # if reset:
    #     [os.remove(file) in file_list]
    return x_train, y_train


def fn(x, A, B, C, D):
    y = A*x+B*np.log(C*x+1e-8)+D
    return y


def get_trained_model(data: Tuple[np.ndarray, np.ndarray]):
    return curve_fit(fn, data[0].flatten(), data[1])


def get_model_score(model, data):
    parameters, _ = model
    return np.sum((data[1] - fn(data[0].flatten(), *parameters)))


def predict_single_client(model, n_samples: int, batch_size: int):
    parameters, covariance = model
    return fn(n_samples//batch_size, *parameters)
