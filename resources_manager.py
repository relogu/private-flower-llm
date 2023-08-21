from __future__ import annotations

import concurrent.futures
import json
import os
import shlex
import subprocess as sp
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from logging import DEBUG, INFO
from socket import getfqdn
from threading import Thread
from typing import Callable, Dict, Tuple

import nvsmi
import psutil
import torch
import torch.multiprocessing as mp
from flwr.client import ClientLike
from flwr.common import log
from flwr.common.typing import Config, NDArrays, Scalar

NVIDIA_SMI_GET_GPUS = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu --format=csv,noheader,nounits"


def get_workers_map(
    client_fn: Callable[[int, str], ClientLike],
    initial_parameters: NDArrays,
    fit_config: Config,
) -> Dict[int, Tuple[str, int]]:
    # Get information about the resources available
    # TODO/FIXME: Take into account the RAM occupation.
    cpus_available = os.sched_getaffinity(0)
    log(DEBUG, "Node %s: available CPU cores = %s", getfqdn(), cpus_available)
    gpus_available = [g for g in nvsmi.get_gpus()]
    log(DEBUG, "Node %s: available GPUs = {%s}", getfqdn(), gpus_available)
    # Build the resources' map
    map_id_name: Dict[int, str] = {}
    map_name_workers: Dict[str, int] = {}
    workers_map: Dict[int, Tuple[str, int]] = {}
    # fit_config['epochs'] = 1
    for gpu in gpus_available:
        map_id_name[gpu.id] = gpu.name
        if gpu.name not in map_name_workers:
            # # Get the total and maximum memory allocated during the
            # # execution of the `fit` function of the client
            results = monitor_client_execution(
                gpu.id, client_fn, initial_parameters, fit_config
            )
            # # Compute the maximum amount of workers that can be allocated
            # # concurrently on the give GPU, subtracting one for safety
            # # given how eager is Pytorch
            max_processes = (results[0] // results[1]) - 1
            # time.sleep(10)
            # max_processes = 0
            max_processes = monitor_multiple_client_execution(
                gpu.id, client_fn, initial_parameters, fit_config, 0
            )
            # map_name_workers[gpu.name] = int(2*((results[0] // results[1]) - 1))
            # map_name_workers[gpu.name] = int(0.5*max_processes)
            map_name_workers[gpu.name] = int(max_processes)
    # Build the returned map
    for gpu_id, name in map_id_name.items():
        workers_map[gpu_id] = (name, map_name_workers[name])

    # TODO/FIXME: Take into account the number of available CPU cores by
    # removing one for safety, given the additional threads that we might
    # need to run. We need to check how many CPU cores we would need to
    # let free.
    # max_n_workers = min(max_n_workers, len(cpus_available) - 1)
    return workers_map


def monitor_multiple_client_execution(
    gpu_id: int,
    client_fn: Callable[[int, str], ClientLike],
    initial_parameters: NDArrays,
    fit_config: Config,
    concurrency: int,
) -> Tuple[float, float]:
    # log(
    #     DEBUG,
    #     "Testing client on GPU with id {%s}",
    #     gpu_id
    # )
    condition = True
    pace = 0.0
    # p = ThreadPoolExecutor(
    #     # max_workers=len(os.sched_getaffinity(0)),
    #     thread_name_prefix=f"node_{getfqdn()}_concurrency_checker",
    # )
    # if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
    #     torch.multiprocessing.set_start_method('spawn')
    best_concurrency = 0
    while condition:
        p: ProcessPoolExecutor = ProcessPoolExecutor()
        concurrency += 1
        futures_list = []
        monitor = GPUMemoryMonitor(gpu_id=gpu_id)
        monitor.start()
        futures_list.append(p.submit(monitor.start))
        start = time.time()

        for i in range(concurrency):
            fake_client: ClientLike = client_fn(cid=0, device=f"cuda:{gpu_id}")
            futures_list.append(
                p.submit(
                    fake_client.fit,
                    parameters=initial_parameters,
                    config=fit_config,
                )
            )
            # p: mp.Process = mp.Process(target=fake_client.fit, args=(initial_parameters, fit_config))
            # p.start()
            # futures_list.append(p)
        finished_fs, _ = concurrent.futures.wait(
            fs=futures_list[1:],
            timeout=None,
        )
        # for p in futures_list:
        #     p.join()

        current_pace = concurrency / (time.time() - start)
        exceptions = [
            (
                "".join(traceback.format_tb(future.exception().__traceback__)),
                future.exception(),
            )
            if future.exception() is not None
            else None
            for future in finished_fs
        ]
        log(
            DEBUG,
            "Node %s, GPU %s, concurrency %s, exceptions %s",
            getfqdn(),
            gpu_id,
            concurrency,
            exceptions,
        )
        for ex in exceptions:
            if ex is not None:
                condition = False

        log(
            DEBUG,
            "Node %s, GPU %s, concurrency %s, pace (new, old) = (%s, %s)",
            getfqdn(),
            gpu_id,
            concurrency,
            current_pace,
            pace,
        )
        if condition and current_pace > pace:
            pace = current_pace
            best_concurrency = concurrency
        else:
            # condition = False
            log(
                DEBUG,
                "Node %s, GPU %s, concurrency %s, slowed down. Closing the loop.",
                getfqdn(),
                gpu_id,
                concurrency,
            )
        if fit_config["batch_size"] == 4 and concurrency >= 10:
            condition = False

        results = (monitor.total_memory, monitor.maximum_allocated_memory)
        percentual = results[1] / results[0]
        log(
            DEBUG,
            "Node %s, GPU %s, concurrency %s, results {%s, %s}",
            getfqdn(),
            gpu_id,
            concurrency,
            results,
            percentual,
        )
        monitor.do_run = False

        p.shutdown()
    # return concurrency-1
    return best_concurrency


def monitor_client_execution(
    gpu_id: int,
    client_fn: Callable[[int, str], ClientLike],
    initial_parameters: NDArrays,
    fit_config: Config,
) -> Tuple[float, float]:
    # log(
    #     DEBUG,
    #     "Testing client on GPU with id {%s}",
    #     gpu_id
    # )
    if torch.multiprocessing.get_start_method(allow_none=True) != "spawn":
        torch.multiprocessing.set_start_method("spawn")
    futures_list = []
    p = ThreadPoolExecutor(
        # max_workers=len(os.sched_getaffinity(0)),
        thread_name_prefix=f"node_{getfqdn()}_concurrency_checker",
    )
    monitor = GPUMemoryMonitor(gpu_id=gpu_id)
    # futures_list.append(p.submit(monitor.start))
    monitor.start()
    fake_client: ClientLike = client_fn(cid=0, device=f"cuda:{gpu_id}")
    # futures_list.append(
    #     p.submit(
    #         fake_client.fit,
    #         parameters=initial_parameters,
    #         config=fit_config,))
    p: mp.Process = mp.Process(
        target=fake_client.fit, args=(initial_parameters, fit_config)
    )
    p.start()
    p.join()
    # fake_client.fit(
    #     parameters=initial_parameters,
    #     config=fit_config,
    # )
    # finished_fs, _ = concurrent.futures.wait(
    #     fs=futures_list[1:],
    #     timeout=None,
    # )
    # TODO/FIXME: It seems necessary to wait one second here to let
    # the monitor observe the real value of the memory.
    # time.sleep(1)
    results = (monitor.total_memory, monitor.maximum_allocated_memory)
    monitor.do_run = False
    # p.shutdown()
    log(DEBUG, "Node %s, GPU %s, results {%s}", getfqdn(), gpu_id, results)
    return results


def get_cuda_prop() -> Dict[str, Device]:
    gpus_prop = {}
    gpus_available = [g for g in nvsmi.get_gpus()]
    for gpu in gpus_available:
        if f"cuda:{gpu.id}" not in gpus_prop:
            monitor = ResourcesMonitor(gpu_id=int(gpu.id))
            monitor.start()
            # TODO: Get initial statistics from the resources monitor
            # TODO: Launch a fake client to assess the concurrency
            # TODO: Get the latest statistics from the resources monitor
            # TODO: Estimate the maximum number of concurrent workers
            time.sleep(1)
            current_concurrency = 10
            gpus_prop[f"cuda:{gpu.id}"] = Device(
                id=gpu.id,
                name=gpu.name,
                type="cuda",
                total_memory=gpu.mem_total,
                allocated_memory=gpu.mem_used,
                concurrency=current_concurrency,
            )
            monitor.do_run = False
    return gpus_prop


def get_cpu_prop(cpu_type: str) -> Dict[str, Device]:
    monitor = ResourcesMonitor(gpu_id=-1)
    monitor.start()
    # TODO: Get initial statistics from the resources monitor
    # TODO: Launch a fake client to assess the concurrency
    # TODO: Get the latest statistics from the resources monitor
    # TODO: Estimate the maximum number of concurrent workers
    time.sleep(1)
    current_concurrency = 10
    cpu_prop = {
        f"{cpu_type}:0": Device(
            id=0,
            name=f"{cpu_type}:0",
            type=f"{cpu_type}",
            total_memory=psutil.virtual_memory().total,
            allocated_memory=psutil.virtual_memory().total
            - psutil.virtual_memory().total,
            concurrency=current_concurrency,
        )
    }
    return cpu_prop


def get_node_manager_properties(config: Config) -> Dict[str, Scalar]:
    # Get hardware accelerator properties
    if torch.cuda.is_available():
        log(INFO, f"Node {getfqdn()}, CUDA acceleration available.")
        device_info = get_cuda_prop()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        log(INFO, f"Node {getfqdn()}, MPS acceleration available.")
        device_info = get_cpu_prop("mps")
    else:
        log(
            INFO,
            f"Node {getfqdn()}, No hardware accelerator available. Assessing CPU execution.",
        )
        device_info = get_cpu_prop("cpu")
    # Get general node properties
    node = Node(
        name=getfqdn(),
        cpu_num=len(os.sched_getaffinity(0)),
        cpu_ram_total=psutil.virtual_memory().total,
        cpu_ram_available=psutil.virtual_memory().total - psutil.virtual_memory().used,
        device_info=device_info,
    )
    log(DEBUG, f"Node {getfqdn()} has complete properties {node}")

    return {"node": str(node)}


@dataclass
class Device:
    """Device info."""

    id: int
    name: str
    type: str
    total_memory: float
    allocated_memory: float
    concurrency: int

    def __init__(
        self,
        id: int,
        name: str,
        type: str,
        total_memory: float,
        allocated_memory: float,
        concurrency: int,
    ):
        self.id = id
        self.name = name
        self.type = type
        self.total_memory = total_memory
        self.allocated_memory = allocated_memory
        self.concurrency = concurrency

    def __repr__(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_str(d: str) -> Device:
        """Create a Device object from a string (built with str(Device))."""
        d = json.loads(d)
        return Device(
            id=d["id"],
            name=d["name"],
            type=d["type"],
            total_memory=d["total_memory"],
            allocated_memory=d["allocated_memory"],
            concurrency=d["concurrency"],
        )


@dataclass
class Node:
    """Node info."""

    name: str
    cpu_num: int
    cpu_ram_total: int
    cpu_ram_available: int
    device_info: Dict[str, Device]

    def __init__(
        self,
        name: str,
        cpu_num: int,
        cpu_ram_total: int,
        cpu_ram_available: int,
        device_info: Dict[str, Device],
    ):
        self.name = name
        self.cpu_num = cpu_num
        self.cpu_ram_total = cpu_ram_total
        self.cpu_ram_available = cpu_ram_available
        self.device_info = device_info

    def __repr__(self):
        return json.dumps(asdict(self))

    @staticmethod
    def from_str(d: str) -> Node:
        """Create a Node from a string (built with str(Node))."""
        d = json.loads(d)
        return Node(
            name=d["name"],
            cpu_num=d["cpu_num"],
            cpu_ram_total=d["cpu_ram_total"],
            cpu_ram_available=d["cpu_ram_available"],
            device_info={
                k: Device.from_str(str(v).replace("'", '"'))
                for k, v in d["device_info"].items()
            },
        )


class ResourcesMonitor(Thread):
    def __init__(
        self,
        gpu_id: int,
        frequency: float = 0.1,
    ) -> None:
        Thread.__init__(self)
        self.frequency = frequency
        self.gpu_id = gpu_id
        self.vram_total_memory = 0.0
        self.vram_maximum_allocated_memory = 0.0
        self.cpu_ram_total = 0.0
        self.cpu_ram_available = 0.0
        self.do_run = True

    def _get_gpu_memory(self) -> Tuple[float, float]:
        """This function reads the output of `nvidia-smi --query`
        launched as a subprocess. The GPU is selected by `self.gpu_id`.
        In particular, it reads the total and allocated memory in MB.

        Raises:
            RuntimeError: if raised by the subprocess launched.

        Returns:
            Tuple[float, float]: the total and allocated memory in MB.
        """
        output_to_list = lambda x: x.decode("ascii").split("\n")
        command = NVIDIA_SMI_GET_GPUS + f" -i {self.gpu_id}"
        try:
            memory_use_info = output_to_list(
                sp.check_output(shlex.split(command), stderr=sp.STDOUT)
            )
        except sp.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        log(
            DEBUG,
            "ResourcesMonitor.get_gpu_memory: memory_use_info=%s",
            memory_use_info,
        )
        # index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu
        return float(str(memory_use_info[0]).split(",")[3]), float(
            str(memory_use_info[0]).split(",")[4]
        )

    def _update_max_values(self):
        """
        This function calls itself every `self.frequency` secs and
        updates the maximum values for `self.vram_total_memory` and
        `self.vram_maximum_allocated_memory`.
        """
        while self.do_run:
            if self.gpu_id >= 0:
                mem = self._get_gpu_memory()
                self.vram_total_memory = max(self.vram_total_memory, mem[0])
                self.vram_maximum_allocated_memory = max(
                    self.vram_maximum_allocated_memory, mem[1]
                )
            self.cpu_ram_total = psutil.virtual_memory().total
            self.cpu_ram_available = (
                psutil.virtual_memory().total - psutil.virtual_memory().used
            )
            log(
                DEBUG,
                "ResourcesMonitor._update_max_values: "
                "mem=%s, vram_total_memory=%s, vram_maximum_allocated_memory=%s, cpu_ram_total=%s, cpu_ram_available=%s",
                mem,
                self.vram_total_memory,
                self.vram_maximum_allocated_memory,
                self.cpu_ram_total,
                self.cpu_ram_available,
            )
            time.sleep(self.frequency)

    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run() method
        invokes the callable object passed to the object's constructor as the
        target argument, if any, with sequential and keyword arguments taken
        from the args and kwargs arguments, respectively.

        """
        try:
            self._update_max_values()
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


if __name__ == "__main__":
    node = get_node_manager_properties({})
    log(INFO, f"NodeManager's properties are: {node}")
    node = Node.from_str(str(node['node']))
    log(INFO, f"Converted to Node object {node}")
