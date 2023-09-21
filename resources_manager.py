from __future__ import annotations

import io
import json
import os
import shlex
import subprocess as sp
import time
from concurrent.futures import Future, ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from logging import DEBUG, INFO
from threading import Thread
from typing import Dict, List, Tuple

import nvsmi
import psutil
import pynvml
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar, log
from pyarrow import csv

NVIDIA_SMI_GET_GPUS_ALL = "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.gr,timestamp --format=csv,noheader,nounits"
NVIDIA_SMI_GET_GPUS_STATS = "nvidia-smi --query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.gr,timestamp --format=csv,nounits"
NVIDIA_SMI_GET_GPUS_MEMORY_ONLY = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader,nounits"


def get_cuda_prop(
    client: NumPyClient, params: NDArrays, config: Dict[str, Scalar]
) -> Dict[str, Device]:
    gpus_prop = {}
    # NOTE: This is for controlling the GPU memory allocation
    pynvml.nvmlInit()
    # NOTE: This is necessary, otherwise it throws an error: https://github.com/pytorch/pytorch/issues/40403
    # NOTE: This also solves the issue of the first round not using all the workers.
    torch.multiprocessing.set_start_method("spawn", force=True)
    p = ProcessPoolExecutor()
    clients = []
    gpus_available = [g for g in nvsmi.get_gpus()]
    for gpu in gpus_available:
        if f"cuda:{gpu.id}" not in gpus_prop:
            config["device"] = f"cuda:{gpu.id}"
            log(INFO, f"Collecting training statistics for GPU {gpu.id}.")
            clients.append(
                p.submit(client.fit, parameters=params, config=deepcopy(config))
            )
            gpus_prop[f"cuda:{gpu.id}"] = gpu
    for c in clients:
        c.result()
    monitors = {}
    for pid in list(p._processes.keys()):
        for dev_id in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
            for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if pid == proc.pid:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    current_gpu = gpus_prop.get(f"cuda:{dev_id}", None)
                    monitors[f"cuda:{dev_id}"] = (
                        current_gpu,
                        proc.usedGpuMemory,
                        mem.total,
                        mem.used,
                        mem.free,
                    )
    p.shutdown(wait=False)
    for gpu_name, (gpu, proc_used, total, used, free) in monitors.items():
        # NOTE: This accounts for other (external) processes running on the same GPU
        current_concurrency = int((total - used + proc_used) // proc_used)
        gpus_prop[gpu_name] = Device(
            id=gpu.id,
            name=gpu.name,
            type="cuda",
            total_memory=gpu.mem_total,
            allocated_memory=gpu.mem_used,
            concurrency=current_concurrency,
        )
    # Shutdown pynvml
    pynvml.nvmlShutdown()
    return gpus_prop


def get_cpu_prop(
    cpu_type: str,
    client: NumPyClient,
    params: NDArrays,
    config: Dict[str, Scalar],
) -> Dict[str, Device]:
    monitor = ResourcesMonitor(gpu_id=-1, list_pids=[os.getpid()])
    monitor.start()
    time.sleep(1)
    log(INFO, f"Collecting training statistics for CPU {cpu_type}.")
    # NOTE: This is necessary, otherwise it throws an error: https://github.com/pytorch/pytorch/issues/40403
    # NOTE: This also solves the issue of the first round not using all the workers.
    torch.multiprocessing.set_start_method("spawn", force=True)
    p = ProcessPoolExecutor()
    future: Future = p.submit(client.fit, parameters=params, config=config)
    future.result()
    p.shutdown(wait=False)
    current_concurrency = monitor.cpu_ram_available // sum(monitor.pid_ram_used)
    cpu_prop = {
        f"{cpu_type}:0": Device(
            id=0,
            name=f"{cpu_type}:0",
            type=f"{cpu_type}",
            total_memory=psutil.virtual_memory().total,
            allocated_memory=psutil.virtual_memory().total
            - psutil.virtual_memory().used,
            concurrency=current_concurrency,
        )
    }
    # Close monitor
    while monitor.is_alive():
        monitor.do_run = False
        time.sleep(0.1)
    del monitor
    return cpu_prop


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
    def from_str(d: str):
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
        list_pids: List[int] = [],
        frequency: float = 0.1,
    ) -> None:
        Thread.__init__(self)
        self.frequency = frequency
        self.gpu_id = gpu_id
        self.list_pids = list_pids
        self.vram_total_memory = 0.0
        self.vram_maximum_allocated_memory = 0.0
        self.cpu_ram_total = 0.0
        self.cpu_ram_available = 0.0
        self.do_run = True
        self.pid_ram_used = []
        self.dead = False

    def _get_gpu_memory(self) -> Tuple[float, float]:
        """This function reads the output of `nvidia-smi --query` launched as a
        subprocess. The GPU is selected by `self.gpu_id`. In particular, it
        reads the total and allocated memory in MB.

        Raises:
            RuntimeError: if raised by the subprocess launched.

        Returns:
            Tuple[float, float]: the total and allocated memory in MB.
        """

        def output_to_list(x):
            return x.decode("ascii").split("\n")

        command = NVIDIA_SMI_GET_GPUS_MEMORY_ONLY + f" -i {self.gpu_id}"
        try:
            current_gpu_stats = output_to_list(
                sp.check_output(shlex.split(command), timeout=3)
            )[
                0
            ]  # [0] is the first line of the output, the second line is always empty
        except sp.CalledProcessError as e:
            raise RuntimeError(
                "command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        # log(
        #     DEBUG,
        #     "ResourcesMonitor.get_gpu_memory: current_gpu_stats=%s, splitted_current_gpu_stats=%s",
        #     current_gpu_stats,
        #     current_gpu_stats.split(","),
        # )
        # NOTE: the ouput has the following values -- memory.total,memory.used,memory.free
        ret_val = (0.0, 0.0)
        try:
            ret_val = float(current_gpu_stats.split(",")[0]), float(
                current_gpu_stats.split(",")[1]
            )
        except:
            log(
                DEBUG,
                "ResourcesMonitor.get_gpu_memory: error=%s retrying",
                current_gpu_stats,
                # ret_val
            )
            ret_val = self._get_gpu_memory()
        return ret_val

    def _update_max_values(self):
        """This function calls itself every `self.frequency` secs and updates
        the maximum values for `self.vram_total_memory` and
        `self.vram_maximum_allocated_memory`."""
        while self.do_run:
            mem = 0.0
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
            self.pid_ram_used = [
                psutil.Process(pid).memory_info().vms for pid in self.list_pids
            ]
            # log(
            #     DEBUG,
            #     "ResourcesMonitor._update_max_values: "
            #     "mem=%s, vram_total_memory=%s, vram_maximum_allocated_memory=%s, cpu_ram_total=%s, cpu_ram_available=%s",
            #     mem,
            #     self.vram_total_memory,
            #     self.vram_maximum_allocated_memory,
            #     self.cpu_ram_total,
            #     self.cpu_ram_available,
            # )
            time.sleep(self.frequency)

    def run(self):
        """Method representing the thread's activity.

        You may override this method in a subclass. The standard run()
        method invokes the callable object passed to the object's
        constructor as the target argument, if any, with sequential and
        keyword arguments taken from the args and kwargs arguments,
        respectively.
        """
        try:
            self._update_max_values()
            log(
                DEBUG,
                "ResourcesMonitor.run: dying",
            )
            self.dead = True
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs


class DaemonResourcesMonitor(Thread):
    def __init__(
        self,
        gpu_ids: List[int],
        frequency: float = 0.1,
    ) -> None:
        Thread.__init__(self)
        self.frequency = frequency
        self.gpu_ids = gpu_ids
        self.do_run = True
        self.gpu_stats = []

    def _get_gpu_stats(self):
        def output_to_list(x):
            return bytes(x)

        command = NVIDIA_SMI_GET_GPUS_STATS + f" -i {','.join(self.gpu_ids)}"
        try:
            current_gpu_stats = output_to_list(
                sp.check_output(shlex.split(command), timeout=3)
            )
        except sp.CalledProcessError as e:
            raise RuntimeError(
                "DaemonResourcesMonitor: command '{}' return with error (code {}): {}".format(
                    e.cmd, e.returncode, e.output
                )
            )
        # NOTE: the ouput has the following values -- index,uuid,**utilization.gpu,memory.total,memory.used,memory.free**,driver_version,name,gpu_serial,display_active,display_mode,**temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.gr**
        self.gpu_stats.append(csv.read_csv(io.BytesIO(current_gpu_stats)))

    def _update_max_values(self):
        while self.do_run:
            self._get_gpu_stats()
            time.sleep(self.frequency)

    def run(self):
        try:
            self._update_max_values()
            log(
                DEBUG,
                "DaemonResourcesMonitor.run: dying",
            )
        finally:
            del self._target, self._args, self._kwargs


if __name__ == "__main__":
    node = Node(
        name="node1",
        cpu_num=psutil.cpu_count(),
        cpu_ram_total=psutil.virtual_memory().total,
        cpu_ram_available=psutil.virtual_memory().total - psutil.virtual_memory().used,
        device_info={},
    )
    log(INFO, f"NodeManager's properties are: {node}")
    node = Node.from_str(str(node["node"]))
    log(INFO, f"Converted to Node object {node}")
    log(INFO, f"Node {node.name} has {len(node.device_info)} acceleration devices.")
