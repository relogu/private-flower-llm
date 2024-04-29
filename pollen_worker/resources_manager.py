"""Resources manager for the Pollen worker.

Handles both metric collection and GPU/CPU resources allocation to workers.
"""

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
from typing import Any, cast

import nvsmi
import psutil
import pyarrow as pa
import pynvml
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar, log
from pyarrow import csv

NVIDIA_SMI_GET_GPUS_ALL = (
    "nvidia-smi"
    " --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,"
    "driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu,"
    "power.draw,clocks.sm,clocks.mem,clocks.gr,timestamp"
    " --format=csv,noheader,nounits"
)
NVIDIA_SMI_GET_GPUS_STATS = (
    "nvidia-smi"
    " --query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,"
    "temperature.gpu,power.draw,clocks.sm,clocks.mem,clocks.gr,timestamp"
    " --format=csv,nounits"
)
NVIDIA_SMI_GET_GPUS_MEMORY_ONLY = (
    "nvidia-smi --query-gpu"
    "=memory.total,memory.used,memory.free --format=csv,noheader,nounits"
)


@dataclass
class Device:
    """Device info."""

    device_id: int
    name: str
    device_type: str
    total_memory: float
    allocated_memory: float
    concurrency: int

    def __init__(
        self,
        device_id: int,
        name: str,
        device_type: str,
        total_memory: float,
        allocated_memory: float,
        concurrency: int,
    ) -> None:
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.total_memory = total_memory
        self.allocated_memory = allocated_memory
        self.concurrency = concurrency

    def __repr__(self) -> str:
        """Return the string representation."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_str(d: str) -> Device:
        """Create a Device object from a string (built with str(Device))."""
        device_dict: dict = json.loads(d)
        return Device(
            device_id=int(device_dict["device_id"]),
            name=device_dict["name"],
            device_type=device_dict["device_type"],
            total_memory=float(device_dict["total_memory"]),
            allocated_memory=float(device_dict["allocated_memory"]),
            concurrency=int(device_dict["concurrency"]),
        )


def merge_devices(devices: list[Device]) -> Device:
    """Merge multiple devices into a single one."""
    assert len(devices) > 0
    if len(devices) == 1:
        return devices[0]
    else:
        return Device(
            device_id=0,
            name="gpu-merged",
            device_type="gpu-merged",
            total_memory=sum([d.total_memory for d in devices]),
            allocated_memory=sum([d.allocated_memory for d in devices]),
            concurrency=1,
        )


def get_gpu_prop(merge: bool = False) -> dict[str, Device]:
    """Return the properties of the GPU in the node w/o assessing anything."""
    # Init return value
    gpus_prop: dict[str, Device] = {}
    # NOTE: This is for controlling the GPU memory allocation
    pynvml.nvmlInit()
    # Loop over GPU devices
    for dev_id in range(pynvml.nvmlDeviceGetCount()):
        # Get the current device's handle
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_id)
        # Get memory info
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus_prop[f"cuda:{dev_id}"] = Device(
            device_id=dev_id,
            name=pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
            device_type="cuda",
            total_memory=mem.total,
            allocated_memory=mem.used,
            # NOTE: Forcing concurrency to one
            concurrency=1,
        )
        # Loop over all running process on the current device
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            log(
                INFO,
                "GPU %s is running process %s that allocates %s bytes.",
                dev_id,
                proc.pid,
                proc.usedGpuMemory,
            )
    # Shutdown pynvml
    pynvml.nvmlShutdown()
    # If `merge`, the worker runs over multiple GPUs
    if merge:
        gpus_prop = {"gpu-merged": merge_devices(list(gpus_prop.values()))}
    return gpus_prop


def get_cuda_prop(
    client: NumPyClient, params: NDArrays, config: dict[str, Scalar]
) -> dict[str, Device]:
    """Assesses the capabilities of the CUDA resources available."""
    gpus_prop = {}
    # NOTE: This is for controlling the GPU memory allocation
    pynvml.nvmlInit()
    # NOTE: This is necessary, otherwise it throws an error: https://github.com/pytorch/pytorch/issues/40403
    # NOTE: This also solves the issue of the first round not using all the workers.
    torch.multiprocessing.set_start_method("spawn", force=True)
    p = ProcessPoolExecutor()
    clients = []
    gpus_available = list(nvsmi.get_gpus())
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
    for gpu_name, (gpu, proc_used, total, used, _free) in monitors.items():
        # NOTE: This accounts for other (external) processes running on the same GPU
        current_concurrency = int((total - used + proc_used) // proc_used)
        gpus_prop[gpu_name] = Device(
            device_id=gpu.id,
            name=gpu.name,
            device_type="cuda",
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
    config: dict[str, Scalar],
) -> dict[str, Device]:
    """Assesses the capabilities of the CPU resources available."""
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
    current_concurrency = int(monitor.cpu_ram_available // sum(monitor.pid_ram_used))
    cpu_prop = {
        f"{cpu_type}:0": Device(
            device_id=0,
            name=f"{cpu_type}:0",
            device_type=f"{cpu_type}",
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
class Node:
    """Node info."""

    name: str = ""
    cpu_num: int = 0
    cpu_ram_total: int = 0
    cpu_ram_available: int = 0
    device_info: dict[str, Device] | None = None

    def __repr__(self) -> str:
        """Return the string representation."""
        return json.dumps(asdict(self))

    @staticmethod
    def from_str(d: str) -> Node:
        """Create a Node from a string (built with str(Node))."""
        dict_json: dict[str, Any] = json.loads(d)
        return Node(
            name=dict_json["name"],
            cpu_num=int(dict_json["cpu_num"]),
            cpu_ram_total=int(dict_json["cpu_ram_total"]),
            cpu_ram_available=int(dict_json["cpu_ram_available"]),
            device_info={
                str(k): Device.from_str(str(v).replace("'", '"'))
                for k, v in cast(dict[str, str], dict(dict_json["device_info"])).items()
            },
        )


class ResourcesMonitor(Thread):
    """Simple resources monitor for CPU and GPU."""

    def __init__(
        self,
        gpu_id: int,
        list_pids: list[int] | None = None,
        frequency: float = 0.1,
    ) -> None:
        Thread.__init__(self)
        self.frequency = frequency
        self.gpu_id = gpu_id
        self.list_pids = list_pids if list_pids is not None else []
        self.vram_total_memory = 0.0
        self.vram_maximum_allocated_memory = 0.0
        self.cpu_ram_total = 0.0
        self.cpu_ram_available = 0.0
        self.do_run = True
        self.pid_ram_used: list[int] = []
        self.dead = False

    def _get_gpu_memory(self) -> tuple[float, float]:
        """Read the output of `nvidia-smi --query` launched as a subprocess.

        The GPU is selected by `self.gpu_id`. In particular, it reads the
        total and allocated memory in MB.

        Raises
        ------
            RuntimeError: if raised by the subprocess launched.

        Returns
        -------
            Tuple[float, float]: the total and allocated memory in MB.
        """

        def output_to_list(x: bytes) -> list[str]:
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
                f"command '{e.cmd}' return with error (code {e.returncode}): {e.output}"
            ) from e
        ret_val = (0.0, 0.0)
        try:
            ret_val = (
                float(current_gpu_stats.split(",")[0]),
                float(current_gpu_stats.split(",")[1]),
            )
        except Exception as e:
            log(
                DEBUG,
                "ResourcesMonitor.get_gpu_memory: error=%s, ret_val=%s. Retrying...",
                e,
                current_gpu_stats,
            )
            ret_val = self._get_gpu_memory()
        return ret_val

    def _update_max_values(self) -> None:
        """Call itself every `self.frequency` secs.

        Updates the maximum values for `self.vram_total_memory` and
        `self.vram_maximum_allocated_memory`.
        """
        while self.do_run:
            mem = (0.0, 0.0)
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
            time.sleep(self.frequency)

    def run(self) -> None:
        """Represent the thread's activity.

        You may override this method in a subclass. The standard run() method invokes
        the callable object passed to the object's constructor as the target argument,
        if any, with sequential and keyword arguments taken from the args and kwargs
        arguments, respectively.
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

            del self._target, self._args, self._kwargs  # type: ignore[attr-defined]


class DaemonResourcesMonitor(Thread):
    """Simple resources monitor for GPU."""

    def __init__(
        self,
        gpu_ids: list[int],
        frequency: float = 0.1,
    ) -> None:
        Thread.__init__(self)
        self.frequency = frequency
        self.gpu_ids = gpu_ids
        self.do_run = True
        self.gpu_stats: list[pa.Table] = []

    def _get_gpu_stats(self) -> None:
        def output_to_list(x: bytes) -> bytes:
            return bytes(x)

        command = (
            NVIDIA_SMI_GET_GPUS_STATS
            + f" -i {','.join([str(i) for i in self.gpu_ids])}"
        )
        try:
            current_gpu_stats = output_to_list(
                sp.check_output(shlex.split(command), timeout=3)
            )
            self.gpu_stats.append(csv.read_csv(io.BytesIO(current_gpu_stats)))
        except sp.CalledProcessError as e:
            raise RuntimeError(
                f"DaemonResourcesMonitor: command '{e.cmd}' "
                f"return with error (code {e.returncode}): {e.output}"
            ) from e

    def _update_max_values(self) -> None:
        while self.do_run:
            self._get_gpu_stats()
            time.sleep(self.frequency)

    def run(self) -> None:
        """Represent the thread's activity.

        You may override this method in a subclass. The standard run() method invokes
        the callable object passed to the object's constructor as the target argument,
        if any, with sequential and keyword arguments taken from the args and kwargs
        arguments, respectively.
        """
        try:
            self._update_max_values()
            log(
                DEBUG,
                "DaemonResourcesMonitor.run: dying",
            )
        finally:
            del self._target, self._args, self._kwargs  # type: ignore[attr-defined]


if __name__ == "__main__":
    node = Node(
        name="node1",
        cpu_num=psutil.cpu_count(),
        cpu_ram_total=psutil.virtual_memory().total,
        cpu_ram_available=psutil.virtual_memory().total - psutil.virtual_memory().used,
        device_info={},
    )
    log(INFO, f"NodeManager's properties are: {node}")
    node = Node.from_str(str(node))
    log(INFO, f"Converted to Node object {node}")
    assert node.device_info is not None
    log(INFO, f"Node {node.name} has {len(node.device_info)} acceleration devices.")
