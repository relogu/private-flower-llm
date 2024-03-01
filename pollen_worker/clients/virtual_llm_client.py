"""Lightweight Flower Client for Pollen, capable of sustaining LLm training.

Clients are trained by workers which are managed by node managers. This type of client
avoids any memory or processing intensive operations in the _init_ function. As such,
virtual clients can be used to simulate a large number of clients on a single machine
even if many are spawned at once.
"""

import copy
import os
import time
from collections.abc import Callable
from logging import DEBUG, INFO, WARNING
from typing import Any

import flwr as fl
import hydra
import psutil
import streaming
import transformers
from anyio import Path
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from omegaconf import DictConfig, OmegaConf

from pollen_worker.clients.llm_client_functions import (
    get_parameters,
    get_raw_model_parameters,
    llm_eval,
    llm_fit,
    set_all_data_paths,
)
from pollen_worker.utils import (
    get_file_names_from_file_number,
    get_open_fds,
    get_referenced_tensors_summary,
    get_selected_objects_types,
)


class VirtualLLMClient(fl.client.NumPyClient):
    """Implement the most lightweight Flower Client."""

    def __init__(
        self,
        *,
        cid: int | str,
        cfg: DictConfig,
    ) -> None:
        # Set init parameters
        self.cid = cid
        self.cfg = cfg
        # Set the save folder specifically for this client and this run
        if self.cfg.save_folder is not None:  # type: ignore[union-attr]
            self.cfg.save_folder = (  # type: ignore[union-attr]
                self.cfg.save_folder
                + f"/{self.cfg.run_name}"
                + "/client_"  # type: ignore[union-attr]
                + str(self.cid)  # type: ignore[union-attr]
            )
            try:
                local_path = Path(
                    str(self.cfg.save_folder).replace(  # type: ignore[union-attr]
                        "s3://checkpoints/", ""
                    )
                )
                log(INFO, "Looking for a checkpoint to load in %s", local_path)
                # NOTE: The suggested `Path.exists(local_path)` doens't work with a
                # direct substitution. This necessitates a fix.
                if os.path.exists(local_path):  # noqa: PTH110
                    self.cfg.load_path = self.cfg.save_folder + "/latest-rank{rank}.pt"
                    log(INFO, "Set checkpoint to load: %s", self.cfg.load_path)
            except Exception as e:
                log(WARNING, "The `load_path` wasn't set.", exc_info=e)
                # log(
                #     DEBUG,
                #     "Error running `os.listdir` for folder %s",
                #     self.cfg.save_folder,
                #     exc_info=e,
                #     stack_info=True,
                # )
        # Set the wandb run name
        if self.cfg.loggers.wandb is not None:
            # Get the server run name
            run_name = self.cfg.loggers.wandb.init_kwargs.name
            # Add the client id to the run name
            new_run_name = run_name + f"_client_{self.cid}"
            server_id = self.cfg.loggers.wandb.init_kwargs.id
            self.cfg.loggers.wandb.init_kwargs.id = server_id + f"_client_{self.cid}"
            # Set the new run name
            self.cfg.loggers.wandb.init_kwargs.name = new_run_name

        transformers.logging.set_verbosity_error()

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"VirtualLLMClient(cid={self.cid})"

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return {}

    def get_parameters(
        self,
        config: Config,
    ) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """
        cfg: DictConfig = copy.deepcopy(self.cfg)
        return get_parameters(config, cfg)

    def fit(
        self, parameters: NDArrays, config: dict
    ) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any]]:
        """Implement the fit step."""
        # log(INFO, f'VirtualLLMClient.fit :: {config}')
        cfg: DictConfig = copy.deepcopy(self.cfg)
        # Set the appropriate path given the `client_id`
        if cfg.data_remote is not None:  # type: ignore[union-attr]
            # Set the appropriate path given the `client_id`
            new_remote_path = (
                str(cfg.data_remote) + f"/client_{self.cid}"  # type: ignore[union-attr]
            )
            cfg = set_all_data_paths(cfg, new_remote_path, False)
        # Tie the local path to the client_id and the run_uuid
        new_local_path = (
            str(cfg.data_local) + f"/client_{self.cid}"  # type: ignore[union-attr]
        )
        cfg = set_all_data_paths(cfg, new_local_path)
        # Execute the fit function
        return llm_fit(parameters, config, cfg)

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Implement the evaluation step."""
        # log(INFO, f'VirtualLLMClient.evaluate :: {config}')
        cfg: DictConfig = copy.deepcopy(self.cfg)
        # Set the appropriate path for the (centralised) val set
        if cfg.data_remote is not None:  # type: ignore[union-attr]
            # Extracts the parent folder from the remote path
            new_remote_path = "s3:/" + str(
                Path(
                    str(cfg.data_remote).replace("s3:/", "")  # type: ignore[union-attr]
                ).parent  # type: ignore[union-attr]
            )
            cfg = set_all_data_paths(cfg, new_remote_path, False)
        # Tie the local path to the client_id and the run_uuid
        new_local_path = str(cfg.data_local) + "/val"  # type: ignore[union-attr]
        cfg = set_all_data_paths(cfg, new_local_path)
        return llm_eval(parameters, config, cfg)


def gen_client_fn(
    cfg: DictConfig,
    **kwargs: dict,
) -> Callable[[int], VirtualLLMClient]:
    """Return generic `client_fn` for Flower Framework."""

    def client_fn(client_id: int) -> VirtualLLMClient:
        client = VirtualLLMClient(
            cid=client_id,
            cfg=copy.deepcopy(cfg),
        )
        return client

    return client_fn


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Test the VirtualLLMClient."""
    time.time()
    log(
        INFO,
        "VirtualLLMClient received the following config:\n%s",
        OmegaConf.to_yaml(cfg, resolve=True),
    )
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    log(
        INFO,
        "VirtualLLMClient received the llm_config:\n%s",
        OmegaConf.to_yaml(_llm_config, resolve=True),
    )
    assert isinstance(_llm_config, DictConfig)
    # Set `max_duration` to a low value for testing
    _llm_config.max_duration = "10ba"  # type: ignore[union-attr]
    _llm_config.local_steps = "10ba"  # type: ignore[union-attr]
    log(
        INFO,
        "VirtualLLMClient received the following config:\n%s",
        OmegaConf.to_yaml(cfg, resolve=True),
    )
    # Get the client generator function
    client_fn = gen_client_fn(
        cfg=copy.deepcopy(_llm_config),
    )
    # Looping over two clients
    # Cleaning stale shared memory

    streaming.base.util.clean_stale_shared_memory()
    # for i in range(2):
    for i in range(1):
        # Get initial model parameters
        parameters = get_raw_model_parameters(copy.deepcopy(_llm_config))
        log(
            INFO, f"get_raw_model_parameters :: parameters' length is {len(parameters)}"
        )
        # Create a virtual client
        virtual_llm_client = client_fn(i)
        # Test virtual client's get_properties function
        properties = virtual_llm_client.get_properties(config={})
        log(INFO, f"VirtualLLMClient.get_properties :: properties={properties}")
        # Test virtual client's get_parameters function
        parameters = virtual_llm_client.get_parameters(config={})
        log(
            INFO,
            "VirtualLLMClient.get_parameters :: parameters' length is %s",
            len(parameters),
        )
        # Test virtual client's fit function
        parameters, num_examples, metrics = virtual_llm_client.fit(
            parameters=parameters, config={}
        )
        log(INFO, f"VirtualLLMClient.fit :: parameters' length is {len(parameters)}")
        log(
            INFO,
            "VirtualLLMClient.fit :: number of example trained is %s.",
            num_examples,
        )
        log(INFO, f"VirtualLLMClient.fit :: train metrics={metrics}")
        # PROFILING
        all_opened = 0
        sum_of_ram = 0
        for proc in psutil.process_iter():
            try:
                n_opened_files = len(proc.open_files())
                ram_occupied = proc.memory_info().rss
                # log(
                #     DEBUG, "Process %s with PID %s has %d opened files."
                #     "RAM occupied: %d bytes.",
                #     proc.name(), proc.pid, n_opened_files, ram_occupied,
                # )
                all_opened += n_opened_files
                sum_of_ram += ram_occupied
            except Exception:
                # log(
                #     DEBUG,
                #     "Error running `len(proc.open_files())` for process %s",
                #     proc.name(),
                #     exc_info=e,
                #     stack_info=True,
                # )
                pass
        log(
            DEBUG,
            "All processes have %d opened files and %d file descriptors. "
            "RAM occupied: %d bytes.",
            all_opened,
            len(get_open_fds()),
            sum_of_ram,
        )
        log(
            DEBUG,
            "Opened file descriptors: %s",
            "\n".join(get_file_names_from_file_number(get_open_fds())),
        )
        proc = psutil.Process()
        n_opened_files = len(proc.open_files())
        log(
            DEBUG,
            "Current process %s has %d opened files. RAM occupied: %d bytes.",
            proc.name(),
            n_opened_files,
            proc.memory_info().rss,
        )

        get_referenced_tensors_summary(cuda_only=False)

        get_selected_objects_types(
            selection=[
                "torch",
                "streaming",
                "multiprocessing",
                "composer",
                "subprocess",
                "llmfoundry",
            ],
            second_selection=[
                "streaming.base.compression",
                "streaming.base.shared.memory.SharedMemory",
                "multiprocessing.context.Process",
            ],
        )
        # Test virtual client's evaluate function
        loss, num_examples, metrics = virtual_llm_client.evaluate(
            parameters=parameters, config={}
        )
        log(INFO, f"VirtualLLMClient.evaluate :: evaluation loss is {loss}")
        log(
            INFO,
            "VirtualLLMClient.evaluate :: number of example evaluated is %s.",
            num_examples,
        )
        log(INFO, f"VirtualLLMClient.evaluate :: evaluation metrics={metrics}")


if __name__ == "__main__":
    main()
