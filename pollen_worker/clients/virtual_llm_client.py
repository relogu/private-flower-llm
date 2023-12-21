"""Lightweight Flower Client for Pollen, capable of sustaining LLm training.

Clients are trained by workers which are managed by node managers. This type of client
avoids any memory or processing intensive operations in the _init_ function. As such,
virtual clients can be used to simulate a large number of clients on a single machine
even if many are spawned at once.
"""
import copy
import time
from logging import INFO
from typing import Any, Callable, Dict, Optional, Union

import flwr as fl
import hydra
import transformers
from composer import Trainer
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from omegaconf import DictConfig, OmegaConf

from pollen_worker.clients.llm_client_functions import (
    _get_trainer_object,
    get_parameters,
    get_raw_model_parameters,
    llm_eval,
    llm_fit,
    set_n_workers_dataloaders,
)


class VirtualLLMClient(fl.client.NumPyClient):
    """Implement the most lightweight Flower Client."""

    def __init__(
        self,
        *,
        cid: Union[int, str],
        cfg: Optional[DictConfig] = None,
        trainer: Optional[Trainer] = None,
    ) -> None:
        self.cid = cid
        self.cfg = cfg
        self.trainer = trainer
        transformers.logging.set_verbosity_error()
        # log(INFO, f'VirtualLLMClient.__init__ :: cid {self.cid}')

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"VirtualLLMClient(cid={self.cid})"

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
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
        # TODO: Decide what to do here.
        # Get the `cfg` object if it exists, raise error otherwise
        # cfg: DictConfig = config.get("cfg", None)
        cfg: Optional[DictConfig] = copy.deepcopy(self.cfg)
        if cfg is None:
            raise ValueError(
                "The `cfg` object is missing from the config/object. "
                "Please ensure that the `cfg` object is passed to the client."
            )
        return get_parameters(config, cfg, self.trainer)

    def fit(
        self, parameters: NDArrays, config: Dict
    ) -> tuple[NDArrays, int, Union[Dict[str, Scalar], dict[Any, Any]]]:
        """Implement the fit step."""
        # log(INFO, f'VirtualLLMClient.fit :: {config}')
        # TODO: Decide what to do here.
        # Get the `cfg` object if it exists, raise error otherwise
        # cfg: DictConfig = config.get("cfg", None)
        cfg: Optional[DictConfig] = copy.deepcopy(self.cfg)
        if cfg is None:
            raise ValueError(
                "The `cfg` object is missing from the config/object. "
                "Please ensure that the `cfg` object is passed to the client."
            )

        return llm_fit(parameters, config, cfg, self.trainer)

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> tuple[float, int, Dict[str, Scalar]]:
        """Implement the evaluation step."""
        # log(INFO, f'VirtualLLMClient.evaluate :: {config}')
        # TODO: Decide what to do here.
        # Get the `cfg` object if it exists, raise error otherwise
        # cfg: DictConfig = config.get("cfg", None)
        cfg: Optional[DictConfig] = copy.deepcopy(self.cfg)
        if cfg is None:
            raise ValueError(
                "The `cfg` object is missing from the config/object. "
                "Please ensure that the `cfg` object is passed to the client."
            )
        return llm_eval(parameters, config, cfg, self.trainer)


def gen_client_fn(
    cfg: Optional[DictConfig] = None,
    trainer: Optional[Trainer] = None,
    **kwargs,
) -> Callable[[int], VirtualLLMClient]:
    """Return generic `client_fn` for Flower Framework."""

    def client_fn(client_id: int) -> VirtualLLMClient:
        client = VirtualLLMClient(
            cid=client_id,
            cfg=copy.deepcopy(cfg),
            trainer=trainer,
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
    # Automatically setting the `n_workers` parameter based on CPU available
    _llm_config = set_n_workers_dataloaders(_llm_config)
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    log(
        INFO,
        "NodeManager received the llm_config:\n%s",
        OmegaConf.to_yaml(_llm_config, resolve=True),
    )
    assert isinstance(_llm_config, DictConfig)
    # Get the client generator function
    gen_client_fn(
        cfg=copy.deepcopy(_llm_config),
    )
    # Get initial model parameters
    parameters = get_raw_model_parameters(copy.deepcopy(_llm_config))
    log(INFO, f"get_raw_model_parameters :: parameters' length is {len(parameters)}")
    # Extract configs to build the trainer
    trainer, _, _ = _get_trainer_object(
        _cfg=copy.deepcopy(_llm_config),
    )
    # Create a virtual client
    virtual_llm_client = VirtualLLMClient(
        cid=0,
        cfg=copy.deepcopy(_llm_config),
        trainer=trainer,
    )
    # Test virtual client's get_properties function
    properties = virtual_llm_client.get_properties(config={})
    log(INFO, f"VirtualLLMClient.get_properties :: properties={properties}")
    # Test virtual client's get_parameters function
    parameters = virtual_llm_client.get_parameters(config={})
    log(
        INFO,
        f"VirtualLLMClient.get_parameters :: parameters' length is {len(parameters)}",
    )

    # Test virtual client's fit function
    parameters, num_examples, metrics = virtual_llm_client.fit(
        parameters=parameters, config={}
    )
    log(INFO, f"VirtualLLMClient.fit :: parameters' length is {len(parameters)}")
    log(INFO, f"VirtualLLMClient.fit :: number of example trained is {num_examples}")
    log(INFO, f"VirtualLLMClient.fit :: train metrics={metrics}")

    # NOTE: Can't do both train and test in the same process currently
    # # Test virtual client's evaluate function
    # loss, num_examples, metrics = virtual_llm_client.evaluate(
    #     parameters=parameters, config={}
    # )
    # log(INFO, f"VirtualLLMClient.evaluate :: evaluation loss is {loss}")
    # log(
    #     INFO,
    #     f"VirtualLLMClient.evaluate :: number of example evaluated is {num_examples}",
    # )
    # log(INFO, f"VirtualLLMClient.evaluate :: evaluation metrics={metrics}")


if __name__ == "__main__":
    main()
