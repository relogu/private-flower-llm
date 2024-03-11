"""Lightweight Flower Client for Pollen.

Clients are trained by workers which are managed by node managers. This type of client
avoids any memory or processing intensive operations in the _init_ function. As such,
virtual clients can be used to simulate a large number of clients on a single machine
even if many are spawned at once.
"""

from collections import OrderedDict
from collections.abc import Callable
from logging import INFO
from typing import Any

import flwr as fl
import torch
import transformers
from flwr.client import NumPyClient
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch import device as device_type
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from pollen_worker.datasets.nlp_util import get_collate_fn
from pollen_worker.models.training_loops import get_input_shapes, get_training_loop
from pollen_worker.pollen_utils import (
    get_client_ds,
    get_device,
    get_model,
    get_optimizer,
)


class VirtualClient(fl.client.NumPyClient):
    """Implement the most lightweight Flower Client."""

    def __init__(
        self,
        *,
        name: str,
        cid: int | str,
    ) -> None:
        self.name = name
        self.cid = cid
        self.client_dataset: Dataset
        transformers.logging.set_verbosity_error()
        # log(INFO, f'VirtualClient.__init__ :: cid {self.cid}')

    def __repr__(self) -> str:
        """Implement the string representation."""
        return f"VirtualClient(name={self.name}, cid={self.cid})"

    def get_properties(self, config: Config) -> dict[str, Scalar]:
        """Implement how to get properties."""
        return {}

    def get_parameters(
        self,
        config: Config,
        net: Module = None,
        device: str = "cpu",
        to_numpy: bool = True,
    ) -> NDArrays:
        """Implement how to get parameters."""
        if net is None:
            net = get_model(name=self.name)
        net.eval()
        if device == "cpu" and to_numpy:
            tmp = [
                val.detach().to(device).numpy()
                for name, val in net.state_dict().items()
                if "bn" not in name
            ]
        elif device == "cpu" and not to_numpy:
            tmp = [
                val.detach().to(device)
                for name, val in net.state_dict().items()
                if "bn" not in name
            ]
        else:
            tmp = [
                val.detach().to(device)
                for name, val in net.state_dict().items()
                if "bn" not in name
            ]
        return tmp

    def set_parameters(
        self,
        parameters: NDArrays,
        net: Module | None = None,
        device: str | device_type = "cpu",
    ) -> Module:
        """Implement how to set parameters."""
        if net is None:
            net = get_model(name=self.name)
        net.eval()
        module_state_dict = net.state_dict()
        keys = [k for k in module_state_dict if "bn" not in k]
        params_dict = zip(keys, parameters, strict=False)
        state_dict = OrderedDict(
            {k: torch.tensor(v, device=device) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=False)
        return net

    def _train_loop(
        self,
        net: Module,
        batch_size: int,
        device: torch.device,
        optimizer: Optimizer,
        criterion: Module,
        epochs: int,
        **kwargs: Any,
    ) -> tuple[Module, dict[Any, Any]]:
        """Train the model on the training set of single client."""
        log(INFO, f"VirtualClient._train_loop :: FAKE with cid {self.cid}")
        # Load the fake data directly into the VRAM
        input_shape = get_input_shapes(name=self.name)
        if self.name == "reddit":
            fake_data = torch.zeros(((batch_size,) + input_shape), device=device).long()
            fake_targets = torch.zeros(
                (batch_size,) + input_shape, device=device
            ).long()
        else:
            fake_data = torch.zeros(((batch_size,) + input_shape), device=device)
            fake_targets = torch.zeros((batch_size,), device=device).long()
        for _ in range(epochs):
            # ========= Define the forward pass ==============
            if self.name == "reddit":
                output = net(fake_data, labels=fake_targets)
                loss = output[0]
            else:
                output = net(fake_data)
                loss = criterion(output, fake_targets)
                loss = loss.mean()

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        log(INFO, f"VirtualClient._train_loop :: finished training of cid {self.cid}")
        return net, {}

    def fit(
        self, parameters: NDArrays, config: dict
    ) -> tuple[NDArrays, int, dict[str, Scalar] | dict[Any, Any]]:
        """Implement the fit step."""
        # log(INFO, f'VirtualClient.fit :: {config}')
        if "device" not in config:
            config["device"] = get_device()
        # Load client's dataset
        if self.client_dataset is None:
            ds, tokenizer = (
                get_client_ds(name=self.name, cid=int(self.cid))
                if not config["is_fake"]
                else (None, None)
            )
        if hasattr(self.client_dataset, "tokenizer"):
            tokenizer = self.client_dataset.tokenizer  # type: ignore[union-attr]
        ds = self.client_dataset

        # Instantiate the trainloader
        trainloader = (
            DataLoader(
                # NOTE: Non-default arguments
                dataset=ds,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=config["n_workers"],
                # NOTE: Prevent runtime error related to BatchNorm, apparently
                drop_last=(self.name == "google_speech"),
                # copy Tensors into CUDA pinned memory before returning them
                pin_memory=True,
                # the device to be used for pinning the memory
                pin_memory_device=str(config["device"]),
                # builds batches from samples
                collate_fn=(
                    get_collate_fn(tokenizer=tokenizer)
                    if tokenizer is not None
                    else None
                ),
                # NOTE: Default arguments
                # how to draw sample from the dataset
                sampler=None,
                # like the above but for batches
                batch_sampler=None,
                # if positive, the timeout value for collecting a batch from workers
                timeout=0,
                # init function for worker processes
                worker_init_fn=None,
                multiprocessing_context=None,
                # This allows to maintain the workers
                prefetch_factor=2 if config["n_workers"] > 0 else None,
                # PRNG to use for random sampling
                generator=None,
                persistent_workers=False,
            )
            if not (config["is_fake"] or ds is None)
            else None
        )
        # Get the number of samples
        n_samples = (
            int(config["batch_size"] * config["local_epochs"])
            if ds is None
            else len(ds)
        )
        # Initialize the model and set its parameters
        net = self.set_parameters(parameters=parameters, device=config["device"])
        net.to(device=config["device"])
        net.train()
        # Train the model
        self._train_loop = (  # type: ignore[method-assign]
            get_training_loop(name=self.name)  # type: ignore[assignment]
            if not config["is_fake"]
            else self._train_loop
        )
        optimizer = get_optimizer(name=self.name, model=net)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean").to(
            device=config["device"]
        )
        net, train_metrics = self._train_loop(
            trainloader=trainloader,
            net=net,
            device=config["device"],
            epochs=config["local_epochs"],
            tokenizer=tokenizer,
            optimizer=optimizer,
            criterion=criterion,
            batch_size=config["batch_size"],
        )
        # log(INFO, f"VirtualClient.fit :: train_metrics {train_metrics}")
        return self.get_parameters(config={}, net=net), n_samples, train_metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict[str, Scalar],
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Implement the evaluation step."""
        return 0.0, 0, {"local_accuracy": 0.0}


def gen_client_fn(
    name: str = "openimage", **kwargs: Any
) -> Callable[[int], NumPyClient]:
    """Return generic `client_fn` for Flower Framework."""

    def client_fn(client_id: int) -> NumPyClient:
        client = VirtualClient(name=name, cid=client_id)
        return client

    return client_fn


if __name__ == "__main__":
    log(INFO, "VirtualClient.__main__ :: testing VirtualClient")

    client = VirtualClient(
        name="openimage",
        cid=0,
    )
    log(INFO, f"VirtualClient.__main__ :: created {client}")
    config: Config = {
        "batch_size": 32,
        "epochs": 1,
    }
    params, n_samples, metrics = client.fit(
        parameters=client.get_parameters(config=config), config=config
    )
    log(INFO, f"VirtualClient.__main__ :: n_samples {n_samples}")
    # log(INFO, f'VirtualClient.__main__ :: params {params}')
