from collections import OrderedDict
from logging import INFO
from typing import Dict

import flwr as fl
import torch
from flwr.common.logger import log
from flwr.common.typing import Config, NDArrays, Scalar
from torch.nn import Module
from torch.utils.data import DataLoader

from models.training_loops import get_input_shapes, get_training_loop
from pollen_utils import get_client_ds, get_model, get_optimizer


class VirtualClient(fl.client.NumPyClient):
    def __init__(
        self,
        *,
        name: str,
        cid: int,
        device: torch.device,
        n_workers: int,
    ):
        self.name = name
        self.cid = cid
        self.device = device
        self.n_workers = n_workers
        # log(INFO, f'VirtualClient.__init__ :: cid {self.cid}')

    def __repr__(self) -> str:
        return f"VirtualClient(name={self.name}, cid={self.cid}, device={self.device}, n_workers={self.n_workers})"

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return {}

    def get_parameters(self, config, net=None, device="cpu", to_numpy=True):
        if net is None:
            net = get_model(name=self.name)
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

    def set_parameters(self, parameters: NDArrays, net: Module = None):
        if net is None:
            net = get_model(name=self.name)
        keys = [k for k in net.state_dict().keys() if "bn" not in k]
        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        return net

    def _train_loop(
        self,
        net: Module,
        batch_size: int,
        device: torch.device,
        epochs: int,
        **kwargs,
    ):
        """Train the model on the training set of single client."""
        # log(INFO, f'VirtualClient._train_loop :: with cid {self.cid}')
        # Get the optimiser
        optimizer = get_optimizer(name=self.name, model=net)
        # Get the criterion
        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device=device)
        # Load the fake data directly into the VRAM
        input_shape = get_input_shapes(name=self.name)
        fake_data = torch.zeros(((batch_size,) + input_shape), device=device)
        fake_targets = torch.zeros((batch_size,), device=device).long()
        for _ in range(epochs):
            # ========= Define the forward pass ==============
            output = net(fake_data)
            loss = criterion(output, fake_targets)
            loss = loss.mean()

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # log(INFO, f'VirtualClient._train_loop :: finished training of cid {self.cid}')
        return net

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]):
        # log(INFO, f'VirtualClient.fit :: {config}')
        # TODO: Load client's dataset
        ds, tokenizer = get_client_ds(name=self.name, cid=self.cid)
        # ds, tokenizer = None, None
        # Instantiate the trainloader
        trainloader = (
            DataLoader(
                # NOTE: Non-default arguments
                dataset=ds,
                batch_size=config["batch_size"],
                shuffle=False,
                num_workers=self.n_workers,
                drop_last=True,
                pin_memory=True,  # copy Tensors into CUDA pinned memory before returning them
                pin_memory_device=str(
                    self.device
                ),  # the device to be used for pinning the memory
                # NOTE: Default arguments
                sampler=None,  # how to draw sample from the dataset
                batch_sampler=None,  # like the above but for batches
                collate_fn=None,  # builds batches from samples
                timeout=0,  # if positive, the timeout value for collecting a batch from workers
                worker_init_fn=None,  # init function for worker processes
                multiprocessing_context=None,
                generator=None,  # PRNG to use for random sampling
                prefetch_factor=2,  # number of batches loaded in advance by each worker
                persistent_workers=False,  # if True, the data loader will not shutdown the worker processes after a dataset has been consumed once. This allows to maintain the workers
            )
            if ds is not None
            else None
        )
        # Get the number of samples
        n_samples = (
            int(config["batch_size"] * config["epochs"]) if ds is None else len(ds)
        )
        # Initialize the model and set its parameters
        net = self.set_parameters(parameters=parameters)
        net.to(device=self.device)
        net.train()
        # Train the model
        # # Fake thing
        # net = self._train_loop(
        #     trainloader=trainloader,
        #     net=net,
        #     batch_size=config["batch_size"],
        #     device=self.device,
        #     epochs=config["epochs"],
        #     tokenizer=tokenizer,
        # )
        # Real thing
        training_loop = get_training_loop(name=self.name)
        optimizer = get_optimizer(name=self.name, model=net)
        criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device=self.device)
        net = training_loop(
            trainloader=trainloader,
            net=net,
            batch_size=config["batch_size"],
            device=self.device,
            epochs=config["epochs"],
            tokenizer=tokenizer,
            optimizer=optimizer,
            criterion=criterion,
        )
        return self.get_parameters(config={}, net=net), n_samples, {}

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ):
        return 0.0, 0, {"local_accuracy": 0.0}


if __name__ == "__main__":
    log(INFO, "VirtualClient.__main__ :: testing VirtualClient")
    from pollen.utils import get_device

    client = VirtualClient(
        name="openimage",
        cid=0,
        device=get_device(),
        n_workers=0,
    )
    log(INFO, f"VirtualClient.__main__ :: created {client}")
    config = {
        "batch_size": 32,
        "epochs": 1,
    }
    params, n_samples, metrics = client.fit(
        parameters=client.get_parameters(config=config), config=config
    )
    log(INFO, f"VirtualClient.__main__ :: n_samples {n_samples}")
    # log(INFO, f'VirtualClient.__main__ :: params {params}')
