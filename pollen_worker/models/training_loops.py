"""Training loops for the different tasks of the Pollen paper."""
from typing import Dict, Tuple

import torch
from flwr.common import Scalar
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer
from transformers.modeling_outputs import MaskedLMOutput

from pollen_worker.datasets.nlp_util import mask_tokens


def get_training_loop(name: str):
    """Return the train loop function given the task's name."""
    if name == "reddit":
        return reddit_training_loop
    elif name == "google_speech":
        return google_speech_training_loop
    else:
        return general_training_loop


def get_input_shapes(name: str):
    """Return the input shapes given the task's name."""
    if name == "reddit":
        return (64,)
    elif name == "google_speech":
        return (1, 32, 32)
    elif "shakespeare" in name:
        return (80,)
    else:
        return (3, 256, 256)


def reddit_training_loop(
    trainloader: DataLoader,
    net: Module,
    device: torch.device,
    epochs: int,
    optimizer: Optimizer,
    tokenizer: AlbertTokenizer,
    **kwargs,
) -> Tuple[Module, Dict[str, Scalar]]:
    """Implement Reddit task's train loop."""
    for _ in range(epochs):
        current_loss = 0.0
        num_masked = 0
        num_correct = 0
        for data in trainloader:
            # TODO: handle steps instead of epochs ?

            # ========= Pre-processing + placement ===========
            data: torch.Tensor = data.to(device=device)
            data, target, masked_indices = mask_tokens(
                # TODO: Read the `mlm_probability` from the config
                data,
                tokenizer,
                mlm_probability=0.15,
                device=device,
            )
            target = target.to(device=device)
            num_masked += len(target[masked_indices])

            # ========= Define the forward pass ==============
            output: MaskedLMOutput = net(input_ids=data, labels=target)
            current_loss += output.loss.item()
            predictions = output.logits.max(2)[1]
            # Only computing accuracy on the masked tokens
            num_correct += (
                (predictions[masked_indices] == data[masked_indices])
                .clone()
                .detach()
                .sum()
                .item()
            )

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
        accuracy = num_correct / num_masked
    return net, {
        "train_loss": current_loss / len(trainloader),
        "accuracy": accuracy,
    }


def google_speech_training_loop(
    trainloader: DataLoader,
    net: Module,
    device: torch.device,
    epochs: int,
    optimizer: Optimizer,
    criterion: Module,
    **kwargs,
) -> Tuple[Module, Dict[str, Scalar]]:
    """Implement Google Speech task's train loop."""
    for _ in range(epochs):
        current_loss = 0.0
        num_samples = 0
        num_correct = 0
        for batch in trainloader:
            # TODO: handle steps instead of epochs ?

            # ========= Pre-processing + placement ===========
            data: torch.Tensor = batch[0]
            # NOTE: The following line is what makes the difference
            # w.r.t. `general_training_loop`
            data = torch.unsqueeze(data, 1).to(device=device)
            target: torch.Tensor = batch[1]
            target = target.to(device=device)
            num_samples += len(target)
            optimizer.zero_grad()

            # ========= Define the forward pass ==============
            output: torch.Tensor = net(data)
            loss: torch.Tensor = criterion(output, target)
            current_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()

            # ========= Define the backward pass ==============
            loss.backward()
            optimizer.step()
        accuracy = num_correct / num_samples
    return net, {
        "train_loss": current_loss / len(trainloader),
        "accuracy": accuracy,
    }


def general_training_loop(
    trainloader: DataLoader,
    net: Module,
    device: torch.device,
    epochs: int,
    optimizer: Optimizer,
    criterion: Module,
    **kwargs,
) -> Tuple[Module, Dict[str, Scalar]]:
    """Implement Shakespeare and Open Image task's train loop."""
    for _ in range(epochs):
        current_loss = 0.0
        num_samples = 0
        num_correct = 0
        for batch in trainloader:
            # TODO: handle steps instead of epochs?

            # ========= Pre-processing + placement ===========
            data: torch.Tensor = batch[0]
            target: torch.Tensor = batch[1]

            data = data.to(device=device)
            target = target.to(device=device)
            num_samples += len(target)
            optimizer.zero_grad()

            # ========= Define the forward pass ==============
            output: torch.Tensor = net(data)
            loss: torch.Tensor = criterion(output, target)
            current_loss += loss.item()
            num_correct += (output.max(1)[1] == target).clone().detach().sum().item()

            # ========= Define the backward pass ==============
            loss.backward()
            optimizer.step()
        accuracy = num_correct / num_samples
    return net, {
        "train_loss": current_loss / len(trainloader),
        "accuracy": accuracy,
    }
