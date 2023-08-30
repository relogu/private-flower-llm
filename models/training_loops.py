import torch
from torch.autograd import Variable
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import AlbertTokenizer

from datasets.nlp_util import mask_tokens


def get_training_loop(name: str):
    if name == "reddit":
        return reddit_training_loop
    elif name == "google_speech":
        return google_speech_training_loop
    else:
        return general_training_loop


def get_input_shapes(name: str):
    if name == "reddit":
        return (64,)
    elif name == "google_speech":
        return (1, 32, 32)
    elif "shakespeare" in name:
        return (80,)
    # elif name == "openimage":
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
):
    for _ in range(epochs):
        for i, data in enumerate(trainloader):
            # TODO: handle steps instead of epochs
            # if i >= n_batches:
            #     break

            # ========= Pre-processing + placement ===========
            data, target = mask_tokens(
                data, tokenizer, mlm_probability=0.15, device=device
            )

            data = Variable(data).to(device=device)
            target = Variable(target).to(device=device)

            # ========= Define the forward pass ==============
            outputs = net(data, labels=target)
            loss = outputs[0]

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def google_speech_training_loop(
    trainloader: DataLoader,
    net: Module,
    device: torch.device,
    epochs: int,
    optimizer: Optimizer,
    criterion: Module,
    **kwargs,
):
    for _ in range(epochs):
        for batch in trainloader:
            # TODO: handle steps instead of epochs
            # if i >= n_batches:
            #     break

            # ========= Pre-processing + placement ===========
            (data, target) = batch
            data = torch.unsqueeze(data, 1).to(device=device)

            target = Variable(target).to(device=device)

            # ========= Define the forward pass ==============
            output = net(data)
            loss = criterion(output, target)
            loss = loss.mean()

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def general_training_loop(
    trainloader: DataLoader,
    net: Module,
    device: torch.device,
    epochs: int,
    optimizer: Optimizer,
    criterion: Module,
    **kwargs,
):
    for _ in range(epochs):
        for batch in trainloader:
            # TODO: handle steps instead of epochs
            # if i >= n_batches:
            #     break

            # ========= Pre-processing + placement ===========
            (data, target) = batch
            data = Variable(data).to(device=device)
            target = Variable(target).to(device=device)

            # ========= Define the forward pass ==============
            output = net(data)
            loss = criterion(output, target)
            loss = loss.mean()

            # ========= Define the backward pass ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
