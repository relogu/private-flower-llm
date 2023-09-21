from typing import Callable, Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from datasets.shakespeare import SHAKESPEARE_LOADED as ShakespeareDataset
from models.shakespeare_leaf_model import ShakespeareLeafNet
from pure_sh_node_manager import get_parameters
from utils import set_parameters


def train(
    net: torch.nn.Module,
    trainloader: DataLoader,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
    net.to(device)
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    # for _ in tqdm(range(epochs)):
    for _ in range(epochs):
        num_samples = 0
        num_correct = 0
        for data in trainloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            num_samples += len(labels)
            optimizer.zero_grad()
            predicitons = net(inputs)
            num_correct += (torch.max(predicitons.data, 1)[1] == labels).sum().item()
            criterion(predicitons, labels.to(device)).backward()
            optimizer.step()
    net.eval()
    these_weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    accuracy = num_correct / num_samples
    return (these_weights, num_samples, {"accuracy": accuracy})


class ShakespeareClient(NumPyClient):
    def __init__(self, data_root: str, client_id: int):
        self.data_root = data_root
        self.trainset = ShakespeareDataset(self.data_root, client_id=client_id)
        """self.evalset = ShakespeareDataset(

        self.data_root, client_id=client_id, dataset="test" )
        """
        self.net = ShakespeareLeafNet()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        net = ShakespeareLeafNet()
        return get_parameters(net)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        set_parameters(self.net, parameters, device=config["device"])
        trainloader = DataLoader(
            self.trainset, batch_size=config["batch_size"], shuffle=True
        )
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        return train(
            net=self.net,
            trainloader=trainloader,
            epochs=config["local_epochs"],
            optimizer=optimizer,
            device=config["device"],
        )


def gen_shakespeare_client_fn(data_root: str) -> Callable[[int], NumPyClient]:
    def client_fn(client_id: int) -> NumPyClient:
        client = ShakespeareClient(data_root, client_id)
        return client

    return client_fn
