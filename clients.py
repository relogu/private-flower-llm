from typing import Callable, Dict, Tuple

import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from datasets.shakespeare import SHAKESPEARE_LOADED as ShakespeareDataset
from models.shakespeare_leaf_model import ShakespeareLeafNet
from utils import set_parameters


class ShakespeareClient(NumPyClient):
    def __init__(self, data_root: str, client_id: int):
        self.data_root = data_root
        self.trainset = ShakespeareDataset(self.data_root, client_id=client_id)
        """self.evalset = ShakespeareDataset(
            self.data_root, client_id=client_id, dataset="test"
        )"""
        self.net = ShakespeareLeafNet()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        trainloader = DataLoader(
            self.trainset, batch_size=config["batch_size"], shuffle=True
        )
        device = config["device"]
        set_parameters(self.net, parameters, device)
        self.net.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )
        for _ in range(config["local_epochs"]):
            num_samples = 0
            num_correct = 0
            for data in trainloader:
                inputs, labels = data[0].to(device), data[1].to(device)
                num_samples += len(labels)
                optimizer.zero_grad()
                predicitons = self.net(inputs)
                num_correct += (
                    (torch.max(predicitons.data, 1)[1] == labels).sum().item()
                )
                criterion(predicitons, labels.to(device)).backward()
                optimizer.step()
        self.net.eval()
        these_weights = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        accuracy = num_correct / num_samples
        return (these_weights, num_samples, {"accuracy": accuracy})


def gen_shakespeare_client_fn(data_root: str) -> Callable[[int], NumPyClient]:
    def client_fn(client_id: int) -> NumPyClient:
        client = ShakespeareClient(data_root, client_id)
        return client

    return client_fn
