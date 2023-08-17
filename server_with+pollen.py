from typing import Dict
from logging import INFO

import flwr as fl
import hydra
from flwr.client import ClientLike
from flwr.common import ndarrays_to_parameters
from flwr.common.typing import Scalar
from flwr.common.logger import log
from hydra.utils import call
from omegaconf import DictConfig

from pollen_client_manager import PollenClientManager
from pollen_server import PollenServer
from pollen_utils import get_clients_population_dict
from rs_fedavg import FedAvgReproducibleSampling
from utils import weighted_average
from virtual_client import VirtualClient


# Define strategy
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # # The number of clients can either be a single integer or a list of int/str
    # num_total_virtual_clients = call(cfg.gen_num_total_virtual_clients)

    # Get the list of cids
    cid_samples_dict = get_clients_population_dict(
        name=cfg.dataset_name,
        batch_size=cfg.batch_size,
    )
    n_total_clients = len(cid_samples_dict)
    n_clients_per_round = 10

    def get_client_fn(cid: int, device: str) -> ClientLike:
        return VirtualClient(
            name=cfg.dataset_name,
            cid=cid,
            device=device,
            n_workers=0,
        )

    def on_fit_config_fn(rnd: int) -> Dict[str, Scalar]:
        return {
            "batch_size": cfg.batch_size,
            "epochs": 1,
        }

    strategy = FedAvgReproducibleSampling(
        total_clients=n_total_clients,
        min_fit_clients=2,
        fraction_evaluate=0.0,
        fraction_fit=n_clients_per_round / n_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=ndarrays_to_parameters(
            get_client_fn(cid=0, device="cpu").get_parameters(config={}, net=None)
        ),
        fit_metrics_aggregation_fn=weighted_average,
    )
    log(INFO, strategy.fraction_fit)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        server=PollenServer(
            cids=cid_samples_dict,
            client_fn=get_client_fn,
            strategy=strategy,
            client_manager=PollenClientManager(),
            placement_policy="rr",
        ),
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
    )


if __name__ == "__main__":
    main()
