from typing import List, Tuple

import flwr as fl
import hydra
from flwr.common import Metrics
from omegaconf import DictConfig

from pollen_strategy import FedAvgReproducibleSampling
from utils import weighted_average
from hydra.utils import call


# Define strategy
@hydra.main(config_path="conf/", config_name="shakespeare", version_base=None)
def main(cfg: DictConfig) -> None:
    # The number of clients can either be a single integer or a list of int/str
    num_total_virtual_clients = call(cfg.gen_num_total_virtual_clients)

    strategy = FedAvgReproducibleSampling(
        num_total_virtual_clients=num_total_virtual_clients,
        num_participating_nodes=1,
        fraction_fit=cfg.fraction_fit,
        min_fit_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=1,
        fit_metrics_aggregation_fn=weighted_average,
    )
    print(strategy.fraction_fit)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
