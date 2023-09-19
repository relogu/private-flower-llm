from logging import DEBUG, INFO
from pathlib import Path

import flwr as fl
import hydra
import transformers
from flwr.client import ClientLike
from flwr.common import ndarrays_to_parameters
from flwr.common.logger import log
from hydra.utils import call, instantiate
from omegaconf import DictConfig

from pollen_client_manager import PollenClientManager
from pollen_server import PollenServer
from pollen_utils import get_clients_population_dict
from utils import weighted_average
from virtual_client import VirtualClient

transformers.logging.set_verbosity_error()


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    log(
        INFO,
        f"Task is: {cfg.task.name} with fake={cfg.task.is_fake} with run unique id: {cfg.run_uuid}",
    )

    # Get the list of cids
    import time

    s_t = time.time()
    try:
        cid_samples_dict = get_clients_population_dict(
            name=cfg.task.name,
            batch_size=cfg.task.batch_size,
        )
    except Exception as e:
        log(DEBUG, f"Exception while getting the clients' dictionary: {e}")
        cid_samples_dict = {k: 1 for k in range(cfg.task.n_clients_per_round)}
    log(INFO, f"Time to get the clients' dictionary: {time.time() - s_t}")
    n_total_clients = len(cid_samples_dict)
    n_clients_per_round = cfg.task.n_clients_per_round

    def get_client_fn(
        cid: int,
    ) -> ClientLike:
        return VirtualClient(
            name=cfg.task.name,
            cid=cid,
        )

    on_fit_config_fn = call(cfg.gen_on_fit_config_fn)
    # Storing the parameters to the hydra output directory
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    strategy = instantiate(
        cfg.task.strategy,
        saving_path=Path(hydra_cfg["runtime"]["output_dir"]),
        min_fit_clients=2,
        fraction_evaluate=0.0,
        fraction_fit=n_clients_per_round / n_total_clients,
        on_fit_config_fn=on_fit_config_fn,
        initial_parameters=ndarrays_to_parameters(
            get_client_fn(cid=0).get_parameters(config={}, net=None)
        ),
        fit_metrics_aggregation_fn=weighted_average,
    )
    log(INFO, f"Fraction fit is: {strategy.fraction_fit}")

    # Start Flower server
    fl.server.start_server(
        server_address=cfg.flwr_address,
        server=PollenServer(
            cids=cid_samples_dict,
            client_fn=get_client_fn,
            strategy=strategy,
            client_manager=PollenClientManager(),
            placement_policy="rr",
            saving_path=Path(hydra_cfg["runtime"]["output_dir"]),
        ),
        config=fl.server.ServerConfig(num_rounds=cfg.task.num_rounds),
    )


if __name__ == "__main__":
    main()
