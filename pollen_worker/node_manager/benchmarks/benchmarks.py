import copy
import hydra
from flwr.common import log, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from pollen_worker.clients.llm_client_functions import get_raw_model_parameters
from omegaconf import DictConfig, OmegaConf

class Benchmarks(object):

    def __init__(self):
        self.main()

    def buffer_length_and_minimum_file_size() -> None:
        x = 1





@hydra.main(config_path="../../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    _llm_config = cfg.llm_config
    OmegaConf.resolve(_llm_config)
    OmegaConf.set_struct(_llm_config, False)
    initial_parameters = ndarrays_to_parameters(
        get_raw_model_parameters(copy.deepcopy(_llm_config))
    )
    print(f"Tesors: {len(initial_parameters.tensors)}")



if __name__ == "__main__":
    main()
