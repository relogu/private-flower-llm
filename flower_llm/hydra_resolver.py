"""Script for resolving the hydra configuration."""

import os
import hydra
from omegaconf import OmegaConf

from flower_llm.conf import base_schema
from flower_llm.conf.base_schema import BaseConfig

base_schema.register_config(name="base_schema")


# Define strategy
@hydra.main(config_path="conf/", config_name="base", version_base=None)
def main(cfg: BaseConfig) -> None:
    """Resolve the configuration and dump it to a YAML file."""
    # Resolve the configuration
    OmegaConf.resolve(cfg)
    # Get the environmental variable for the dump folder
    save_path = os.environ.get("POLLEN_SAVE_PATH", "")
    # Raise an error if the environmental variable is not set
    if not save_path:
        raise ValueError("The environmental variable POLLEN_SAVE_PATH is not set.")
    # Dump the configuration to a YAML file under the dump folder
    OmegaConf.save(cfg, save_path + "/config.yaml")


if __name__ == "__main__":
    main()
