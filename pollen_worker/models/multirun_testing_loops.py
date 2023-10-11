"""Hydra-specific script to recursively test models from a multirun.

It operates over all the directories in the multirun_output_dir and applies centralised
evluation over the concatenated client test sets.
"""

from copy import deepcopy
from pathlib import Path

import hydra
from models.testing_loops import main as test_main
from omegaconf import DictConfig


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Recursively test models from all the directories in the multirun_output_dir."""
    multirun_output_dir = Path(cfg.multirun_output_dir)
    print(multirun_output_dir)
    visitd_dirs = set()

    # In case you want to evaluate prior to an experiment ending
    # and need to start where you left off
    for i in range(cfg.wandb.start_eval):
        visitd_dirs.add(multirun_output_dir / str(i))

    for dir in multirun_output_dir.iterdir():
        if dir.is_dir() and dir not in visitd_dirs:
            new_cfg = deepcopy(cfg)
            new_cfg.output_dir = str(dir)
            test_main(new_cfg)
            visitd_dirs.add(dir)


if __name__ == "__main__":
    main()
