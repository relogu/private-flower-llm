"""Hydra-specific script to recursively test models from a multirun.

It operates over all the directories in the multirun_output_dir and applies centralised
evluation over the concatenated client test sets.
"""

from copy import deepcopy
from logging import INFO
from pathlib import Path

import hydra
from flwr.common import log
from omegaconf import DictConfig, OmegaConf

from pollen_worker.models.testing_loops import main as test_main


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Recursively test models from all the directories in the multirun_output_dir."""
    multirun_output_dir = Path(cfg.multirun_output_dir)
    log(INFO, multirun_output_dir)
    visitd_dirs = set()

    # In case you want to evaluate prior to an experiment ending
    # and need to start where you left off
    for i in range(cfg.wandb.start_eval):
        visitd_dirs.add(multirun_output_dir / str(i))

    for _dir in multirun_output_dir.iterdir():
        if _dir.is_dir() and _dir not in visitd_dirs:
            new_cfg = deepcopy(cfg)
            new_cfg.output_dir = str(_dir)
            test_main(new_cfg)
            visitd_dirs.add(_dir)


def composite_testing_loops() -> None:
    """Test models from all listed directories."""
    multirun_output_dir = [
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-02/15-33-23",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-01/09-48-27",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-01/03-45-26",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-09-30/23-06-36",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-01/11-46-08",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-09-30/13-32-43",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-09-29/16-25-08",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-01/13-42-44",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-10-01/19-05-19",
        "/nfs-share/aai30/projects/pollen_worker/outputs/2023-09-22/07-30-00",
    ]

    log(INFO, multirun_output_dir)

    for d in multirun_output_dir:
        cfg_path = Path(d) / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.output_dir = d
        cfg.task.n_clients = -1
        cfg.use_wandb = True
        test_main(cfg)


def task_testing_loops(task: str = "reddit") -> None:
    """Test models from all listed directories."""
    output_dir = Path("/nfs-share/aai30/projects/pollen_worker/outputs")

    output_dirs = [
        _dir
        for top_dir in output_dir.iterdir()
        if top_dir.is_dir()
        for _dir in top_dir.iterdir()
        if _dir.is_dir()
    ]

    for d in output_dirs:
        cfg_path = Path(d) / ".hydra" / "config.yaml"
        cfg = OmegaConf.load(cfg_path)
        if cfg.task.name != task:
            log(INFO, f"Skipping {cfg.task.name}")
            continue

        cfg.output_dir = d
        cfg.task.n_clients = -1
        cfg.use_wandb = True
        cfg.seed = 1337
        test_main(cfg)


if __name__ == "__main__":
    task_testing_loops()
