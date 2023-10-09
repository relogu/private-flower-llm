from copy import deepcopy
from pathlib import Path

import hydra
from omegaconf import DictConfig

from models.testing_loops import main as test_main


@hydra.main(config_path="../conf/", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Recursively test models from all the directories in the multirun_output_dir."""
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python -m models.multirun_testing_loops multirun_output_dir="/nfs-share/aai30/projects/pollen_worker/multirun/2023-09-18/08-59-49" task="shakespeare_memory" use_wandb=True wandb.start_eval=0
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python -m models.multirun_testing_loops multirun_output_dir=/nfs-share/aai30/projects/pollen_worker/multirun/2023-09-18/10-46-20 task="reddit" use_wandb=True wandb.start_eval=0
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python -m models.multirun_testing_loops multirun_output_dir=/nfs-share/aai30/projects/pollen_worker/multirun/2023-09-18/12-18-57 task="openimage" use_wandb=True wandb.start_eval=0
    # srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive python -m models.multirun_testing_loops multirun_output_dir=/nfs-share/aai30/projects/pollen_worker/multirun/2023-09-19/20-57-36 task="google_speech" use_wandb=True wandb.start_eval=0

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
