#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=ray_worker_bench
#SBATCH --tasks-per-node=1

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd /nfs-share/aai30/projects/pollen_worker/
poetry shell

# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=shakespeare_memory task.num_rounds=100 use_wandb=True wandb.setup.project=test flwr_address=127.0.0.1:1046"

# Launch the server, uncomment the end of the line if you what separed output logs.
poetry run python src/ray_simulation.py $CUSTOM_HYDRA_ARGS 

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_ray_singlenode.sh
# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash alex_slurm/slurm_ray_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_ray_singlenode.sh