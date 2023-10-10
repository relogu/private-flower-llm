#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:2
#SBATCH --job-name=ray_worker_bench
#SBATCH --tasks-per-node=1

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd /nfs-share/aai30/projects/pollen_worker
poetry shell


# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="-m run_uuid=$run_uuid task=google_speech task.num_rounds=100 local_epochs=1,2,3 task.learning_rate=0.001,0.005,0.01,0.05,0.1 flwr_address=127.0.0.1:1046"


# Launch the server, uncomment the end of the line if you what separed output logs.
poetry run python pollen_worker/ray_simulation.py $CUSTOM_HYDRA_ARGS 

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash alex_slurm/slurm_ray_singlenode.sh
# Use what follows for a batch job
# sbatch alex_slurm/slurm_ray_singlenode.sh