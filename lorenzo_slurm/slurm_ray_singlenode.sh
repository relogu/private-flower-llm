#!/bin/bash
#SBATCH -c 11
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH --job-name=pollen_worker_bench
#SBATCH --partition=interactive
#SBATCH --tasks-per-node=1

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd /nfs-share/ls985/pollen_worker
poetry shell

# Clean the shared memory objects
  

# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=shakespeare_memory task.num_rounds=100 local_epochs=10 task.batch_size=10 flwr_address=127.0.0.1:1046"

# Launch the server, uncomment the end of the line if you what separed output logs.
poetry run python pollen_worker/ray_simulation.py $CUSTOM_HYDRA_ARGS 

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash lorenzo_slurm/slurm_ray_singlenode.sh
# Use what follows for a batch job
# sbatch lorenzo_slurm/slurm_ray_singlenode.sh