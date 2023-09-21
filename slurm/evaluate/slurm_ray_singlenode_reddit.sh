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
source /nfs-share/ls985/anaconda3/bin/activate ray111-test
cd /nfs-share/aai30/projects/pollen_worker

# Clean the shared memory objects
python clean_memory.py

# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="multirun_output_dir=/nfs-share/aai30/projects/pollen_worker/multirun/2023-09-18/10-46-20 task="reddit" use_wandb=True wandb.start_eval=0"


# Launch the server, uncomment the end of the line if you what separed output logs.
python -m models.multirun_testing_loops $CUSTOM_HYDRA_ARGS 

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_ray_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_ray_singlenode.sh