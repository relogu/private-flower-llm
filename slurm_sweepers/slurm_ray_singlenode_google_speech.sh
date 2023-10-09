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

# This wandb agent should be returned to you after you run wandb seweep 
wandb agent camlsys/pollen/hj0ihtrr