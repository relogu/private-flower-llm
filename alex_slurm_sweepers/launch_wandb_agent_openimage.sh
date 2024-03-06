#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=hypertune
#SBATCH --partition=priority
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G

cd /nfs-share/aai30/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

wandb agent camlsys/pollen/fqhsueci