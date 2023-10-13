#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=hypertune
#SBATCH --partition=priority
#SBATCH --tasks-per-node=1
#SBATCH --mem=100G

cd /nfs-share/aai30/projects/pollen_worker
poetry shell

wandb agent camlsys/pollen/fqhsueci