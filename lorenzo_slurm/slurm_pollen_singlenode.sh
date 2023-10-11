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

# Set the custom hydra arguments that will be passed to the server and the node manager
# CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=shakespeare_memory task.n_clients_per_round=10 task.num_rounds=100 local_epochs=1 placement_policy=lb flwr_address=127.0.0.1:1044"
CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=openimage task.n_clients_per_round=10 task.num_rounds=100 local_epochs=1 placement_policy=llb flwr_address=127.0.0.1:1044"

# Launch the server.
poetry run python -m pollen_worker.server_with+pollen $CUSTOM_HYDRA_ARGS &

# Launch the node manager.
poetry run python -m pollen_worker.pure_sh_node_manager $CUSTOM_HYDRA_ARGS

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_pollen_singlenode.sh