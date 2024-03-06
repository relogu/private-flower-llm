#!/bin/bash
#SBATCH -c 10
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH --job-name=PR1-llb
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --dependency=afterany:78083

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

# Set the custom hydra arguments that will be passed to the server and the node manager
policy="llb"
echo "Using policy $policy"
CUSTOM_HYDRA_ARGS="num_nodes=1 run_uuid=$run_uuid task=reddit task.n_clients_per_round=100 task.num_rounds=100 local_epochs=1 placement_policy=$policy flwr_address=127.0.0.1:6482"

# Launch the server.
poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS &

# Launch the node manager.
poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_pollen_singlenode.sh