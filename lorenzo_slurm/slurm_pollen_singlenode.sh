#!/bin/bash
#SBATCH -c 11
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH --job-name=openimage_pollen
#SBATCH --partition=interactive
#SBATCH --tasks-per-node=1

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd $HOME/projects/pollen_worker
poetry shell

# Set the custom hydra arguments that will be passed to the server and the node manager
for policy in "lb" "llb" "rr"; do
    echo "Using policy $policy"
    # CUSTOM_HYDRA_ARGS="num_nodes=1 run_uuid=$run_uuid task=shakespeare_memory task.n_clients_per_round=200 task.num_rounds=100 local_epochs=1 placement_policy=$policy flwr_address=127.0.0.1:6479"
    # CUSTOM_HYDRA_ARGS="num_nodes=1 run_uuid=$run_uuid task=openimage task.n_clients_per_round=100 task.num_rounds=100 local_epochs=1 placement_policy=$policy flwr_address=127.0.0.1:6482"
    # CUSTOM_HYDRA_ARGS="num_nodes=1 run_uuid=$run_uuid task=google_speech task.n_clients_per_round=100 task.num_rounds=100 local_epochs=1 placement_policy=$policy flwr_address=127.0.0.1:6485"
    # CUSTOM_HYDRA_ARGS="num_nodes=1 run_uuid=$run_uuid task=reddit task.n_clients_per_round=100 task.num_rounds=100 local_epochs=1 placement_policy=$policy flwr_address=127.0.0.1:6488"

    # Launch the server.
    poetry run python -m pollen_worker.server_with+pollen $CUSTOM_HYDRA_ARGS &

    # Launch the node manager.
    poetry run python -m pollen_worker.pure_sh_node_manager $CUSTOM_HYDRA_ARGS
done

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_pollen_singlenode.sh