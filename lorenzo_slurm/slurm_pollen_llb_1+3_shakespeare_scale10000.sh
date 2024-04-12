#!/bin/bash
#SBATCH --nodelist=mauao,ngongotaha
#SBATCH --gres=gpu:3
#SBATCH --job-name=PS13-llb-scale10k
#SBATCH --cpus-per-task 24
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --dependency=afterany:78351

#! Need to force the nodes. Otherwise, the nodes might be allocated randomly.
#! Head node is `mauao`, 128.232.115.0
node_1="mauao"
node_2="ngongotaha"
ip="128.232.115.0"

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# activate the environment and go to the pollen_worker directory
cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate


# Set the custom hydra arguments that will be passed to the server and the node manager
policy="llb"
echo "Using policy $policy"
CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=shakespeare_memory task.n_clients_per_round=10000 task.num_rounds=10 local_epochs=1 placement_policy=$policy flwr_address=$ip:6379"

echo "STARTING POLLEN SERVER at $node_1"
poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &

echo "STARTING POLLEN NODE MANAGER at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" --gres=gpu:1 -c 24 \
    poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &

echo "STARTING POLLEN NODE MANAGER at $node_2"
srun --nodes=1 --ntasks=1 -w "$node_2" \
    poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none

# How to use this script? Use what follows for a interactive job
# srun --nodelist mauao,ngongotaha --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:1 --partition=interactive bash lorenzo_slurm/slurm_pollen_multinode.sh
# Use what follows for a batch job
# sbatch lorenzo_slurm/slurm_pollen_multinode.sh