#!/bin/bash
#SBATCH --job-name=PG13-llb-scale2k
#SBATCH --output=%x-%j.out
#SBATCH --partition=normal
#SBATCH --gres=gpu:a40:1
#SBATCH -c 10
#SBATCH hetjob
#SBATCH --partition=normal
#SBATCH --gres=gpu:rtx2080:3
#SBATCH -c 24

#! Need to force the nodes. Otherwise, the nodes might be allocated randomly.
#! Head node is `mauao`, 128.232.115.0
node_1="mauao"
node_2="ngongotaha"
ip="128.232.115.0"
echo "The script is being executed in node $(hostname)"

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# Activate the environment and go to the pollen_worker directory
# cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate


# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=google_speech task.n_clients_per_round=2000 task.num_rounds=100 local_epochs=1 placement_policy=llb flwr_address=$ip:6481"

echo "STARTING POLLEN SERVER at $node_1"
poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &

sleep 15

echo "STARTING POLLEN NODE MANAGER at $node_1"
srun --het-group=0 \
    poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &

echo "STARTING POLLEN NODE MANAGER at $node_2"
srun --het-group=1 \
    poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none 
