#!/bin/bash
#!SBATCH --job-name=hetero-test
#SBATCH --job-name=PF13-mw-2000
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
# # activate the environment and go to the pollen_worker directory
# cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate


# Set the custom hydra arguments that will be passed to the server and the node manager
# for policy in "lb" "bu" "rr"; do
for policy in "lb"; do
    run_uuid="PF-13-$policy-mw-2000-$timestamp"
    echo "Using policy $policy"
    CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=flair task.n_clients_per_round=2000 task.num_rounds=250 local_epochs=2 placement_policy=$policy flwr_address=$ip:7382"

    echo "STARTING POLLEN SERVER at $node_1"
    GRPC_VERBOSITY=debug poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &
    sleep 15
    echo "STARTING POLLEN NODEMANAGER at $node_1"
    GRPC_VERBOSITY=debug srun --het-group=0 \
        poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS cap_num_workers_per_gpu=1 hydra/job_logging=none hydra/hydra_logging=none &
    echo "STARTING POLLEN NODEMANAGER at $node_2"
    GRPC_VERBOSITY=debug srun --het-group=1 \
        poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS cap_num_workers_per_gpu=5 hydra/job_logging=none hydra/hydra_logging=none 
done
