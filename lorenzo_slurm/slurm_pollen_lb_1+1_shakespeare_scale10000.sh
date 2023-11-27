#!/bin/bash
#SBATCH --nodelist=mauao,ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=PS11-lb-scale10k
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --dependency=afterany:78477

#! Need to force the nodes. Otherwise, the nodes might be allocated randomly.
#! Head node is `mauao`, 128.232.115.0
node_1="mauao"
node_2="ngongotaha"
ip="128.232.115.0"

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# activate the environment and go to the pollen_worker directory
cd /nfs-share/ls985/projects/pollen_worker
poetry shell


# Set the custom hydra arguments that will be passed to the server and the node manager
policy="lb"
echo "Using policy $policy"
CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=shakespeare_memory task.n_clients_per_round=10000 task.num_rounds=10 local_epochs=1 placement_policy=$policy flwr_address=$ip:6379"

echo "STARTING POLLEN SERVER at $node_1"
poetry run python -m pollen_worker.server_with+pollen $CUSTOM_HYDRA_ARGS &

echo "STARTING POLLEN NODE MANAGER at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" --gres=gpu:1 -c 8 \
    poetry run python -m pollen_worker.pure_sh_node_manager $CUSTOM_HYDRA_ARGS &

echo "STARTING POLLEN NODE MANAGER at $node_2"
srun --nodes=1 --ntasks=1 -w "$node_2" \
    poetry run python -m pollen_worker.pure_sh_node_manager $CUSTOM_HYDRA_ARGS

# How to use this script? Use what follows for a interactive job
# srun --nodelist mauao,ngongotaha --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:1 --partition=interactive bash lorenzo_slurm/slurm_pollen_multinode.sh
# Use what follows for a batch job
# sbatch lorenzo_slurm/slurm_pollen_multinode.sh