#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=pollen_worker_bench
#SBATCH --partition=interactive
#SBATCH --tasks-per-node=1


timestamp=$(date +%Y-%m-%d_%H%M%S)
source /nfs-share/ls985/anaconda3/bin/activate ray111-test
cd /nfs-share/ls985/pollen_worker

# Clean the shared memory objects
python clean_memory.py

# Launch the server
python server_with+pollen.py num_rounds=100 & # >> server_$timestamp.out 2>&1 &
# Waiting for the server to start
sleep 15
# Launch the node manager
python pure_sh_node_manager.py # >> node_manager_$timestamp.out 2>&1

# Clean the shared memory objects
python clean_memory.py

# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh