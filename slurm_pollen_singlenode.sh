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

python server.py num_rounds=1 & # >> server_$timestamp.out 2>&1 &
sleep 10
python node_manager.py # >> node_manager_$timestamp.out 2>&1
