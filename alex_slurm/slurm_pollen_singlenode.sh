#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=pollen_worker_bench
#SBATCH --tasks-per-node=1

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd /nfs-share/aai30/projects/pollen_worker/
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate


# Clean the shared memory objects
 

# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=shakespeare_memory task.num_rounds=100 flwr_address=127.0.0.1:1043"

# Launch the server, uncomment the end of the line if you what separed output logs.
poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none & # >> server_$timestamp.out 2>&1 &

# Launch the node manager, uncomment the end of the line if you what separed output logs.
# NOTE that the `nsys` command is used to profile the node manager. It uses Nsight Systems software for NVIDIA.
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu \
# -f true --cudabacktrace=true \
# --osrt-threshold=10000 -x true \
# -o ${timestamp}_nvidia_nsight \
poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS # >> node_manager_$timestamp.out 2>&1

# This might become necessary, kept just in case.
# # Clean the shared memory objects
#   

# How to use this script? Use what follows for a interactive job
# srun -w ngongotaha -c 8 --gres=gpu:1 --partition=interactive bash slurm_pollen_singlenode.sh
# Use what follows for a batch job
# sbatch slurm_pollen_singlenode.sh