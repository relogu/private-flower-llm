#!/bin/bash
#SBATCH -c 10
#SBATCH -w mauao
#SBATCH --gres=gpu:1
#SBATCH --job-name=conc-PC1
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#SBATCH -t 04:00:00
#SBATCH --dependency=afterany:2131

# \activate the environment and go to the pollen_worker directory
# cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

# Set the custom hydra arguments that will be passed to the server and the node manager
# for policy in "lb" "llb" "rr"; do
for n_workers in $(seq 1 16); do
    policy="rr"
    # Get the timestamp and the unique run id
    timestamp=$(date +%Y-%m-%d_%H%M%S)
    run_uuid=conc-est-cifar-1gpu-a40-$n_workers-$timestamp
    export RUN_UUID=$run_uuid
    export CONCURRENCY=$n_workers
    echo "Using policy $policy with $n_workers workers. Run UUID: $run_uuid"
    CUSTOM_HYDRA_ARGS="num_nodes=1 cap_num_workers_per_gpu=$CONCURRENCY run_uuid=$run_uuid task=cifar10 task.num_rounds=100 placement_policy=$policy flwr_address=127.0.0.1:9481"

    # Launch the server.
    GRPC_VERBOSITY=debug poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none  &
    sleep 30

    # Launch the node manager.
    GRPC_VERBOSITY=debug poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &
    # Retrieve the PID of the last process launched
    node_manager_pid=$!

    # Lauch GPU monitoring
    nvidia-smi --query-gpu=index,utilization.gpu,memory.total,memory.used,memory.free,timestamp --format=csv,noheader,nounits --filename=gpu-$RUN_UUID-$CONCURRENCY.csv --loop-ms=100 &
    # Retrieve the PID of the last process launched
    gpu_monitoring_pid=$!
    # Launch CPU, MEMORY, and DISK monitoring
    ./monitoring.sh $node_manager_pid &
    # Retrieve the PID of the last process launched
    monitoring_pid=$!
    # Wait for the node manager to finish
    wait $node_manager_pid
    # Kill the GPU monitoring
    kill $gpu_monitoring_pid
    # Kill the CPU, MEMORY, and DISK monitoring
    kill $monitoring_pid
done
