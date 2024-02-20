#!/bin/bash
#! Check if there's an input argument
if [[ $# -eq 0 ]]; then
    echo "No input argument supplied."
    exit 1
fi
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
    echo "Assuming the script is executing in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_hpc_env.sh
else
    echo "Assuming the script is executing NOT in the CSD3."
    # Adding CUDA paths to environment variables
    export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    # Check CUDA
    nvcc -V
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh $1
shift

#! Set `DATA_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_data_config.sh

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
mkdir -p $SAVE_PATH

#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh

#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}  # Default to 0 if not set
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
IFS=',' read -ra DEVICES <<< "$CUDA_VISIBLE_DEVICES"  # Split on comma

#! Additional settings specific for the current testing
# TESTING_OPTIONS=""
# TESTING_OPTIONS="pollen.server_address='localhost:50735' llm_config.console_log_interval=512ba pollen.n_nodes=$N_GPUS"
TESTING_OPTIONS="pollen.server_address='localhost:50735' llm_config.console_log_interval=512ba pollen.n_nodes=1"
# TESTING_OPTIONS="pollen.server_address='localhost:50735' llm_config.console_log_interval=512ba pollen.n_nodes=1 fl.n_clients_per_round=4"
# TESTING_OPTIONS="pollen.server_address='localhost:50735' run_uuid='chiappe1' fl.n_clients_per_round=5 llm_config.console_log_interval=50ba"
# TESTING_OPTIONS="pollen.server_address='localhost:50736' run_uuid='chiappe2' fl.n_clients_per_round=10 llm_config.console_log_interval=100ba"
# TESTING_OPTIONS="pollen.server_address='localhost:50737' run_uuid='chiappe3' fl.n_clients_per_round=5 llm_config.console_log_interval=50ba"

#! Launch ServerWithPollen
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.server_with+pollen $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $TESTING_OPTIONS pollen.saving_path=$SAVE_PATH 2>&1 | tee $SAVE_PATH/server.log &

#! Wait for 30 seconds. This is needed because of how the client connection behaves.
sleep 30

#! Launch NodeManager
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
NCCL_DEBUG="INFO" CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $TESTING_OPTIONS is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/node_manager_$DEVICE.log &

#! Keep the pid and wait for it 
BACK_PID=$!
wait $BACK_PID
