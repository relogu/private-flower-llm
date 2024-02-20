#!/bin/bash
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
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh "760M"
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
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Set Pollen and FL config
POLLEN_CONFIG="pollen.server_address='localhost:50743' pollen.refresh_period=5 fl.n_rounds=60"
#! Launch ServerWithPollen
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.server_with+pollen $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG pollen.saving_path=$SAVE_PATH 2>&1 | tee $SAVE_PATH/server.log &
#! Wait for 30 seconds. This is needed because of how the client connection behaves.
sleep 30
#! Launch NodeManager
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/node_manager.log &
#! Keep the pid and wait for it 
BACK_PID=$!
wait $BACK_PID