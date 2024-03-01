#!/bin/bash
#! Moving to the project fo lder
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
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh "small"
#! Set `DATA_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_data_config.sh
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export POLLEN_SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
export SAVE_PATH="s3://checkpoints"
mkdir -p $POLLEN_SAVE_PATH
#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Set Pollen and FL config
POLLEN_CONFIG="pollen.server_address='localhost:50735' run_uuid=fed-small-$DATETIME pollen.refresh_period=100 fl.n_rounds=50 llm_config.scheduler.t_max=10000ba llm_config.scheduler.t_warmup=0ba llm_config.save_overwrite=true"
#! Launch ServerWithPollen
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.launch_pollen_server $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG pollen.saving_path=$POLLEN_SAVE_PATH 2>&1 | tee $POLLEN_SAVE_PATH/server.log &
#! Wait for 30 seconds. This is needed because of how the client connection behaves.
sleep 30
#! Launch NodeManager
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/node_manager.log &
#! Keep the pid and wait for it 
BACK_PID=$!
wait $BACK_PID
 