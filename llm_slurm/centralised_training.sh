#!/bin/bash
#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
    echo "Illegal number of parameters."
    echo "Usage: centralised_training.sh <llm_model_config>"
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
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh $1
#! Set `DATA_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_data_config.sh "full" false false
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export POLLEN_SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
    export SAVE_PATH="s3://checkpoints/centralised-$1-$DATETIME"
fi
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
    export RUN_UUID="centralised-$1-$DATETIME"
fi
mkdir -p $POLLEN_SAVE_PATH
#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh
export LLM_OPTIONS="llm_config.save_interval=100ba llm_config.console_log_interval=1ba llm_config.save_folder=$SAVE_PATH llm_config.save_num_checkpoints_to_keep=1"
echo "LLM_OPTIONS=$LLM_OPTIONS"
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Additional config
export LLM_OPTIONS="$LLM_OPTIONS run_uuid=$RUN_UUID"
#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$(uuidgen) poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $HOME/projects/pollen_worker/pollen_worker/centralised_train.py $EXTERNAL_CONFIGS $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it 
BACK_PID=$!
wait $BACK_PID
