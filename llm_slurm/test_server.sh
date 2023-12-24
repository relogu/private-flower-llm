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
    export DATA_TMP_DIR="$HOME/rds/rds-ndl32-camlsys-DNlKPrIaphU/datasets"
else
    echo "Assuming the script is executing NOT in the CSD3."
    export DATA_TMP_DIR="$HOME/tmp"
    # Remove shared memories of the user if they exist
    find /dev/shm -name '*pollen*' -type f -delete 
    find /dev/shm -name '*_locals' -type f -delete
fi
mkdir -p $DATA_TMP_DIR
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh $1

#! Set `DATA_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_data_config.sh

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
mkdir -p $SAVE_PATH

#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh

#! Test ServerWithPollen
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.server_with+pollen $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG is_test=true hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/server.log 
