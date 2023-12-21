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
    echo "Assuming the script is executing not in the CSD3."
    export DATA_TMP_DIR="$HOME/tmp"
    # Remove shared memories of the user if they exist
    find /dev/shm -name '*pollen*' -type f -delete 
    find /dev/shm -name '*_locals' -type f -delete
fi
mkdir -p $DATA_TMP_DIR
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate
#! Export the endpoint of the S3 object store
export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Set data paths
# DATA_VERSION="small"
DATA_VERSION="full"
IS_LOCAL=false
# IS_LOCAL=true
if [[ "$DATA_VERSION" == "small" ]]; then
    if [[ "$IS_LOCAL" == true ]]; then
        DATA_CONFIG="llm_config.data_local=/local/scratch/small-c4 llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    else
        DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://small-c4-dataset llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    fi
else
    if [[ "$IS_LOCAL" == true ]]; then
        DATA_CONFIG="llm_config.data_local=/local/scratch/c4 llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    else
        DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://c4-dataset llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    fi
fi

#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh $1

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
mkdir -p $SAVE_PATH

#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh

#! Test VirtualLLMClient
poetry run python -m pollen_worker.clients.virtual_llm_client $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG is_test=true hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/node_manager.log 
