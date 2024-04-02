#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"
echo "PROJECT_PATH=$PROJECT_PATH"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
  echo "Error parsing options" >&2
  exit 1
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -p | --project_path )
      PROJECT_PATH="$2"; shift 2 ;;
    -- )
      shift; break ;;
    * )
      break ;;
  esac
done
#! Check if there's an input argument
if [[ $# -eq 0 ]]; then
    echo "No input argument supplied."
    exit 1
fi
#! Moving to the project folder
cd $PROJECT_PATH
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
    echo "Assuming the script is executing in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $PROJECT_PATH/llm_slurm/install_hpc_env.sh
else
    echo "Assuming the script is executing NOT in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $PROJECT_PATH/llm_slurm/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. $PROJECT_PATH/llm_slurm/set_llm_config.sh $1
shift
#! Set `DATA_CONFIG` environment variable
. $PROJECT_PATH/llm_slurm/set_llm_data_config.sh
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export SAVE_PATH="$PROJECT_PATH/checkpoints/$DATETIME"
mkdir -p $SAVE_PATH
#! Set `LLM_OPTIONS` environment variable
. $PROJECT_PATH/llm_slurm/set_llm_options.sh
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Additional settings specific for the current testing
TESTING_OPTIONS=""
#! Test VirtualLLMClient
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
RUN_UUID=chiappe APPOINTED_CUDA_DEVICE=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.clients.virtual_llm_client $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $TESTING_OPTIONS is_test=true hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/virtual_llm_client.log 
