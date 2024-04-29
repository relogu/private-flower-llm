#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
	echo "centralised_training.sh: Error parsing options" >&2
	exit 1
fi

eval set -- "$OPTIONS"

while true; do
	case "$1" in
	-p | --project_path)
		PROJECT_PATH="$2"
		shift 2
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done
MODEL_SIZE=$1
echo "centralised_training.sh: MODEL_SIZE=$MODEL_SIZE"

#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
	echo "centralised_training.sh: Illegal number of parameters."
	echo "Usage: centralised_training.sh <llm_model_config>"
	exit 1
fi
echo "centralised_training.sh: PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd $PROJECT_PATH
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "centralised_training.sh: Assuming the script is executing in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. $PROJECT_PATH/llm_slurm/install_hpc_env.sh
else
	echo "centralised_training.sh: Assuming the script is executing NOT in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. $PROJECT_PATH/llm_slurm/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. $PROJECT_PATH/llm_slurm/set_llm_config.sh $MODEL_SIZE
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export POLLEN_SAVE_PATH="$PROJECT_PATH/checkpoints/$DATETIME"
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="centralised-$MODEL_SIZE-$DATETIME"
fi
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="s3://checkpoints/$RUN_UUID"
fi
mkdir -p $POLLEN_SAVE_PATH
#! Set `LLM_OPTIONS` environment variable
export LLM_OPTIONS="$LLM_OPTIONS dataset=c4 dataset/streams@dataset.train.streams=centralised dataset/streams@dataset.val.streams=centralised"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.save_interval=100ba llm_config.console_log_interval=100ba llm_config.save_folder=$SAVE_PATH"
echo "centralised_training.sh: LLM_OPTIONS=$LLM_OPTIONS"
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
echo "centralised_training.sh: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Additional config
export LLM_OPTIONS="$LLM_OPTIONS run_uuid=$RUN_UUID"
#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$(uuidgen) poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $PROJECT_PATH/pollen_worker/centralised_train.py $EXTERNAL_CONFIGS $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it
BACK_PID=$!
wait $BACK_PID
