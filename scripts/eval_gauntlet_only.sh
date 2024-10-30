#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
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
MODEL_SIZE="$1"
echo "centralised_training.sh: MODEL_SIZE=$MODEL_SIZE"

#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
	echo "centralised_training.sh: Illegal number of parameters."
	echo "Usage: centralised_training.sh <llm_model_config>"
	exit 1
fi
echo "centralised_training.sh: PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "centralised_training.sh: Assuming the script is executing in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. "$PROJECT_PATH"/scripts/install_hpc_env.sh
else
	echo "centralised_training.sh: Assuming the script is executing NOT in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. "$PROJECT_PATH"/scripts/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. "$PROJECT_PATH"/scripts/set_llm_config.sh "$MODEL_SIZE"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="centralised-$MODEL_SIZE-$DATETIME"
fi
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="s3://checkpoints/$RUN_UUID"
fi
export POLLEN_SAVE_PATH="$PROJECT_PATH/runs/$RUN_UUID/$DATETIME"
mkdir -p "$POLLEN_SAVE_PATH"

#! Set dataset related configurations
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache"
mkdir -p $DATASET_CACHE_DIR

#! Dataset configuration
export LLM_OPTIONS="$LLM_OPTIONS dataset=fed-c4"                                     # Dataset name
export LLM_OPTIONS="$LLM_OPTIONS dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4" # Path of the local cache for the training dataset
export LLM_OPTIONS="$LLM_OPTIONS dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4"   # Path of the local cache for the evaluation dataset
export LLM_OPTIONS="$LLM_OPTIONS dataset/streams@dataset.train.streams=8_clients"    # Stream configuration for the training dataset -- 8 clients
export LLM_OPTIONS="$LLM_OPTIONS dataset/streams@dataset.val.streams=8_clients"      # Stream configuration for the training dataset --  8 clients
export LLM_OPTIONS="$LLM_OPTIONS centralized.stream_id=null"                         # ID of the stream to use only for centralized training (they are concatenated if null)
export LLM_OPTIONS="$LLM_OPTIONS centralized.eval_only=true"                         # Only executes the initial evaluation
export LLM_OPTIONS="$LLM_OPTIONS +wte_parameters_path=null"                          # Path to the WTE parameters

#! ClientOpt (AdamW + Cosine LR scheduler) parameters
export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=0ba"       # No training (Eval only)
export LLM_OPTIONS="$LLM_OPTIONS llm_config.scheduler.t_max=0ba"    # No training (Eval only)
export LLM_OPTIONS="$LLM_OPTIONS llm_config.scheduler.t_warmup=0ba" # No training (Eval only)

#! Load a model from a checkpoint of type .pt (residing in the S3 bucket)
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.load_path=$CHECKPOINT_PATH"

#! Load a model from a checkpoint of type NDArrays
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=$CHECKPOINT_PATH"

#! General training parameters
export LLM_OPTIONS="$LLM_OPTIONS llm_config.save_interval=200ba"                # Save checkpoint interval
export LLM_OPTIONS="$LLM_OPTIONS llm_config.console_log_interval=100ba"         # Console log interval
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true"                    # Enable evaluation at the first step
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_interval=250ba"                # Local evaluation interval
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_subset_num_batches=-1"         # Evaluate the entire validation set
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.fsdp_config"                       # Removes FSDP
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.device_eval_microbatch_size=auto" # Automatic microbatch size for evaluation
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_eval_batch_size=256"         # Evaluation batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=auto"  # Automatic microbatch size for training
export LLM_OPTIONS="$LLM_OPTIONS llm_config.model.attn_config.attn_impl=torch"  # Shut down flash attention

#! Getting visible GPUs
if [[ $(nvidia-smi -L) == *'No devices'* ]]; then
	echo "No NVIDIA devices found."
	N_GPUS=0
elif [[ $(nvidia-smi -L) == *'not found'* ]]; then
	echo "nvidia-smi not present."
	N_GPUS=0
else
	N_GPUS=$(nvidia-smi -L | wc -l)
fi
if [ "$N_GPUS" -eq 0 ]; then
	echo "No GPUs found. Exiting."
	CUDA_VISIBLE_DEVICES=""
else
	CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
fi
echo "centralised_training.sh: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! Additional config
export LLM_OPTIONS="$LLM_OPTIONS run_uuid=$RUN_UUID"

#! Evaluation gauntlet configuration
export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
# export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=empty eval_gauntlet_config=empty"  # Exclude MosaicML Gauntlet

#! DeepSpeed configuration file
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.deepspeed_config_file=null" # Disable DeepSpeed

echo "centralised_training.sh: LLM_OPTIONS=$LLM_OPTIONS"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$DATETIME"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $EXTERNAL_CONFIGS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1
APPOINTED_CUDA_DEVICE=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$(uuidgen) poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $PROJECT_PATH/flower_llm/centralised_train.py hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it
BACK_PID=$!
wait $BACK_PID
