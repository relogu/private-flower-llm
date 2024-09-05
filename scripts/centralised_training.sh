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
export LLM_OPTIONS="$LLM_OPTIONS centralized.stream_id=null dataset=fed-c4 dataset/streams@dataset.train.streams=8_clients dataset/streams@dataset.val.streams=1_client_small dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4" # C4 - centralized, when stream_id=null streaming will be merged anyway, so any stream configuration is fine

#! Size specific optimization parameters
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=88000ba llm_config.scheduler.t_max=88000ba llm_config.scheduler.t_warmup=1000ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=4.0e-4" # DiLoCo - 75M
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=5000ba llm_config.scheduler.t_max=5000ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=6.0e-4" # MosaicML (+200ba) - 125M
export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=25000ba llm_config.scheduler.t_max=25000ba llm_config.scheduler.t_warmup=400ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=4.0e-4" # DisTrO
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=0ba llm_config.scheduler.t_max=0ba llm_config.scheduler.t_warmup=0ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=6.0e-4" # No training (Eval only)

#! Load a model from a checkpoint of type .pt (residing in the S3 bucket)
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.load_path=$CHECKPOINT_PATH"
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.load_path=/nfs-share/ls985/projects/flower_llm/centralised-760M-20240305_190707/ep0-ba17500-rank0.pt"  # Centralised 760M
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.load_path=/nfs-share/ls985/projects/flower_llm/centralised-1B-20240229_104204/ep0-ba25500-rank0.pt"  # Centralised 1B
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.load_path=/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/centralised-7B-20240724_190529_centralised/ep0-ba63900-rank0.pt" # Centralised 7B

#! Load a model from a checkpoint of type NDArrays
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=$CHECKPOINT_PATH"
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/fed-350M-2024505_100605/server/19/current_server_parameters.npz"  # Federated 350M  -- 1
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/fed-350M-20240505_100605/server/19/current_server_parameters.npz"  # Federated 350M  -- 2
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/fed-350M-20240506_204125/server/51/current_server_parameters.npz"  # Federated 350M  -- 3
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/fed_1B_long_checkpoint.npz"  # Federated 1B long run
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/fed_1B_short_checkpoint.npz"  # Federated 1B short run
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/flower_llm_checkpoints/fed-3B-20240702_141112/server/25/current_server_parameters.npz"  # Federated 3B
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=/nfs-share/ls985/projects/flower_llm/fed_7B_checkpoint.npz"  # Federated 7B
# export LLM_OPTIONS="$LLM_OPTIONS pretrained_model_path=s3://checkpoints/matrix-125M-p-tle/server/10/current_server_parameters.npz" # Test

#! General training parameters
export LLM_OPTIONS="$LLM_OPTIONS llm_config.save_interval=1000ba"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.console_log_interval=100ba"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=false"           # Disable evaluation first
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_interval=250ba"        # Local evaluation interval
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_subset_num_batches=-1" # Evaluate the entire validation set
# export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.compile_config={}"                         # Compiles the model with default parameters
export LLM_OPTIONS="$LLM_OPTIONS llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP" # Shard only the gradient operation -- most of the times convenient when GPUs are poorly connected
# export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.fsdp_config"                                # Removes FSDP
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.optimizer_monitor"             # Clears OptimizerMonitor (not supported when using DeepSpeed)
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.lr_monitor"                    # Clears LRMonitor
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.memory_monitor"                # Clears MemoryMonitor
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.runtime_estimator"             # Clears RuntimeEstimator
export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.activation_monitor_full_model" # Clears ActivationMonitorFullModel
export LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=512"              # DisTrO 8 devices batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=128"              # DisTrO 2 devices batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=256"              # DisTrO 4 devices batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=64"               # DisTrO single device batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=8"           # DisTrO 4xA40 devices microbatch size -- w/ and w/o compilation -- no FSDP
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=auto"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_fp16" # Standard Automatic Mixed Precision float16 context -- use with < Ampere GPUs
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_bf16" # Standard Automatic Mixed Precision brainfloat16 precision context -- use with >= Ampere GPUs
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.precision=amp_fp8 ++llm_config.model.fc_type=te"

#! Model parameters
export LLM_OPTIONS="$LLM_OPTIONS llm_config.model.n_heads=8 llm_config.model.n_layers=16 ++llm_config.model.attn_config.rope=true ++llm_config.model.attn_config.rope_impl=dail ++llm_config.model.attn_config.rope_theta=10000" # DisTrO model
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.model.attn_config.attn_impl=torch"                                                                                                                                                   # Use PyTorch's attention implementation

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
# export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.eval_gauntlet.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=empty eval_gauntlet_config=empty" # Empty gauntlet

#! DeepSpeed configuration file
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.deepspeed_config_file='/nfs-share/ls985/projects/flower_llm/flower_llm/conf/deepspeed_config/empty.json'"
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.deepspeed_config_file=null"

echo "centralised_training.sh: LLM_OPTIONS=$LLM_OPTIONS"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $EXTERNAL_CONFIGS $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True APPOINTED_CUDA_DEVICE=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$(uuidgen) poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $PROJECT_PATH/flower_llm/centralised_train.py hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it
BACK_PID=$!
wait $BACK_PID
