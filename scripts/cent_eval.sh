#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "Error parsing options" >&2
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

echo "PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "Assuming the script is executing in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. "$PROJECT_PATH"/scripts/install_hpc_env.sh
else
	echo "Assuming the script is executing NOT in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. "$PROJECT_PATH"/scripts/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. "$PROJECT_PATH"/scripts/set_llm_config.sh $MODEL_SIZE
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="adapt-$MODEL_SIZE-$DATETIME"
fi


if [ -z "$TOKENIZER" ]; then
	export TOKENIZER="EleutherAI/gpt-neox-20b"
fi



if [ -z "$RESTORE_RUN_UUID" ]; then
	export RESTORE_RUN_UUID=null
fi

if [ -z "$RESUME_ROUND" ]; then
	export RESUME_ROUND=null
fi

if [ -z "$N_LOCAL_STEPS" ]; then
	export N_LOCAL_STEPS=1100
fi

if [ -z "$STREAM" ]; then
	export STREAM="1_clients_balanced"
fi

if [ -z "$EVAL_INTERVAL" ]; then
	export EVAL_INTERVAL=200
fi

if [ -z "$SAVE_INTERVAL" ]; then
	export SAVE_INTERVAL=100
fi

if [ -z "$RESTORE_RUN_CENT_UUID" ]; then
	export RESTORE_RUN_CENT_UUID=null
fi

if [ -z "$RESTORE_RUN_CENT_BATCHES" ]; then
	export RESTORE_RUN_CENT_BATCHES=null
fi

if [ -z "$RESTORE_RUN_CENT_EPOCH" ]; then
	export RESTORE_RUN_CENT_EPOCH=null
fi

if [ -z "$DATASET" ]; then
	export DATASET="fed-c4"
fi

if [ -z "$VOCAB_SIZE" ]; then
	export VOCAB_SIZE=50368
fi

if [ -z "$RESIZE_VOCAB" ]; then
	export RESIZE_VOCAB=50368
fi

if [ -z "$TOTAL_ROUNDS" ]; then
	export TOTAL_ROUNDS=21
fi

if [ -z "$EVAL_SUBSET_NUM_BATCHES" ]; then
	export EVAL_SUBSET_NUM_BATCHES=-1
fi

if [ -z "$PERSONALIZED_KEYS" ]; then
	export PERSONALIZED_KEYS=[model.transformer.wte.weight,model.transformer.wpe.weight]
fi

if [ -z "$RANDOM_KEYS" ]; then
	export RANDOM_KEYS=[model.transformer.wte.weight,model.transformer.wpe.weight]
fi

if [ -z "$START_LR" ]; then
	export START_LR=6.0e-4
fi

if [ -z "$TASKS" ]; then
	export TASKS=tasks_v0.3
fi

if [ -z "$EVAL_GAUNT" ]; then
	export EVAL_GAUNT=eval_gauntlet_v0.3
fi

#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="s3://checkpoints/$RUN_UUID"
fi
export POLLEN_SAVE_PATH="$PROJECT_PATH/runs/$RUN_UUID/$DATETIME"
mkdir -p "$POLLEN_SAVE_PATH"


export LOAD_PATH="s3://checkpoints/$RESTORE_RUN_CENT_UUID/ep$RESTORE_RUN_CENT_EPOCH-ba$RESTORE_RUN_CENT_BATCHES-rank0.pt"

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
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! S3 communication stack settings
MINIO_COMM_STACK_OPTIONS="use_s3_comm=true s3_comm_config.bucket_name=checkpoints"
#! Set Pollen and FL config

POLLEN_CONFIG="run_uuid=$RUN_UUID pollen.refresh_period=100 pollen.fit_collaborative=true"
# NOTE: set dataset
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache_eval"
mkdir -p $DATASET_CACHE_DIR
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4-c4 dataset/streams@dataset.train.streams=8_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4-c4"
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4 dataset/streams@dataset.train.streams=16_clients dataset/streams@dataset.val.streams=16_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4" # C4 - 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset=$DATASET dataset/streams@dataset.train.streams=$STREAM dataset/streams@dataset.val.streams=$STREAM  dataset.train.root_local=$DATASET_CACHE_DIR/$DATASET dataset.val.root_local=$DATASET_CACHE_DIR/$DATASET"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true pollen.saving_path=$SAVE_PATH llm_config.save_folder=$SAVE_PATH llm_config.save_overwrite=true pollen.n_nodes=1"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=$RESUME_ROUND pollen.restore_run_uuid=$RESTORE_RUN_UUID pollen.restore_cent_run_uuid=$RESTORE_RUN_CENT_UUID pollen.restore_cent_run_batches=$RESTORE_RUN_CENT_BATCHES pollen.copy_client_checkpoints=false"
# POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=8 fl.n_clients_per_round=8 fl.n_rounds=176" # FL setting
export COMPOSER_FAIL_ON_VOCAB_MISMATCH=0
export ALLOW_EMBEDDING_RESIZING=0

POLLEN_CONFIG="$POLLEN_CONFIG fl.eval_fl=false fl.reset_dataset_state=true fl.reset_timestamp=true fl.reset_checkpoint=true fl.remap_tokens=false fl.resize_vocab=$RESIZE_VOCAB fl.n_clients_per_round=1 fl.n_total_clients=1  fl.n_rounds=$TOTAL_ROUNDS fl.random_init_freq=1 fl.personalized_keys=$PERSONALIZED_KEYS fl.random_keys=$RANDOM_KEYS" # fl.unfrozen_layers=[model.transformer.wte.weight,model.transformer.wpe.weight]
#  fl.personalized_keys=[model.transformer.wte.weight,model.transformer.wpe.weight]                                                                            # FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_name=FEDAVG"
export LLM_OPTIONS="$LLM_OPTIONS +llm_config.model.allow_embedding_resizing=false +llm_config.model.fail_on_vocab_mismatch=false llm_config.model.vocab_size=$VOCAB_SIZE llm_config.max_duration=${N_LOCAL_STEPS}ba llm_config.scheduler.t_max=${N_LOCAL_STEPS}ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=$START_LR" # MosaicML (+200ba) - 125M
export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=$TASKS eval_gauntlet_config=$EVAL_GAUNT eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${SAVE_INTERVAL}ba llm_config.console_log_interval=100ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true llm_config.eval_interval=${EVAL_INTERVAL}ba llm_config.eval_subset_num_batches=${EVAL_SUBSET_NUM_BATCHES} llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP ++llm_config.device_eval_microbatch_size=auto llm_config.tokenizer.name=${TOKENIZER}"
# POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config" # Used DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.use_orig_params=false"

export CENT_OPTIONS=" centralized.eval_only=true centralized.split_eval=true llm_config.load_path=$LOAD_PATH"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/E_$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS $LLM_OPTIONS $CENT_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log


#! Launch centralised training script
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1
APPOINTED_CUDA_DEVICE=$CUDA_VISIBLE_DEVICES CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 RUN_UUID=$RUN_UUID poetry run composer --world_size $N_GPUS --node_rank 0 --master_addr 127.0.0.1 $PROJECT_PATH/flower_llm/centralised_train.py hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/centralised_train.log &
#! Keep the pid and wait for it
BACK_PID=$!
wait $BACK_PID