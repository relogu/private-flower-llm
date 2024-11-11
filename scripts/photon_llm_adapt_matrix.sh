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

if [ -z "$DATASET" ]; then
	export DATASET="fed-c4"
fi

if [ -z "$USE_WANDB" ]; then
	export USE_WANDB=true
fi

if [ -z "$TOKENIZER" ]; then
	export TOKENIZER="EleutherAI/gpt-neox-20b"
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
	export PERSONALIZED_KEYS="[model.transformer.wte.weight,model.transformer.wpe.weight]"
fi

if [ -z "$RANDOM_KEYS" ]; then
	export RANDOM_KEYS="[model.transformer.wte.weight,model.transformer.wpe.weight]"
fi

if [ -z "$START_LR" ]; then
	export START_LR=6.0e-4
fi

#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="s3://checkpoints/$RUN_UUID"
fi
export POLLEN_SAVE_PATH="$PROJECT_PATH/runs/$RUN_UUID/$DATETIME"
mkdir -p "$POLLEN_SAVE_PATH"
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
MINIO_COMM_STACK_OPTIONS="use_s3_comm=false s3_comm_config.bucket_name=checkpoints"
#! Set Pollen and FL config

POLLEN_CONFIG="run_uuid=$RUN_UUID pollen.refresh_period=100 pollen.fit_collaborative=true use_wandb=$USE_WANDB"
# NOTE: set dataset
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache"
mkdir -p $DATASET_CACHE_DIR
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4-c4 dataset/streams@dataset.train.streams=8_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4-c4"
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4 dataset/streams@dataset.train.streams=16_clients dataset/streams@dataset.val.streams=16_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4" # C4 - 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset=$DATASET dataset/streams@dataset.train.streams=$STREAM dataset/streams@dataset.val.streams=$STREAM  dataset.train.root_local=$DATASET_CACHE_DIR/$DATASET dataset.val.root_local=$DATASET_CACHE_DIR/$DATASET"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true pollen.saving_path=$SAVE_PATH llm_config.save_folder=$SAVE_PATH llm_config.save_overwrite=true pollen.n_nodes=1"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=$RESUME_ROUND pollen.restore_run_uuid=$RESTORE_RUN_UUID pollen.restore_cent_run_uuid=$RESTORE_RUN_CENT_UUID pollen.restore_cent_run_batches=$RESTORE_RUN_CENT_BATCHES pollen.copy_client_checkpoints=false"
# POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=8 fl.n_clients_per_round=8 fl.n_rounds=176" # FL setting
export COMPOSER_FAIL_ON_VOCAB_MISMATCH=0
export ALLOW_EMBEDDING_RESIZING=0

POLLEN_CONFIG="$POLLEN_CONFIG fl.eval_fl=null fl.reset_dataset_state=true fl.reset_timestamp=true fl.reset_checkpoint=true fl.resize_vocab=$RESIZE_VOCAB fl.use_unigram_metrics=true fl.n_clients_per_round=1 fl.n_total_clients=1  fl.n_rounds=$TOTAL_ROUNDS fl.random_init_freq=1 fl.personalized_layers=$PERSONALIZED_KEYS fl.random_layers=$RANDOM_KEYS" # fl.unfrozen_layers=[model.transformer.wte.weight,model.transformer.wpe.weight]
#  fl.personalized_keys=[model.transformer.wte.weight,model.transformer.wpe.weight]                                                                            # FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_name=FEDAVG"
export LLM_OPTIONS="$LLM_OPTIONS +llm_config.model.allow_embedding_resizing=false +llm_config.model.fail_on_vocab_mismatch=false llm_config.model.vocab_size=$VOCAB_SIZE llm_config.max_duration=${N_LOCAL_STEPS}ba llm_config.scheduler.t_max=${N_LOCAL_STEPS}ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=$START_LR" # MosaicML (+200ba) - 125M
export LLM_OPTIONS="$LLM_OPTIONS eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${SAVE_INTERVAL}ba llm_config.console_log_interval=100ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true llm_config.eval_interval=${EVAL_INTERVAL}ba llm_config.eval_subset_num_batches=${EVAL_SUBSET_NUM_BATCHES} llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP ++llm_config.device_eval_microbatch_size=auto llm_config.tokenizer.name=${TOKENIZER}" # Automatic microbatch size for evaluation"
# POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config" # Used DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.use_orig_params=false"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS $LLM_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address '[::]:53762' --fleet-api-address '[::]:58792' 2>&1 | tee "$POLLEN_SAVE_PATH"/superlink.log &
SUPERLINK_PID=$!
sleep 5

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app flower_llm.client_app:app --insecure --superlink '[::]:58792' --persist-client 2>&1 | tee "$POLLEN_SAVE_PATH"/node_manager.log &
#! Keep the pid of the NodeManager
CLIENTAPP_PID=$!

#! Launch ServerWithPollen as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app flower_llm.server_app:app --insecure --superlink '[::]:53762' 2>&1 | tee "$POLLEN_SAVE_PATH"/server.log &
SERVERAPP_PID=$!

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
#! Wait for the ServerApp to finish
wait $SERVERAPP_PID
#! Kill the ClientApp and Superlink
kill $CLIENTAPP_PID
kill $SUPERLINK_PID
