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
. "$PROJECT_PATH"/scripts/set_llm_config.sh "350M"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="fed-350M-$DATETIME"
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
N_LOCAL_STEPS=500
POLLEN_CONFIG="run_uuid=$RUN_UUID pollen.refresh_period=100 pollen.fit_collaborative=false"
# NOTE: set dataset
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache"
mkdir -p $DATASET_CACHE_DIR
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4-c4 dataset/streams@dataset.train.streams=8_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4-c4"
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4 dataset/streams@dataset.train.streams=16_clients dataset/streams@dataset.val.streams=16_clients dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4" # C4 - 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset=icl_datasets dataset/streams@dataset.train.streams=the_pile_16_clients_unique dataset/streams@dataset.val.streams=the_pile_16_clients_unique dataset.train.root_local=$DATASET_CACHE_DIR/icl_datasets dataset.val.root_local=$DATASET_CACHE_DIR/icl_datasets "
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true pollen.saving_path=$SAVE_PATH llm_config.save_folder=$SAVE_PATH llm_config.save_overwrite=true pollen.n_nodes=1"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=-1 pollen.restore_run_uuid=null"
# POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=8 fl.n_clients_per_round=8 fl.n_rounds=176" # FL setting
export COMPOSER_FAIL_ON_VOCAB_MISMATCH=0
export ALLOW_EMBEDDING_RESIZING=0
POLLEN_CONFIG="$POLLEN_CONFIG fl.eval_fl=null fl.n_clients_per_round=4 fl.n_total_clients=16  fl.n_rounds=27 fl.personalized_layers=[transformer.wte.weight,transformer.wpe.weight]"
#  fl.personalized_keys=[model.transformer.wte.weight,model.transformer.wpe.weight]                                                                            # FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_name=FEDAVG" # DiLoCo
export LLM_OPTIONS="$LLM_OPTIONS llm_config.max_duration=13500ba llm_config.scheduler.t_max=13500ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=3.0e-4"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba llm_config.console_log_interval=100ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true llm_config.eval_interval=500ba llm_config.eval_subset_num_batches=50 llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP"
# POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config" # Used DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.use_orig_params=false"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS $LLM_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address '[::]:58752' --fleet-api-address '[::]:56752' 2>&1 | tee "$POLLEN_SAVE_PATH"/superlink.log &
SUPERLINK_PID=$!
sleep 5

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app flower_llm.client_app:app --insecure --superlink '[::]:56752' --persist-client 2>&1 | tee "$POLLEN_SAVE_PATH"/node_manager.log &
#! Keep the pid of the NodeManager
CLIENTAPP_PID=$!

#! Launch ServerWithPollen as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app flower_llm.server_app:app --insecure --superlink '[::]:58752' 2>&1 | tee "$POLLEN_SAVE_PATH"/server.log &
SERVERAPP_PID=$!

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
#! Wait for the ServerApp to finish
wait $SERVERAPP_PID
#! Kill the ClientApp and Superlink
kill $CLIENTAPP_PID
kill $SUPERLINK_PID