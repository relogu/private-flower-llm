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
. "$PROJECT_PATH"/scripts/set_llm_config.sh "1B"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="fed-1B-$DATETIME"
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
MINIO_COMM_STACK_OPTIONS="use_s3_comm=true s3_comm_config.bucket_name=checkpoints"
#! Set Pollen and FL config
N_LOCAL_STEPS=500
POLLEN_CONFIG="run_uuid=$RUN_UUID pollen.refresh_period=20 fl.n_rounds=176"
# NOTE: set dataset
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache"
mkdir -p $DATASET_CACHE_DIR

#! Dataset configuration
POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4 dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4 dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4" # C4
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=8_clients dataset/streams@dataset.val.streams=8_clients"                      # 8 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=64_clients dataset/streams@dataset.val.streams=64_clients"                    # 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=32_clients dataset/streams@dataset.val.streams=32_clients"                    # 32 clients

#! Photon configuration
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true"            # Enable checkpointing
POLLEN_CONFIG="$POLLEN_CONFIG pollen.saving_path=$SAVE_PATH"     # Save path for Photon Server
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_folder=$SAVE_PATH" # Save path for PhotonLLM Client
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_overwrite=true"    # Overwrite the existing checkpoint for PhotonLLM Client
POLLEN_CONFIG="$POLLEN_CONFIG pollen.n_nodes=1"                  # Number of nodes in the Photon federation
POLLEN_CONFIG="$POLLEN_CONFIG pollen.fit_collaborative=false"    # Non-collaborative training on local GPUs
POLLEN_CONFIG="$POLLEN_CONFIG pollen.fit_collaborative=true"     # Collaborative training on local GPUs
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=-1"            # Resume round for Photon Server
POLLEN_CONFIG="$POLLEN_CONFIG pollen.restore_run_uuid=null"      # Restore run UUID for Photon Server

#! FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=32"     # Number of total clients in the federation
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_clients_per_round=32" # Number of clients per round
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_rounds=50"            # Total number of rounds

#! ServerOpt
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_name=NESTOROV"                                                          # Server optimizer strategy
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_kwargs.server_learning_rate=0.7 fl.strategy_kwargs.server_momentum=0.9" # DiLoCo parameters
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_kwargs.server_learning_rate=1.0 fl.strategy_kwargs.server_momentum=0.0" # FedAvg

#! ClientOpt (AdamW + Cosine LR scheduler) parameters
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.t_max=25000ba llm_config.scheduler.t_warmup=400ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=4.0e-4" # DisTrO setting

#! Training hyperparameters
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba" # Save checkpoint interval
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.console_log_interval=100ba"       # Console log interval
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.local_steps=${N_LOCAL_STEPS}ba"   # Local steps
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_first=true"               # Enable evaluation at the first step
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_interval=250ba"           # Local evaluation interval
export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_subset_num_batches=-1"    # Evaluate the entire validation set
POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config"                     # Use DDP only
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.precision=amp_fp16"               # Fastest precision context when using DDP
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.global_train_batch_size=64"       # DisTrO single device batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.global_train_batch_size=256"   # DisTrO 4 devices batch size
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=auto"
export LLM_OPTIONS="$LLM_OPTIONS llm_config.device_train_microbatch_size=8" # DisTrO 4xA40 devices microbatch size w/ compilation
POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.compile_config={}"               # Compile the model at Trainer initialization

#! Evaluation gauntlet configuration
export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.eval_gauntlet.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
export LLM_OPTIONS="$LLM_OPTIONS icl_tasks_config=empty eval_gauntlet_config=empty"                                                                                                                                           # Empty gauntlet

#! DeepSpeed configuration file
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.deepspeed_config_file='/nfs-share/ls985/projects/flower_llm/flower_llm/conf/deepspeed_config/empty.json'" # Empty DeepSpeed configuration file (default)
export LLM_OPTIONS="$LLM_OPTIONS ++llm_config.deepspeed_config_file=null"                                                                               # Disable DeepSpeed

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address '[::]:50758' --fleet-api-address '[::]:51758' 2>&1 | tee "$POLLEN_SAVE_PATH"/superlink.log &

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app flower_llm.client_app:app --insecure --superlink '[::]:51758' --persist-client 2>&1 | tee "$POLLEN_SAVE_PATH"/node_manager.log &
#! Keep the pid of the NodeManager
BACK_PID=$!

#! Launch ServerWithPollen as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app flower_llm.server_app:app --insecure --superlink '[::]:50758' 2>&1 | tee "$POLLEN_SAVE_PATH"/server.log &

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
#! Wait for the NodeManager to finish
wait $BACK_PID
