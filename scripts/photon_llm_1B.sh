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
MINIO_COMM_STACK_OPTIONS="use_s3_comm=false"                                                # Don't use S3 communication stack
MINIO_COMM_STACK_OPTIONS="use_s3_comm=true"                                                 # Use S3 communication stack
MINIO_COMM_STACK_OPTIONS="$MINIO_COMM_STACK_OPTIONS s3_comm_config.bucket_name=checkpoints" # S3 bucket name
#! Set Pollen and FL config
N_LOCAL_STEPS=500
# NOTE: set dataset
export DATASET_CACHE_DIR="/local/scratch/flower_llm/dataset_cache"
mkdir -p $DATASET_CACHE_DIR

#! Dataset configuration
POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4"                                     # Dataset name
POLLEN_CONFIG="$POLLEN_CONFIG dataset.train.root_local=$DATASET_CACHE_DIR/fed-c4" # Path of the local cache for the training dataset
POLLEN_CONFIG="$POLLEN_CONFIG dataset.val.root_local=$DATASET_CACHE_DIR/fed-c4"   # Path of the local cache for the evaluation dataset
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=32_clients"   # Stream configuration for the training dataset -- 32 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.val.streams=32_clients"     # Stream configuration for the training dataset --  32 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=8_clients"    # Stream configuration for the training dataset -- 8 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.val.streams=8_clients"      # Stream configuration for the training dataset --  8 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.train.streams=64_clients"   # Stream configuration for the training dataset -- 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset/streams@dataset.val.streams=64_clients"     # Stream configuration for the training dataset --  64 clients
POLLEN_CONFIG="$POLLEN_CONFIG centralized.stream_id=null"                         # ID of the stream to use only for centralized training (they are concatenated if null)

#! Photon configuration
POLLEN_CONFIG="$POLLEN_CONFIG run_uuid=$RUN_UUID"                # Run UUID
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=false"           # Disable checkpointing
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true"            # Enable checkpointing
POLLEN_CONFIG="$POLLEN_CONFIG pollen.saving_path=$SAVE_PATH"     # Save path for Photon Server
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_folder=$SAVE_PATH" # Save path for PhotonLLM Client
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_overwrite=true"    # Overwrite the existing checkpoint for PhotonLLM Client
POLLEN_CONFIG="$POLLEN_CONFIG pollen.n_nodes=1"                  # Number of nodes in the Photon federation
POLLEN_CONFIG="$POLLEN_CONFIG pollen.fit_collaborative=true"     # Collaborative training on local GPUs
POLLEN_CONFIG="$POLLEN_CONFIG pollen.fit_collaborative=false"    # Non-collaborative training on local GPUs
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=-1"            # Resume round fro latest for the Photon Server
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=null"          # Start from scratch
POLLEN_CONFIG="$POLLEN_CONFIG pollen.refresh_period=60"          # PhotonLLM Client workers refresh period
POLLEN_CONFIG="$POLLEN_CONFIG pollen.restore_run_uuid=null"      # Restore run UUID for Photon Server

#! FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=64"    # Number of total clients in the federation
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_clients_per_round=8" # Number of clients per round
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_rounds=50"           # Total number of rounds

#! ServerOpt
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_name=NESTOROV"                                                          # Server optimizer strategy
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_kwargs.server_learning_rate=0.7 fl.strategy_kwargs.server_momentum=0.9" # DiLoCo parameters
POLLEN_CONFIG="$POLLEN_CONFIG fl.strategy_kwargs.server_learning_rate=1.0 fl.strategy_kwargs.server_momentum=0.0" # FedAvg

#! ClientOpt (AdamW + Cosine LR scheduler) parameters
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.t_max=25000ba"  # Total number of training steps
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.t_warmup=100ba" # Number of warmup steps
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.alpha_f=0.1"    # Decay factor for the cosine scheduler
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.optimizer.lr=2.0e-4"      # Learning rate for the optimizer
POLLEN_CONFIG="$POLLEN_CONFIG fl.reset_optimizer=false"            # Keep the local optimizer every round
POLLEN_CONFIG="$POLLEN_CONFIG fl.reset_optimizer=true"             # Reset local optimizer every round

#! Training hyperparameters
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba" # Save checkpoint interval
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.console_log_interval=100ba"       # Console log interval
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.local_steps=${N_LOCAL_STEPS}ba"   # Local steps
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.eval_first=true"                  # Enable evaluation at the first step
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.eval_interval=250ba"              # Local evaluation interval
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.eval_subset_num_batches=-1"       # Evaluate the entire validation set
# POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config"                                  # Use DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.compile_config={}"                           # Compile the model at Trainer initialization
POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.sharding_strategy=FULL_SHARD" # Shard only the gradient operation -- most of the times convenient when GPUs are poorly connected
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.precision=amp_fp16"                         # Fastest precision context when using DDP
POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.device_eval_microbatch_size=auto"         # Automatic microbatch size for evaluation
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.device_eval_batch_size=128"                 # Evaluation batch size
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.global_train_batch_size=64"                 # DisTrO single device batch size (one client simulates one device)
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.device_train_microbatch_size=8"             # 4xA40 devices microbatch size -- w/ and w/o compilation -- no FSDP
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.device_train_microbatch_size=auto"          # Automatic microbatch size

#! Model parameters
# POLLEN_CONFIG="$POLLEN_CONFIG llm_config.model.n_heads=8 llm_config.model.n_layers=16 ++llm_config.model.attn_config.rope=true ++llm_config.model.attn_config.rope_impl=dail ++llm_config.model.attn_config.rope_theta=10000" # DisTrO model
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.model.attn_config.attn_impl=torch" # Use PyTorch's attention implementation
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.model.attn_config.attn_impl=flash" # Use Flash attention implementation

#! Evaluation gauntlet configuration
# POLLEN_CONFIG="$POLLEN_CONFIG icl_tasks_config=tasks_v0.3 eval_gauntlet_config=eval_gauntlet_v0.3 eval_gauntlet_config.destination_dir=$DATASET_CACHE_DIR/eval icl_tasks_config.root_dir=$DATASET_CACHE_DIR" # Complete MosaicML Gauntlet
POLLEN_CONFIG="$POLLEN_CONFIG icl_tasks_config=empty eval_gauntlet_config=empty" # Empty gauntlet

#! DeepSpeed configuration file
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.deepspeed_config_file='/nfs-share/ls985/projects/flower_llm/flower_llm/conf/deepspeed_config/empty.json'" # Empty DeepSpeed configuration file (default)
POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.deepspeed_config_file=null" # Disable DeepSpeed

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
export TMPDIR="/local/scratch/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS $EXTERNAL_CONFIGS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address '[::]:50758' --fleet-api-address '[::]:51758' 2>&1 | tee "$POLLEN_SAVE_PATH"/superlink.log &
SUPERLINK_PID=$!
sleep 5

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app flower_llm.client_app:app --insecure --superlink '[::]:51758' --persist-client 2>&1 | tee "$POLLEN_SAVE_PATH"/node_manager.log &
#! Keep the pid of the ClientApp
CLIENTAPP_PID=$!

#! Launch ServerWithPollen as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app flower_llm.server_app:app --insecure --superlink '[::]:50758' 2>&1 | tee "$POLLEN_SAVE_PATH"/server.log &
#! Keep the pid of the ServerApp
SERVERAPP_PID=$!

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
#! Wait for the ServerApp to finish
wait $SERVERAPP_PID
#! Kill the ClientApp and Superlink
kill $CLIENTAPP_PID
kill $SUPERLINK_PID
