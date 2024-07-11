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
. "$PROJECT_PATH"/scripts/set_llm_config.sh "7B"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="fed-7B-$DATETIME"
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
N_LOCAL_STEPS=10
POLLEN_CONFIG="run_uuid=$RUN_UUID pollen.refresh_period=20 fl.n_rounds=176"
# NOTE: set dataset
# POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-the_pile dataset/streams@dataset.train.streams=the_pile_64_clients dataset/streams@dataset.val.streams=the_pile_64_clients"  # The Pile - 8 split 8 - 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG dataset=fed-c4 dataset/streams@dataset.train.streams=64_clients dataset/streams@dataset.val.streams=64_clients" # C4 - 64 clients
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true pollen.saving_path=$SAVE_PATH llm_config.save_folder=$SAVE_PATH llm_config.save_overwrite=true pollen.n_nodes=1 pollen.fit_collaborative=true"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=-1 pollen.restore_run_uuid=null"
POLLEN_CONFIG="$POLLEN_CONFIG fl.n_total_clients=64 fl.n_clients_per_round=4 fl.n_rounds=200" # FL setting
POLLEN_CONFIG="$POLLEN_CONFIG fl.server_learning_rate=0.7 fl.server_momentum=0.9"             # Server-side optimizer
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.t_max=63900ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=1.2e-4"
# POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba llm_config.console_log_interval=${N_LOCAL_STEPS}ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba llm_config.console_log_interval=100ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.eval_first=true llm_config.eval_interval=5ba llm_config.eval_subset_num_batches=1"
# POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config" # Use DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.compile_config={}"  # Compile the model at Trainer initialization
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.use_orig_params=false"

#! Set `TMPDIR` that is used for storing the temporary files for caching the dataset (not the dataset cache though)
# export TMPDIR="/tmp/flower_llm/$RUN_UUID/$DATETIME"
export TMPDIR="/local/scratch/tmp/flower_llm/$RUN_UUID"
mkdir -p "$TMPDIR"

#! Run Hydra resolver
HYDRA_FULL_ERROR=1 poetry run python -m flower_llm.hydra_resolver $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee "$POLLEN_SAVE_PATH"/hydra_resolver.log

#! Start a Superlink
# GRPC_VERBOSITY=debug
poetry run flower-superlink --insecure --driver-api-address '[::]:50760' --fleet-api-address '[::]:51760' 2>&1 | tee "$POLLEN_SAVE_PATH"/superlink.log &

#! Launch NodeManager as a SuperNode - ClientApp
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
# NCCL_DEBUG=INFO NCCL_NVB_DISABLE=1 NCCL_NVLS_ENABLE=0 # For running on Lambda Labs faulty machine
# GRPC_VERBOSITY=debug
CUDA_LAUNCH_BLOCKING=1 poetry run flower-client-app flower_llm.node_manager.node_manager:client_app --insecure --superlink '[::]:51760' --persist-client 2>&1 | tee "$POLLEN_SAVE_PATH"/node_manager.log &
#! Keep the pid of the NodeManager
BACK_PID=$!

#! Launch ServerWithPollen as a ServerApp
# GRPC_VERBOSITY=debug
poetry run flower-server-app flower_llm.launch_pollen_server:server_app --insecure --superlink '[::]:50760' 2>&1 | tee "$POLLEN_SAVE_PATH"/server.log &

# Enable CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
#! Wait for the NodeManager to finish
wait $BACK_PID
