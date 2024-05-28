#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
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
. $PROJECT_PATH/llm_slurm/set_llm_config.sh "7B"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export POLLEN_SAVE_PATH="$PROJECT_PATH/checkpoints/$DATETIME"
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
	export RUN_UUID="fed-7B-$DATETIME"
fi
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
	export SAVE_PATH="s3://checkpoints/$RUN_UUID"
fi
mkdir -p $POLLEN_SAVE_PATH
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS - 1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! S3 communication stack settings
MINIO_COMM_STACK_OPTIONS="use_s3_comm=true s3_comm_config.bucket_name=checkpoints"
#! Set Pollen and FL config
N_LOCAL_STEPS=500
POLLEN_CONFIG="pollen.server_address='[::]:50760' run_uuid=$RUN_UUID pollen.refresh_period=20 fl.n_rounds=176"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.checkpoint=true pollen.saving_path=$SAVE_PATH llm_config.save_folder=$SAVE_PATH llm_config.save_overwrite=true pollen.n_nodes=1 pollen.fit_collaborative=true"
POLLEN_CONFIG="$POLLEN_CONFIG pollen.resume_round=-1 pollen.restore_run_uuid=null"
POLLEN_CONFIG="$POLLEN_CONFIG fl.rescale_global_model=false fl.rescale_momentum_vector=false fl.server_learning_rate=0.1 fl.server_momentum=0.9"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.scheduler.t_max=63900ba llm_config.scheduler.t_warmup=100ba llm_config.scheduler.alpha_f=0.1 llm_config.optimizer.lr=1.2e-4"
POLLEN_CONFIG="$POLLEN_CONFIG llm_config.save_interval=${N_LOCAL_STEPS}ba llm_config.console_log_interval=${N_LOCAL_STEPS}ba llm_config.local_steps=${N_LOCAL_STEPS}ba"
POLLEN_CONFIG="$POLLEN_CONFIG ~llm_config.fsdp_config" # Use DDP only
# POLLEN_CONFIG="$POLLEN_CONFIG ++llm_config.fsdp_config.use_orig_params=false"
#! Launch ServerWithPollen
GRPC_VERBOSITY=debug HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.launch_pollen_server $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/server.log &
#! Wait for 30 seconds. This is needed because of how the client connection behaves.
sleep 30
#! Launch NodeManager
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
GRPC_VERBOSITY=debug CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/node_manager.log &
#! Keep the pid of the NodeManager
BACK_PID=$!
# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
#! Wait for the NodeManager to finish
wait $BACK_PID
