#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"
# Get the current date and time
DATETIME=$(date '+%Y%m%d_%H%M%S')

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "slurm_submit.mauao: Error parsing options" >&2
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
echo "slurm_submit.mauao: PROJECT_PATH=$PROJECT_PATH"

LOCAL_BATCH_SIZE=8
LOCAL_STEPS=512
CPR=8
TOTAL_STEPS=$((24800 * 256 / (LOCAL_BATCH_SIZE)))
WARMUP_STEPS=$((100 * 256 / (LOCAL_BATCH_SIZE)))
N_ROUNDS=$((TOTAL_STEPS / (LOCAL_STEPS)))
EVAL_FREQ=$((N_ROUNDS / 155))
export RUN_UUID="fed-1B-${CPR}cpr${LOCAL_STEPS}-bs$LOCAL_BATCH_SIZE-$DATETIME"

export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.max_duration=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.t_max=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.t_warmup=${WARMUP_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.alpha_f=0.1"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.optimizer.lr=2.0e-4"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.global_train_batch_size=$LOCAL_BATCH_SIZE"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_learning_rate=1.0"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_momentum=0.0"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS pollen.fit_collaborative=false"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_clients_per_round=$CPR"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS pollen.checkpoint=true"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_rounds=$N_ROUNDS"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.local_steps=${LOCAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_eval_batch_size=256"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_subset_num_batches=100"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_interval=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.reset_optimizer=false"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.save_interval=${LOCAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.n_total_clients=$CPR"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.eval_fl=$EVAL_FREQ"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS use_s3_comm=false"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_bf16"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.fsdp_config"

# Resume options
# export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS  pollen.resume_round=-1"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG="INFO"

bash $HOME/projects/flower_llm/scripts/photon_llm_1B.sh
