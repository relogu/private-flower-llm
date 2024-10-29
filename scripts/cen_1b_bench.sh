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

BATCH_SIZE=64
TOTAL_STEPS=24800
WARMUP_STEPS=100
EVAL_FREQ=100
export RUN_UUID="cen-bench-1B-bs$BATCH_SIZE-$DATETIME"

export EXTERNAL_CONFIGS=""
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.max_duration=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.t_max=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.t_warmup=${WARMUP_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.scheduler.alpha_f=0.1"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.optimizer.lr=2.0e-4"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.global_train_batch_size=$BATCH_SIZE"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_train_microbatch_size=auto"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.precision=amp_bf16"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ~llm_config.fsdp_config" # DDP
# export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.fsdp_config.sharding_strategy=FULL_SHARD" # FSDP (full shard)
# export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.fsdp_config.sharding_strategy=SHARD_GRAD_OP" # FSDP (shard grad op)
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_learning_rate=1.0"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.strategy_kwargs.server_momentum=0.0"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS pollen.fit_collaborative=false"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS pollen.checkpoint=true"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.device_eval_batch_size=256"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_subset_num_batches=100"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_interval=${TOTAL_STEPS}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS fl.reset_optimizer=false"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.save_interval=${EVAL_FREQ}ba"
export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.eval_first=false"

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG="INFO"

bash $HOME/projects/flower_llm/scripts/centralised_training.sh 1B
