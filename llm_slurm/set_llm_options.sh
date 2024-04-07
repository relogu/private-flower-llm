#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"

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
#! Set the run configuration
STEPS="500"
#! Test configuration: just 10 steps during training
# STEPS="2"
#! Configuration: local checkpointing, one local checkpoint, no callbacks
export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=$SAVE_PATH llm_config.local_steps=${STEPS}ba llm_config.save_num_checkpoints_to_keep=1"
#! Configuration: remove default callbacks except for the LR monitor
# export LLM_OPTIONS="$LLM_OPTIONS ~llm_config.callbacks.speed_monitor ~llm_config.callbacks.memory_monitor ~llm_config.callbacks.runtime_estimator"
#! Configuration: reduce number of eval batches
# export LLM_OPTIONS="$LLM_OPTIONS llm_config.eval_subset_num_batches=2"

#! Remove the positional arguments
eval set --
