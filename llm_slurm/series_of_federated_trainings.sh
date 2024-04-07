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

unset RUN_UUID
export RUN_UUID="fed-1B-20240322_221900-fix-nesto"
unset SAVE_PATH
. $PROJECT_PATH/llm_slurm/pollen_llm_1B.sh

# unset RUN_UUID
# export RUN_UUID="fed-75M-reset-20240323_183000-fix-nesto"
# unset SAVE_PATH
# . $PROJECT_PATH/llm_slurm/pollen_llm_75M.sh

# unset RUN_UUID
# export RUN_UUID="fed-125M-reset-20240323_213000-fix-nesto"
# unset SAVE_PATH
# . $PROJECT_PATH/llm_slurm/pollen_llm_125M.sh
