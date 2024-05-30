#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
#SBATCH -c 192
#SBATCH -w ruapehu
#SBATCH --job-name=benchmark_cpu_workers
#SBATCH --tasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
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

export CPU_CONCURRENCY=8
unset RUN_UUID
export RUN_UUID="fed-pollen_smalll_benchmark_${CPU_CONCURRENCY}_worker"
unset SAVE_PATH
. "$PROJECT_PATH"/llm_slurm/pollen_llm_small.sh
