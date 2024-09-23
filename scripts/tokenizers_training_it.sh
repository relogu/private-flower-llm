#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"
SAVE_PATH="$PROJECT_PATH/tokenizers_logs"
mkdir -p "$SAVE_PATH"

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
#! Export the endpoint of the S3 object store, using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name it --truncate_num_samples 3000000 --splits train --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_it_train_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name it --truncate_num_samples 3000000 --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_it_train_50257_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name it --truncate_num_samples 3000000 --splits train --vocab_size 250112 2>&1 | tee "$SAVE_PATH"/tokenizer_training_it_train_250112_$DATETIME.log
