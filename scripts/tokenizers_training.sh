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

# Vocab size 50257

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names wikipedia --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_wikipedia_train_50257_$DATETIME.log

# Running - pyo3_runtime.PanicException: likelihood is NAN. Input sentence may be too long.
# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names arxiv --splits train --vocab_size 50257 --truncate_num_samples 500000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_arxiv_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names gutenberg --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_gutenberg_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names hackernews --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_hackernews_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names pubmedcentral --splits train --vocab_size 50257 --truncate_num_samples 2000000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_pubmedcentral_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names freelaw --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_freelaw_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names philpapers --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_philpapers_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names dmmathematics --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_dmmathematics_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names enronemails --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_enronemails_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names europarl --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_europarl_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names nihexporter --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_nihexporter_train_50257_$DATETIME.log

# Running - Exception: The vocabulary is not large enough to contain all chars
# We gave up -> we'll use a pretrained one from HF -> couldn't find anything there, we'll use the EleutherAI one
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names github --splits train --vocab_size 50257 --truncate_num_samples 100000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_github_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names pubmedabstract --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_pubmedabstract_train_50257_$DATETIME.log

# Running - pyo3_runtime.PanicException: likelihood is NAN. Input sentence may be too long.
# We gave up -> we'll use that from hacker news
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names stackexchange --splits train --vocab_size 50257 --truncate_num_samples 500000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_stackexchange_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer_from_tokenized.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --names usptobackgrounds --splits train --vocab_size 50257 2>&1 | tee "$SAVE_PATH"/tokenizer_training_usptobackgrounds_train_50257_$DATETIME.log
