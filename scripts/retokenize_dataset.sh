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
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name wikipedia --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925012052_v-50257_l-2048_d-pile_n-wikipedia_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_wikipedia_train_50257_$DATETIME.log

# Running
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name arxiv --splits train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240926004213_v-50257_l-2048_d-pile_n-arxiv_s-train --num_workers 20 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_arxiv_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name gutenberg --splits train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924191746_v-50257_l-2048_d-pile_n-gutenberg_s-train --num_workers 20 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_gutenberg_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name hackernews --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924201903_v-50257_l-2048_d-pile_n-hackernews_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_hackernews_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name pubmedcentral --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240927033759_v-50257_l-2048_d-pile_n-pubmedcentral_s-train --num_workers 20 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_pubmedcentral_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name freelaw --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924233850_v-50257_l-2048_d-pile_n-freelaw_s-train --num_workers 20 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_freelaw_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name philpapers --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924223340_v-50257_l-2048_d-pile_n-philpapers_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_philpapers_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name dmmathematics --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925013513_v-50257_l-2048_d-pile_n-dmmathematics_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_dmmathematics_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name enronemails --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924183552_v-50257_l-2048_d-pile_n-enronemails_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_enronemails_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name europarl --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925005842_v-50257_l-2048_d-pile_n-europarl_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_europarl_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name nihexporter --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925073338_v-50257_l-2048_d-pile_n-nihexporter_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_nihexporter_train_50257_$DATETIME.log

# We gave up -> we'll use a pretrained one from HF -> couldn't find anything there, we'll use the EleutherAI one
# DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name github --splits val train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_github_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name pubmedabstract --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925111822_v-50257_l-2048_d-pile_n-pubmedabstract_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_pubmedabstract_train_50257_$DATETIME.logog

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name stackexchange --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240924201903_v-50257_l-2048_d-pile_n-hackernews_s-train --num_workers 20 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_stackexchange_train_50257_$DATETIME.log

# Done
DATETIME=$(date '+%Y%m%d_%H%M%S') python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/retokenize_dataset.py --name usptobackgrounds --splits val train --encode_tokenizer /nfs-share/ls985/projects/flower_llm/tokenizer_20240925100410_v-50257_l-2048_d-pile_n-usptobackgrounds_s-train 2>&1 | tee "$SAVE_PATH"/retokenize_dataset_usptobackgrounds_train_50257_$DATETIME.log
