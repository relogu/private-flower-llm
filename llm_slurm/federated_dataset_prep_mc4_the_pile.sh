#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
	echo "convert_hf_dataset_to_mds.sh: Error parsing options" >&2
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

#! Setting the helper
if [[ $1 == "--help" ]] || [[ $1 == "-h" ]]; then
	echo "Usage: bash convert_hf_dataset_to_mds.sh <splitS> <n_clients> <dataset> <dataset_subset> <data_root>."
	echo -e "\t<split>: list containing a combination of 'small_val', 'small_train', 'val', 'train'. Default: 'val train'"
	echo -e "\t<n_clients>: integer. Default: 8"
	echo -e "\t<dataset>: 'c4' or 'pile'. Default: 'c4'"
	echo -e "\t<dataset_subset>: 'en' or 'all'. Default: 'en'"
	echo -e "\t<data_root>: path. Default: '/local/scratch'"
	echo -e "\tExample: bash convert_hf_dataset_to_mds.sh small 8 c4 en"
	echo -e "\tExample: bash convert_hf_dataset_to_mds.sh full 16 c4 en"
	echo -e "\tExample: bash convert_hf_dataset_to_mds.sh full 32 c4 en"
	exit 1
fi

PRESET="mc4"

#! Set/get the variables
SPLIT="full"
TOKENIZER="google/mt5-base"
EOS_TOKEN=""
DATASET="allenai/c4"
DATASET_SUBSET="es bg da de el fi fr hi it no ru sr sv  th uk vi zh en"

DATA_ROOT="/local/scratch/aai30"

if [[ $PRESET == "mc4" ]]; then
	TOKENIZER="google/mt5-base"
	EOS_TOKEN=""
	DATASET="allenai/c4"
	DATASET_SUBSET="es bg da de el fi fr hi it no ru sr sv  th uk vi zh en"
	DATA_ROOT="$DATA_ROOT/fed_mc4"
elif [[ $PRESET == "the_pile" ]]; then
	TOKENIZER="EleutherAI/gpt-neox-20b"
	EOS_TOKEN="<|endoftext|>"
	DATASET="monology/pile-uncopyrighted"
	DATASET_SUBSET="default"
	DATA_ROOT="$DATA_ROOT/fed_the_pile_c8"
fi

if [[ $# -eq 0 ]]; then
	echo "convert_hf_dataset_to_mds.sh: Using default values for all input arguments."
elif [[ $# -eq 1 ]]; then
	SPLIT=$1
elif [[ $# -eq 2 ]]; then
	SPLIT=$1
	TOKENIZER=$2
elif [[ $# -eq 3 ]]; then
	SPLIT=$1
	TOKENIZER=$2
	EOS_TOKEN=$3
elif [[ $# -eq 4 ]]; then
	SPLIT=$1
	TOKENIZER=$2
	EOS_TOKEN=$3
	DATASET=$4
elif [[ $# -eq 5 ]]; then
	SPLIT=$1
	TOKENIZER=$2
	EOS_TOKEN=$3
	DATASET=$4
	DATASET_SUBSET=$5
elif [[ $# -eq 6 ]]; then
	SPLIT=$1
	TOKENIZER=$2
	EOS_TOKEN=$3
	DATASET=$4
	DATASET_SUBSET=$5
	DATA_ROOT=$6
else
	echo "convert_hf_dataset_to_mds.sh: Invalid number of input arguments. Try 'bash convert_hf_dataset_to_mds.sh --help/-h' for more information."
	exit 1
fi
mkdir -p $DATA_ROOT
if [[ $SPLIT == "full" ]]; then
	SPLIT_NAME="val train"
	echo "convert_hf_dataset_to_mds.sh: Default splits selected: $SPLIT_NAME."
else
	SPLIT_NAME=$SPLIT
	echo "convert_hf_dataset_to_mds.sh: The selected splits are $SPLIT_NAME."
fi

IFS=' ' read -r -a DATASET_SUBSET_ARRAY <<<"$DATASET_SUBSET"

echo "convert_hf_dataset_to_mds.sh: PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd $PROJECT_PATH
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "convert_hf_dataset_to_mds.sh: Assuming the script is executing in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. $PROJECT_PATH/llm_slurm/install_hpc_env.sh
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate
echo "convert_hf_dataset_to_mds.sh: Creating the partition data root directory: $DATA_ROOT"
mkdir -p $DATA_ROOT
#! Get info about CPU resources available
if [ -z "${SLURM_CPUS_PER_TASK}" ]; then
	export NUM_CPUS=$(nproc --all)
else
	export NUM_CPUS=$SLURM_CPUS_PER_TASK
fi
echo "convert_hf_dataset_to_mds.sh: Number of CPU cores available: $NUM_CPUS"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Execute the command
for SUBSET in "${DATASET_SUBSET_ARRAY[@]}"; do
	LOCAL_DATA_ROOT="$DATA_ROOT/$SUBSET"
	if [ -n "$EOS_TOKEN" ]; then
		poetry run python -m pollen_worker.dataset.convert_and_partition_dataset \
			--dataset $DATASET \
			--data_subset $SUBSET \
			--num_clients 8 \
			--splits $SPLIT_NAME \
			--out_root $LOCAL_DATA_ROOT \
			--compression zstd \
			--concat_tokens 2048 \
			--tokenizer $TOKENIZER \
			--eos_text $EOS_TOKEN \
			--num_workers $NUM_CPUS

	else
		poetry run python -m pollen_worker.dataset.convert_and_partition_dataset \
			--dataset $DATASET \
			--data_subset $SUBSET \
			--num_clients 8 \
			--splits $SPLIT_NAME \
			--out_root $LOCAL_DATA_ROOT \
			--compression zstd \
			--concat_tokens 2048 \
			--tokenizer $TOKENIZER \
			--num_workers $NUM_CPUS

	fi
done
# --tokenizer_kwargs # add these if you want to pass additional kwargs to the tokenizer
# --no_wrap # set this if you want to wrap long sequences
# --bos_text # default
# --local # default
# --remote # default
# --shuffle # default
# --shuffle_seed # default

#! Remove the positional arguments
eval set --
