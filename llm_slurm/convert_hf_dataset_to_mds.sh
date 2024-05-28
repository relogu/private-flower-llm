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
#! Set/get the variables
SPLIT="full"
N_CLIENTS=8
DATASET="c4"
DATASET_SUBSET="en"
MOSAICML_DATA_ROOT="/local/scratch"
if [[ $# -eq 0 ]]; then
	echo "convert_hf_dataset_to_mds.sh: Using default values for all input arguments."
elif [[ $# -eq 1 ]]; then
	SPLIT=$1
elif [[ $# -eq 2 ]]; then
	SPLIT=$1
	N_CLIENTS=$2
elif [[ $# -eq 3 ]]; then
	SPLIT=$1
	N_CLIENTS=$2
	DATASET=$3
elif [[ $# -eq 4 ]]; then
	SPLIT=$1
	N_CLIENTS=$2
	DATASET=$3
	DATASET_SUBSET=$4
elif [[ $# -eq 5 ]]; then
	SPLIT=$1
	N_CLIENTS=$2
	DATASET=$3
	DATASET_SUBSET=$4
	MOSAICML_DATA_ROOT=$5
else
	echo "convert_hf_dataset_to_mds.sh: Invalid number of input arguments. Try 'bash convert_hf_dataset_to_mds.sh --help/-h' for more information."
	exit 1
fi
mkdir -p $MOSAICML_DATA_ROOT
if [[ $SPLIT == "full" ]]; then
	SPLIT_NAME="val train"
	echo "convert_hf_dataset_to_mds.sh: Default splits selected: $SPLIT_NAME."
else
	SPLIT_NAME=$SPLIT
	echo "convert_hf_dataset_to_mds.sh: The selected splits are $SPLIT_NAME."
fi
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
#! Set the data root
DATA_ROOT="$MOSAICML_DATA_ROOT/fed_$DATASET/c$N_CLIENTS"
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
poetry run python -m pollen_worker.dataset.convert_dataset_hf \
	--dataset $DATASET \
	--data_subset $DATASET_SUBSET \
	--splits $SPLIT_NAME \
	--out_root $DATA_ROOT \
	--compression zstd \
	--concat_tokens 2048 \
	--tokenizer EleutherAI/gpt-neox-20b \
	--eos_text '<|endoftext|>' \
	--num_workers $NUM_CPUS \
	--num_clients $N_CLIENTS
# --tokenizer_kwargs # add these if you want to pass additional kwargs to the tokenizer
# --no_wrap # set this if you want to wrap long sequences
# --bos_text # default
# --local # default
# --remote # default
# --shuffle # default
# --shuffle_seed # default

#! Remove the positional arguments
eval set --
