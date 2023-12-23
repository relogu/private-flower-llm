#!/bin/bash
#! Setting the helper
if [[ $1 = "--help" ]] || [[ $1 = "-h" ]]; then
    echo "Usage: bash convert_hf_dataset_to_mds.sh <split> <n_clients> <dataset> <dataset_subset> <data_root>."
    echo -e "\t<split>: 'small' or 'full'. Default: 'full'"
    echo -e "\t<n_clients>: integer. Default: 10"
    echo -e "\t<dataset>: 'c4' or 'pile'. Default: 'c4'"
    echo -e "\t<dataset_subset>: 'en' or 'all'. Default: 'en'"
    echo -e "\t<data_root>: path. Default: '/local/scratch'"
    echo -e "\tExample: bash convert_hf_dataset_to_mds.sh small 10 c4 en"
    exit 1
fi
#! Set/get the variables
SPLIT="full"
N_CLIENTS=10
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
    echo "Invalid number of input arguments. Try 'bash convert_hf_dataset_to_mds.sh --help/-h' for more information."
    exit 1
fi
mkdir -p $MOSAICML_DATA_ROOT
if [[ $SPLIT == "small" ]]; then
    SPLIT_NAME="val_small train_small"
elif [[ $SPLIT == "full" ]]; then
    SPLIT_NAME="val train"
else
    echo "Invalid split. Try 'bash convert_hf_dataset_to_mds.sh --help/-h' for more information."
    exit 1
fi
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
    echo "Assuming the script is executing in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_hpc_env.sh
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate
#! Set the data root
DATA_ROOT="$MOSAICML_DATA_ROOT/fed_$DATASET/c$N_CLIENTS"
echo "Creating the partition data root directory: $DATA_ROOT"
mkdir -p $DATA_ROOT
#! Get info about CPU resources available
if [ -z "${SLURM_CPUS_PER_TASK}" ]; then
    export NUM_CPUS=$(nproc --all)
else
    export NUM_CPUS=$SLURM_CPUS_PER_TASK
fi
echo "Number of CPU cores available: $NUM_CPUS"
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
