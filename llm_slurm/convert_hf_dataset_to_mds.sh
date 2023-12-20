#!/bin/bash
#! Moving to the project folder
cd $HOME/projects/pollen_worker
if ! [[ $(hostname) == 'mauao' ]] && ! [[ $(hostname) == *'fluidstack'* ]]; then
    echo "Assuming the script is executing in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_hpc_env.sh
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate
#! Set the variables
DATASET="c4"
DATASET_SUBSET="en" # Can also be "all"
N_CLIENTS=10
#! Check whether the MOSAICML_DATA_ROOT has been set or not
if [ -z "${MOSAICML_DATA_ROOT}" ]; then
    echo "MOSAICML_DATA_ROOT is not set. Exiting..."
    exit 1
else
    echo "MOSAICML_DATA_ROOT is set to '${MOSAICML_DATA_ROOT}'"
fi
#! Check whether the MOSAICML_DATA_ROOT directory exists or not
if [ -d "$MOSAICML_DATA_ROOT" ]; then
    echo "MOSAICML_DATA_ROOT directory exists"
else
    echo "MOSAICML_DATA_ROOT directory does not exist. Creating..."
    mkdir -p $MOSAICML_DATA_ROOT
fi
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
    --splits train_small val_small \
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
