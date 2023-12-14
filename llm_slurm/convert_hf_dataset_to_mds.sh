#!/bin/bash


#! Moving to the project folder
cd /nfs-share/$USER/projects/pollen_worker
# #! Activate Poetry environment
# poetry shell
#! Add the appropriate CUDA version to the paths
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://127.0.0.1:9000'
export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Check the output of `nvcc -V`
nvcc -V
DATASET="c4"
DATASET_SUBSET="en" # Can also be "all"
N_CLIENTS=10
DATA_ROOT="/local/scratch/fed_$DATASET/c$N_CLIENTS"
mkdir -p $DATA_ROOT
#! Get info about resources available
NUM_CPUS=$SLURM_CPUS_PER_TASK
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
