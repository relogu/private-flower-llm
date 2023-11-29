#!/bin/bash


#! Moving to the project folder
cd /nfs-share/ls985/projects/llm-foundry
#! Activate Poetry environment
poetry shell
#! Add the appropriate CUDA version to the path
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
#! Set data paths
DATA_ROOT=/home/ls985/c4
DATA_ROOT_MDS=/home/ls985/mds-c4
DATA_ROOT_SMALL=/home/ls985/my-copy-c4
DATA_ROOT_SMALL_MDS=/home/ls985/my-mds-copy-c4
#! Set config paths
CONFIG_MPT_125M=scripts/train/yamls/pretrain/mpt-125m.yaml

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVE_PATH="/nfs-share/ls985/projects/llm-foundry/checkpoints/$DATETIME"

#! Single-Node training
poetry run composer scripts/train/train.py $CONFIG_MPT_125M train_loader.dataset.split=train_small eval_loader.dataset.split=val_small data_local=$DATA_ROOT_SMALL

#! Multi-Node via CLI args
poetry run composer --world_size 2 --node_rank 0 --master_addr 127.0.0.1 --master_port 6378 scripts/train/train.py $CONFIG_MPT_125M train_loader.dataset.split=train_small eval_loader.dataset.split=val_small data_local=$DATA_ROOT_SMALL device_train_microbatch_size=20 save_interval=10ba save_num_checkpoints_to_keep=1 save_folder=$SAVE_PATH
