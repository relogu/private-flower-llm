#!/bin/bash


#! Moving to the project folder
cd /nfs-share/ls985/projects/pollen_worker
#! Activate Poetry environment
poetry shell
#! Add the appropriate CUDA version to the paths
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#! Check the output of `nvcc -V`
nvcc -V
#! Set data paths
DATA_ROOT=/home/ls985/c4
DATA_ROOT_MDS=/home/ls985/mds-c4
DATA_ROOT_SMALL=/home/ls985/my-copy-c4
DATA_ROOT_SMALL_MDS=/home/ls985/my-mds-copy-c4

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVE_PATH="/nfs-share/ls985/projects/pollen_worker/checkpoints/$DATETIME"

#! LLM-related options
LLM_OPTIONS="llm_config.train_loader.dataset.split=train_small llm_config.eval_loader.dataset.split=val_small llm_config.data_local=$DATA_ROOT_SMALL llm_config.device_train_microbatch_size=20 llm_config.save_interval=10ba llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH llm_config.loggers.wandb.project=llm llm_config.loggers.wandb.name=test-node-manager-mpt-125m llm_config.train_loader.num_workers=12 llm_config.eval_loader.num_workers=12 llm_config.max_duration=1ba llm_config.autoresume=True"

#! Test NodeManager
poetry run python -m pollen_worker.pure_sh_node_manager $LLM_OPTIONS
