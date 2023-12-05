#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

#! Moving to the project folder
cd /nfs-share/ls985/projects/pollen_worker
# #! Activate Poetry environment
# poetry shell
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
LLM_OPTIONS="llm_config.train_loader.dataset.split=train_small llm_config.eval_loader.dataset.split=val_small llm_config.data_local=$DATA_ROOT_SMALL llm_config.device_train_microbatch_size=20 llm_config.save_interval=10ba llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH llm_config.loggers.wandb.project=llm llm_config.loggers.wandb.name='test_node_manager_mpt_125m' llm_config.train_loader.num_workers=12 llm_config.eval_loader.num_workers=12 llm_config.max_duration=10ba llm_config.autoresume=False" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"
LLM_OPTIONS="llm_config.train_loader.dataset.split=train_small llm_config.eval_loader.dataset.split=val_small llm_config.data_local=$DATA_ROOT_SMALL llm_config.device_train_microbatch_size=20 llm_config.save_interval=10ba llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH llm_config.train_loader.num_workers=12 llm_config.eval_loader.num_workers=12 llm_config.max_duration=2ba llm_config.autoresume=False" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"

#! Test NodeManager
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.server_with+pollen $LLM_OPTIONS
