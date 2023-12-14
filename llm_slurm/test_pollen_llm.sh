#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/


#! Moving to the project folder
cd /nfs-share/$USER/projects/pollen_worker
# #! Activate Poetry environment
# poetry shell

# Remove shared memories of the user if they exist
find /dev/shm -name '*pollen*' -type f -delete 
find /dev/shm -name '*_locals' -type f -delete
#! Add the appropriate CUDA version to the paths
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://127.0.0.1:9000'
export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Check the output of `nvcc -V`
nvcc -V
#! Set data paths
# DATA_VERSION="small"
DATA_VERSION="full"
IS_LOCAL=false
# IS_LOCAL=true
if [ "$DATA_VERSION" == "small" ]; then
    if [ "$IS_LOCAL" == true ]; then
        DATA_CONFIG="llm_config.data_local=/local/scratch/$USER/small-c4llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    else
        DATA_CONFIG="llm_config.data_local=/tmp/$USER/small-c4 llm_config.data_remote=s3://small-c4-dataset llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    fi
else
    if [ "$IS_LOCAL" == true ]; then
        DATA_CONFIG="llm_config.data_local=/local/scratch/$USER/c4 llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    else
        DATA_CONFIG="llm_config.data_local=/tmp/$USER/c4 llm_config.data_remote=s3://c4-dataset llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    fi
fi
#! Get info about resources available
NUM_CPUS=$SLURM_CPUS_PER_TASK
echo "Number of CPU cores available: $NUM_CPUS"
#! Set `llm_config` names
#! NOTE: The following settings have been tested to be compatible with and to maximise the throughput of the A40s on Mauao
LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=512" # The model is so small that the batch sizes really don't matter
LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=50"
LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=18 llm_config.device_eval_batch_size=40"
LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch"
LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch"
LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30 llm_config.model.attn_config.attn_impl=torch"
LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=15 llm_config.model.attn_config.attn_impl=torch"
# LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1"
#! Set the run configuration
LLM_CONFIG=$LLM_CONFIG_MPT_SMALL_CPU
# LLM_CONFIG=$LLM_CONFIG_MPT_125M
# LLM_CONFIG=$LLM_CONFIG_MPT_350M
# LLM_CONFIG=$LLM_CONFIG_MPT_760M
# LLM_CONFIG=$LLM_CONFIG_MPT_1B
# LLM_CONFIG=$LLM_CONFIG_MPT_3B
# LLM_CONFIG=$LLM_CONFIG_MPT_7B
# LLM_CONFIG=$LLM_CONFIG_MPT_70B

#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
SAVE_PATH="/nfs-share/$USER/projects/pollen_worker/checkpoints/$DATETIME"
mkdir -p $SAVE_PATH

#! LLM-related options
NUM_STEPS="100ba"
SAVE_INTERVAL="101ba"
# LLM_OPTIONS="llm_config.save_interval=11ba llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH llm_config.loggers.wandb.project=llm llm_config.loggers.wandb.name='test_node_manager_mpt_125m' llm_config.train_loader.num_workers=$NUM_CPUS llm_config.eval_loader.num_workers=$NUM_CPUS llm_config.max_duration=10ba llm_config.autoresume=True" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"
LLM_OPTIONS="llm_config.save_interval=$SAVE_INTERVAL llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH llm_config.train_loader.num_workers=$NUM_CPUS llm_config.eval_loader.num_workers=$NUM_CPUS llm_config.max_duration=$NUM_STEPS llm_config.autoresume=True" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"

#! Test NodeManager
HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.server_with+pollen $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG pollen.saving_path=$SAVE_PATH 2>&1 | tee $SAVE_PATH/server.log &


HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $SAVE_PATH/node_manager.log 
