#!/bin/bash
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
    echo "Assuming the script is executing in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_hpc_env.sh
else
    echo "Assuming the script is executing NOT in the CSD3."
    #! Executing the environment preparation script
    #! NOTE: Must use "." to execute, "sh" doesn't work
    . $HOME/projects/pollen_worker/llm_slurm/install_env.sh
fi
#! Set `LLM_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_config.sh "75M"
#! Set `DATA_CONFIG` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_data_config.sh
#! Saving path
DATETIME=$(date '+%Y%m%d_%H%M%S')
export POLLEN_SAVE_PATH="$HOME/projects/pollen_worker/checkpoints/$DATETIME"
#! If SAVE_PATH hasn't been set, set it to the default value
if [ -z "$SAVE_PATH" ]; then
    export SAVE_PATH="s3://checkpoints"
fi
#! If RUN_UUID hasn't been set, set it to the default value
if [ -z "$RUN_UUID" ]; then
    export RUN_UUID="fed-75M-$DATETIME"
fi
mkdir -p $POLLEN_SAVE_PATH
#! Set `LLM_OPTIONS` environment variable
. $HOME/projects/pollen_worker/llm_slurm/set_llm_options.sh
#! Getting visible GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N_GPUS-1)))
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
#! AWS S3 object store settings
aws_access_key_id=$(grep 'aws_access_key_id' ~/.aws/credentials | awk -F' = ' '{print $2}')
aws_secret_access_key=$(grep 'aws_secret_access_key' ~/.aws/credentials | awk -F' = ' '{print $2}')
MINIO_COMM_STACK_OPTIONS="use_minio_comm=false minio.minio_client.endpoint=$S3_ENDPOINT_URL minio.minio_client.access_key=$aws_access_key_id minio.minio_client.secret_key=$aws_secret_access_key minio.minio_state.bucket_name=checkpoints"
#! Set Pollen and FL config
POLLEN_CONFIG="pollen.server_address='localhost:50744' run_uuid=$RUN_UUID pollen.refresh_period=20 fl.n_rounds=176 pollen.checkpoint=true llm_config.scheduler.t_max=88000ba llm_config.scheduler.t_warmup=1000ba llm_config.scheduler.alpha_f=0.00001 llm_config.optimizer.lr=4.0e-4 llm_config.save_overwrite=true pollen.n_nodes=1 pollen.resume_round=-1"
#! Launch ServerWithPollen
GRPC_VERBOSITY=debug HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.launch_pollen_server $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG pollen.saving_path=$SAVE_PATH $MINIO_COMM_STACK_OPTIONS 2>&1 | tee $POLLEN_SAVE_PATH/server.log &
#! Wait for 30 seconds. This is needed because of how the client connection behaves.
sleep 30
#! Launch NodeManager
#! NOTE: Adding `NCCL_BLOCKING_WAIT=1` breaks the optimizer's checkpointing. We don't know why yet.
GRPC_VERBOSITY=debug CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 poetry run python -m pollen_worker.node_manager.node_manager $LLM_CONFIG $LLM_OPTIONS $DATA_CONFIG $POLLEN_CONFIG $MINIO_COMM_STACK_OPTIONS is_test=false hydra/job_logging=none hydra/hydra_logging=none 2>&1 | tee $POLLEN_SAVE_PATH/node_manager.log &
#! Keep the pid and wait for it 
BACK_PID=$!
wait $BACK_PID