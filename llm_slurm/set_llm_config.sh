#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"
echo "PROJECT_PATH=$PROJECT_PATH"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
  echo "Error parsing options" >&2
  exit 1
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -p | --project_path )
      PROJECT_PATH="$2"; shift 2 ;;
    -- )
      shift; break ;;
    * )
      break ;;
  esac
done
#! Check if there's an input argument
if [[ $# -eq 0 ]]; then
    echo "No input argument supplied."
    exit 1
fi
#! Get info about GPU resources available
GPU_TYPE=$(nvidia-smi -L)
#! Set `llm_config` names
if [[ $GPU_TYPE == *'A40'* ]]; then
    echo "Assuming we are running on A40-equipped machines."
    #! NOTE: We're assuming 'amp_bf16' is used
    #! From: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=1497e11"
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=flash llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=32" # FSDP goes OOM
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # FSDP goes OOM
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # FSDP goes OOM
elif [[ $GPU_TYPE == *'A100'* ]]; then
    echo "Assuming we are running on A100-equipped machines."
    #! Already hardcoded in the MosaicML's callback, but if passed, we avoid a very bad bug
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=312e12"
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=flash llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=512 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=64 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=8 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=4"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1"
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1"
elif [[ $GPU_TYPE == *'H100'* ]]; then
    echo "Assuming we are running on H100-equipped machines."
    #! Already hardcoded in the MosaicML's callback, but if passed, we avoid a very bad bug
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=312e12"
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=flash llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=512 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=64 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=8 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=8 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1"
elif [[ $GPU_TYPE == *'L40'* ]]; then
    echo "Assuming we are running on L40-equipped machines."
    #! NOTE: We're assiming 'amp_bf16' is used
    #! From: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/datasheets/L-40/product-brief-L40.pdf
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=18105e10"
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=flash llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=64 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=32"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=16" # FSDP goes OOM
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # FSDP goes OOM
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # FSDP goes OOM
else
    echo "Unknown GPU type: $GPU_TYPE. Using defaults..."
fi
#! Set the run configuration
if [[ "$1" == "small" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_SMALL_CPU"
elif [[ "$1" == "16M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_16M"
elif [[ "$1" == "75M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_75M"
elif [[ "$1" == "125M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_125M"
elif [[ "$1" == "160M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_160M"
elif [[ "$1" == "350M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_350M"
elif [[ "$1" == "420M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_420M"
elif [[ "$1" == "540M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_540M"
elif [[ "$1" == "760M" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_760M"
elif [[ "$1" == "1B" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_1B"
elif [[ "$1" == "3B" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_3B"
elif [[ "$1" == "7B" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_7B"
elif [[ "$1" == "70B" ]]; then
    export LLM_CONFIG="$FLOP_COUNT $LLM_CONFIG_MPT_70B"
else
    echo "Invalid input argument: $1"
    echo "Valid input arguments are: small, 16M, 75M, 125M, 160M, 350M, 420M, 540M, 760M, 1B, 3B, 7B, 70B"
    exit 1
fi

echo "Selected LLM config: $1"
