#!/bin/bash
#! Check if there's an input argument
if [[ $# -eq 0 ]]; then
    echo "No input argument supplied."
    exit 1
fi
#! Get info about GPU resources available
GPU_TYPE=$(nvidia-smi -L)
#! Set `llm_config` names
if [[ $GPU_TYPE == *'A40'* ]]; then
    echo "Assuming we are running on A40s (Mauao)."
    # NOTE: We're assiming 'amp_bf16' is used
    # From: https://images.nvidia.com/content/Solutions/data-center/a40/nvidia-a40-datasheet.pdf
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=1497e11"
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=729" # The model is so small that the batch sizes really don't matter
    # LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=82 llm_config.device_eval_batch_size=200" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=40" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=40" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=16 llm_config.device_eval_batch_size=40"
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=80"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=40 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=3 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
elif [[ $GPU_TYPE == *'A100'* ]]; then
    echo "Assuming we are running on A100-equipped machines."
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=312e12" # Already hardcoded in the MosaicML's callback, but if passed, we avoid a very bad mistake of them
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=512 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=1523" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=82 llm_config.device_eval_batch_size=350" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=80"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=86" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=15 llm_config.device_eval_batch_size=84"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=5"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1" # CANNOT DO IT
elif [[ $GPU_TYPE == *'H100'* ]]; then
    echo "Assuming we are running on H100-equipped machines."
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=312e12" # Already hardcoded in the MosaicML's callback, but if passed, we avoid a very bad mistake of them
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=512 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=1523" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=82 llm_config.device_eval_batch_size=350" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_75M="llm_config=mpt-75m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_160M="llm_config=mpt-160m llm_config.device_train_microbatch_size=40 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=60" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_420M="llm_config=mpt-420m llm_config.device_train_microbatch_size=32 llm_config.device_eval_batch_size=80"
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=86" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=80" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=18 llm_config.device_eval_batch_size=60" # This has issues (we can't scale up the microbatch size even though it seems possible) -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=10 llm_config.device_eval_batch_size=64"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1" # CANNOT DO IT
elif [[ $GPU_TYPE == *'L40'* ]]; then
    echo "Assuming we are running on L40s-equipped machines."
    # NOTE: We're assiming 'amp_bf16' is used
    # From: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/datasheets/L-40/product-brief-L40.pdf
    FLOP_COUNT="llm_config.callbacks.speed_monitor.gpu_flops_available=18105e10"
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=128" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_16M="llm_config=mpt-16m llm_config.device_train_microbatch_size=128 llm_config.device_eval_batch_size=50" # This has issues (we can't scale up the microbatch size even though it seems possible)  -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=50 llm_config.device_eval_batch_size=50" # This has issues (we can't scale up the microbatch size even though it seems possible)  -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=15 llm_config.device_eval_batch_size=40" # This has issues (we can't scale up the microbatch size even though it seems possible)  -> https://github.com/Dao-AILab/flash-attention/issues/483
    LLM_CONFIG_MPT_540M="llm_config=mpt-540m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=40 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=3 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
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
