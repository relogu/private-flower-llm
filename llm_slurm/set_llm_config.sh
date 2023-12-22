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
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=128" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=50"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=18 llm_config.device_eval_batch_size=40"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Cannot use Triton here
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=15 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # CANNOT DO IT
elif [[ $GPU_TYPE == *'A100-SXM4-80GB'* ]]; then
    echo "Assuming we are running on A100-SXM4-80GB-equipped machines (CSD3)."
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=512 llm_config.eval_subset_num_batches=-1" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20" # This has issues (we can't scale up the microbatch size even though it seems possible)
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=20" # This has issues (we can't scale up the microbatch size even though it seems possible)
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=20"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=15"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=5"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.fsdp_config.activation_checkpointing=false llm_config.fsdp_config.mixed_precision=DEFAULT" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1" # CANNOT DO IT
elif [[ $GPU_TYPE == *'L40'* ]]; then
    echo "Assuming we are running on L40s-equipped machines (Fluidstack)."
    #! NOTE: We didn't investigate the performance at inference
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=128" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=50 llm_config.device_eval_batch_size=50" # This has issues (we can't scale up the microbatch size even though it seems possible)
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=15 llm_config.device_eval_batch_size=40"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=3 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Same as A40s: cannot use Triton here. Weirdly, we had to downgrade the microbatch size
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch" # Weirdly, we had to downgrade the microbatch size
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=15 llm_config.model.attn_config.attn_impl=torch" # CANNOT DO IT
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1" # CANNOT DO IT
else
    echo "Unknown GPU type: $GPU_TYPE. Using defaults..."
fi
#! Set the run configuration
if [[ "$1" == "small" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_SMALL_CPU
elif [[ "$1" == "125M" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_125M
elif [[ "$1" == "350M" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_350M
elif [[ "$1" == "760M" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_760M
elif [[ "$1" == "1B" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_1B
elif [[ "$1" == "3B" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_3B
elif [[ "$1" == "7B" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_7B
elif [[ "$1" == "70B" ]]; then
    export LLM_CONFIG=$LLM_CONFIG_MPT_70B
else
    echo "Invalid input argument: $1"
    echo "Valid input arguments are: small, 125M, 350M, 760M, 1B, 3B, 7B, 70B"
    exit 1
fi

echo "Selected LLM config: $1"
