#!/bin/bash
#! Set `llm_config` names
if [[ $(hostname) == 'mauao' ]]; then
    echo "Assuming we are running on the A40s of Mauao."
    #! NOTE: The following settings have been tested to be compatible with and to maximise the throughput of the A40s on Mauao
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=128" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=50"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=18 llm_config.device_eval_batch_size=40"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=35 llm_config.model.attn_config.attn_impl=torch"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30 llm_config.model.attn_config.attn_impl=torch"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=15 llm_config.model.attn_config.attn_impl=torch"
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1"
else
    echo "Assuming we are running on A100s-equipped machine."
    #! NOTE: The following settings have been tested to be compatible with and to maximise the throughput of the A100s on the CSD3
    LLM_CONFIG_MPT_SMALL_CPU="llm_config=mpt-small-cpu llm_config.model.init_device=meta llm_config.model.loss_fn=fused_crossentropy llm_config.model.attn_config.attn_impl=triton llm_config.precision=amp_bf16 llm_config.device_train_microbatch_size=256 llm_config.eval_subset_num_batches=-1 llm_config.device_eval_batch_size=128" # The model is so small that the batch sizes really don't matter
    LLM_CONFIG_MPT_125M="llm_config=mpt-125m llm_config.device_train_microbatch_size=20 llm_config.device_eval_batch_size=50"
    LLM_CONFIG_MPT_350M="llm_config=mpt-350m llm_config.device_train_microbatch_size=18 llm_config.device_eval_batch_size=40"
    LLM_CONFIG_MPT_760M="llm_config=mpt-760m llm_config.device_train_microbatch_size=4 llm_config.device_eval_batch_size=35"
    LLM_CONFIG_MPT_1B="llm_config=mpt-1b llm_config.device_train_microbatch_size=2 llm_config.device_eval_batch_size=35"
    LLM_CONFIG_MPT_3B="llm_config=mpt-3b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=30"
    LLM_CONFIG_MPT_7B="llm_config=mpt-7b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=15"
    LLM_CONFIG_MPT_70B="llm_config=mpt-70b llm_config.device_train_microbatch_size=1 llm_config.device_eval_batch_size=1"
fi
#! Set the run configuration
export LLM_CONFIG=$LLM_CONFIG_MPT_SMALL_CPU
# export LLM_CONFIG=$LLM_CONFIG_MPT_125M
# export LLM_CONFIG=$LLM_CONFIG_MPT_350M
# export LLM_CONFIG=$LLM_CONFIG_MPT_760M
# export LLM_CONFIG=$LLM_CONFIG_MPT_1B
# export LLM_CONFIG=$LLM_CONFIG_MPT_3B
# export LLM_CONFIG=$LLM_CONFIG_MPT_7B
# export LLM_CONFIG=$LLM_CONFIG_MPT_70B