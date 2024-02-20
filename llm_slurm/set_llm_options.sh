#!/bin/bash
#! Set the run configuration
STEPS="500"
# #! Configuration: no local checkpointing, no cache limit, no autoresume
# export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=null llm_config.autoresume=false llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=${STEPS}ba"
# #! Test configuration: no local checkpointing, no cache limit, no autoresume, just 10 steps (both train and eval)
# STEPS="10"
# export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=null llm_config.autoresume=false llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=${STEPS}ba llm_config.eval_subset_num_batches=10"
#! Configuration: local checkpointing (w/ overwrite), no cache limit, no autoresume
export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=$SAVE_PATH llm_config.autoresume=false llm_config.save_overwrite=true llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=${STEPS}ba llm_config.save_num_checkpoints_to_keep=0"
# #! Configuration: local checkpointing (w/ overwrite), no cache limit, no autoresume, limit eval to 10 steps
# export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=$SAVE_PATH llm_config.autoresume=false llm_config.save_overwrite=true llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=${STEPS}ba llm_config.save_num_checkpoints_to_keep=0 llm_config.eval_subset_num_batches=10"
