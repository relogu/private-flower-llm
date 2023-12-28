#!/bin/bash
#! Set the run configuration
NUM_LOCAL_STEPS="512"
SAVE_INTERVAL="512ba"
export LLM_OPTIONS="llm_config.save_interval=$SAVE_INTERVAL llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH  llm_config.autoresume=True llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=$NUM_LOCAL_STEPS llm_config.max_duration=88000ba" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"
