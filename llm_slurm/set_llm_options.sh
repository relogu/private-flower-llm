#!/bin/bash
#! Set the run configuration
NUM_STEPS="100ba"
SAVE_INTERVAL="101ba"
export LLM_OPTIONS="llm_config.save_interval=$SAVE_INTERVAL llm_config.save_num_checkpoints_to_keep=1 llm_config.save_folder=$SAVE_PATH  llm_config.max_duration=$NUM_STEPS llm_config.autoresume=True llm_config.train_loader.dataset.cache_limit=1gb llm_config.eval_loader.dataset.cache_limit=1gb" # llm_config.load_path=$SAVE_PATH/ckpt-0.pt"
