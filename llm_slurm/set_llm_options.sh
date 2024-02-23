#!/bin/bash
#! Set the run configuration
STEPS="500"
# #! Test configuration: just 10 steps during training
# STEPS="2"
#! Configuration: local checkpointing, one local checkpoint, no callbacks
export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=$SAVE_PATH llm_config.local_steps=${STEPS}ba llm_config.save_num_checkpoints_to_keep=1 llm_config.callbacks=null"
