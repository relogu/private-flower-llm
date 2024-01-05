#!/bin/bash
#! Set the run configuration
STEPS="512"
#! Configuration: no local checkpointing, no cache limit, no autoresume
export LLM_OPTIONS="llm_config.save_interval=${STEPS}ba llm_config.console_log_interval=${STEPS}ba llm_config.save_folder=null llm_config.autoresume=false llm_config.train_loader.dataset.cache_limit=null llm_config.eval_loader.dataset.cache_limit=null llm_config.local_steps=$STEPS llm_config.max_duration=88000ba"
