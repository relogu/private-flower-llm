#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Evaluate of the evaluation set of the C4 dataset the global model at every round for the list of run_uuid given
# EXPERIMENTS_LIST=(
#     "centB-125M-20240907_115804" - different handling
#     "matrix-125M-20240823-tle"
#     "pers-125M-20240823_131628"" -- personalized
#     "full-125M-20240823_125915"
#     "centB-125M-p-20240904_113154" - different handling
#     "matrix-125M-p-tle"
#     "pers-125M-p-20240821_tle"" -- personalized
#     "full-125M-p-tle"
#     "pers-350M-p-20240825_232233"  -- 350M -- personalized — executing
#     "centB-350M-p-20240905_093402"  -- 350M- different handling
#     "matrix-350M-p-20240825_231845" -- 350M
#     "full-350M-p-20240825_tle" -- 350M
# )

########### Models with 125M parameters ###########
EXPERIMENTS_LIST=(
	"matrix-125M-20240823-tle"
	"full-125M-20240823_125915"
	"matrix-125M-p-tle"
	"full-125M-p-tle"
)

# Iterate over the experiment run_uuids
for experiment in "${EXPERIMENTS_LIST[@]}"; do
	echo "Evaluating experiment: $experiment"
	# Iterate over the federated rounds
	for federated_round in {0..10}; do
		echo "Federated Round: $federated_round"
		DATETIME=$(date '+%Y%m%d_%H%M%S')                                                                                               # Get the current datetime for composing the current run_uuid
		export EXTERNAL_CONFIGS="centralized.eval_only=true"                                                                            # Run the evaluation only
		export EXTERNAL_CONFIGS="use_wandb=false ~llm_config.loggers.wandb ~llm_config.loggers.tensorboard"                             # Disable wandb and tensorboard
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS icl_tasks_config=empty eval_gauntlet_config=empty"                                   # Disable ICL and eval_gauntlet
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=flash"                                        # Use the flash attention implementation
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=torch"                                        # Use the torch attention implementation
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto llm_config.device_eval_batch_size=128" # Use auto microbatching for evaluation
		export CHECKPOINT_PATH="s3://checkpoints/$experiment/server/$federated_round/current_server_parameters.npz"                     # Select thec checkpoint path
		export RUN_UUID="mclr-eval-$DATETIME"
		bash $PROJECT_PATH/scripts/eval_gauntlet_only.sh "125M"
		sleep 30
	done
done

########## Models with 350M parameters ##########
EXPERIMENTS_LIST=(
	"matrix-350M-p-20240825_231845"
	"full-350M-p-20240825_tle"
)

# Iterate over the experiment run_uuids
for experiment in "${EXPERIMENTS_LIST[@]}"; do
	echo "Evaluating experiment: $experiment"
	# Iterate over the federated rounds
	for federated_round in {0..27}; do
		echo "Federated Round: $federated_round"
		DATETIME=$(date '+%Y%m%d_%H%M%S')                                                                                               # Get the current datetime for composing the current run_uuid
		export EXTERNAL_CONFIGS="centralized.eval_only=true"                                                                            # Run the evaluation only
		export EXTERNAL_CONFIGS="use_wandb=false ~llm_config.loggers.wandb ~llm_config.loggers.tensorboard"                             # Disable wandb and tensorboard
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS icl_tasks_config=empty eval_gauntlet_config=empty"                                   # Disable ICL and eval_gauntlet
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=flash"                                        # Use the flash attention implementation
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS llm_config.model.attn_config.attn_impl=torch"                                        # Use the torch attention implementation
		export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS ++llm_config.device_eval_microbatch_size=auto llm_config.device_eval_batch_size=128" # Use auto microbatching for evaluation
		export CHECKPOINT_PATH="s3://checkpoints/$experiment/server/$federated_round/current_server_parameters.npz"                     # Select thec checkpoint path
		export RUN_UUID="mclr-eval-$DATETIME"
		bash $PROJECT_PATH/scripts/eval_gauntlet_only.sh "350M"
		sleep 30
	done
done
