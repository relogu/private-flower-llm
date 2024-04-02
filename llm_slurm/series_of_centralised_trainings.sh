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


#! Small LM as configured by MosaicML. We definetly need to modify it to make it usable.
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=10ba llm_config.max_duration=88000ba llm_config.optimizer.lr=6.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "small"


#! 75M LM as configured by us. We followed what's reported in DiLoCo paper.
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=2900ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "75M"

#! 75M LM as configured by us. We followed what's reported in DiLoCo paper. Proposing reducing the minimum LR to 4.0e-6
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=2900ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.01"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "75M"


#! 125M LM as configured by MosaicML.
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=4800ba llm_config.max_duration=88000ba llm_config.optimizer.lr=6.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "125M"

#! 125M LM as configured by MosaicML. Proposed reducing the maximum LR to 4.0e-4
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=4800ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "125M"


#! 160M LM as configured by us. We followed what's reported in DiLoCo paper. We added some customisations.
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=6200ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "160M"

#! 160M LM as configured by us. We followed what's reported in DiLoCo paper. We added some customisations. Proposing reducing the minimum LR to 4.0e-6
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=6200ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.01"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "160M"


#! 350M LM as configured by MosaicML. It blows up at step 16000
# export RUN_UUID="centralised-350M-20240305_191112"
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=13400ba llm_config.max_duration=88000ba llm_config.optimizer.lr=3.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "350M"

#! 350M LM as configured by MosaicML. Proposed reducing the minimum LR to 3.0e-6
unset RUN_UUID
unset SAVE_PATH
unset STEPS_DONE
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=13400ba llm_config.max_duration=88000ba llm_config.optimizer.lr=3.0e-4 llm_config.scheduler.alpha_f=0.01"
# export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
. $PROJECT_PATH/llm_slurm/centralised_training.sh "350M"


#! 420M LM as configured by us. We followed what's reported in DiLoCo paper. We added some customisations.
# export RUN_UUID="centralised-420M-20240305_190626"
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=16000ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "420M"

#! 420M LM as configured by us. We followed what's reported in DiLoCo paper. We added some customisations. Proposing reducing the minimum LR to 4.0e-6
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=16000ba llm_config.max_duration=88000ba llm_config.optimizer.lr=4.0e-4 llm_config.scheduler.alpha_f=0.01"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "420M"


#! 760M LM as configured by MosaicML.
# export RUN_UUID="centralised-760M-20240305_190707"
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=29000ba llm_config.max_duration=88000ba llm_config.optimizer.lr=2.5e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "760M"


#! 1B LLM as configured by MosaicML.
# export RUN_UUID="centralised-1B-20240229_104204"
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE="22500"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=24000ba llm_config.max_duration=88000ba llm_config.optimizer.lr=2.0e-4 llm_config.scheduler.alpha_f=0.1"
# export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "1B"


#! 3B LLM as configured by MosaicML.
# unset RUN_UUID
# export SAVE_PATH="s3://checkpoints/$RUN_UUID"
# export STEPS_DONE
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba llm_config.scheduler.t_max=51500ba llm_config.max_duration=88000ba llm_config.optimizer.lr=1.6e-4 llm_config.scheduler.alpha_f=0.1"
# # export EXTERNAL_CONFIGS="$EXTERNAL_CONFIGS wandb.setup.allow_val_change=true wandb.setup.id=$RUN_UUID llm_config.load_path=s3://checkpoints/$RUN_UUID/ep0-ba$STEPS_DONE-rank0.pt"
# . $PROJECT_PATH/llm_slurm/centralised_training.sh "3B"

