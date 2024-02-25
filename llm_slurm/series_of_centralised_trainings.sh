#!/bin/bash

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "75M" "4800ba"

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "75M" "88000ba"

#! Skipping for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "4800ba"

#! Skipping for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-420M-20240224_161116"
# export RUN_UUID="centralised-420M-20240224_161116"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true wandb.setup.id=centralised-420M-20240224_161116 llm_config.load_path=s3://checkpoints/centralised-420M-20240224_161116/ep0-ba2000-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-420M-20240224_223822"
export RUN_UUID="centralised-420M-20240224_223822"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba wandb.setup.allow_val_change=true wandb.setup.id=centralised-420M-20240224_223822 llm_config.load_path=s3://checkpoints/centralised-420M-20240224_223822/ep0-ba5500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-160M-20240223_095752"
# export RUN_UUID="centralised-160M-20240223_095752"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=1000ba wandb.setup.id=h33305xu llm_config.load_path=s3://checkpoints/centralised-160M-20240223_095752/ep0-ba8500-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "160M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-160M-20240223_112700"
export RUN_UUID="centralised-160M-20240223_112700"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=0ba wandb.setup.id=rmfap7hs llm_config.load_path=s3://checkpoints/centralised-160M-20240223_112700/ep0-ba8500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "160M" "88000ba"

#! It's the same as the one after this but with fewer total steps, so skipping
# export SAVE_PATH="s3://checkpoints/centralised-1B-20240223_131951"
# export RUN_UUID="centralised-1B-20240223_131951"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=100ba wandb.setup.id=qlwnyu8o llm_config.load_path=s3://checkpoints/centralised-1B-20240223_131951/ep0-ba3500-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "1B" "24800ba"

export SAVE_PATH="s3://checkpoints/centralised-1B-20240223_223131"
export RUN_UUID="centralised-1B-20240223_223131"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=100ba wandb.setup.id=87n121y2 llm_config.load_path=s3://checkpoints/centralised-1B-20240223_223131/ep0-ba3500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "1B" "88000ba"

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "29000ba"

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "88000ba"

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "51500ba"

unset SAVE_PATH
unset RUN_UUID
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "88000ba"
