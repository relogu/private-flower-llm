#!/bin/bash

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-75M-20240225_003829"
# export RUN_UUID="centralised-75M-20240225_003829"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba wandb.setup.allow_val_change=true wandb.setup.id=centralised-75M-20240225_003829 llm_config.load_path=s3://checkpoints/centralised-75M-20240225_003829/ep0-ba4800-rank0.pt"
# #! Note: The first experiment is run with 4800ba, so we are just extending it for another 83200ba
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "75M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-75M-20240225_010800"
# export RUN_UUID="centralised-75M-20240225_010800"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba wandb.setup.allow_val_change=true wandb.setup.id=centralised-75M-20240225_010800 llm_config.load_path=s3://checkpoints/centralised-75M-20240225_010800/ep0-ba15500-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "75M" "88000ba"

#! Skip for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "4800ba"

#! Skip for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-420M-20240224_161116"
# export RUN_UUID="centralised-420M-20240224_161116"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true wandb.setup.id=centralised-420M-20240224_161116 llm_config.load_path=s3://checkpoints/centralised-420M-20240224_161116/ep0-ba2000-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-420M-20240224_223822"
# export RUN_UUID="centralised-420M-20240224_223822"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba wandb.setup.allow_val_change=true wandb.setup.id=centralised-420M-20240224_223822 llm_config.load_path=s3://checkpoints/centralised-420M-20240224_223822/ep0-ba16500-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

#! This blows up
# export SAVE_PATH="s3://checkpoints/centralised-160M-20240223_095752"
# export RUN_UUID="centralised-160M-20240223_095752"
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=1000ba wandb.setup.id=h33305xu llm_config.load_path=s3://checkpoints/centralised-160M-20240223_095752/ep0-ba8500-rank0.pt"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "160M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-160M-20240223_112700"
export RUN_UUID="centralised-160M-20240223_112700"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=0ba wandb.setup.id=rmfap7hs llm_config.load_path=s3://checkpoints/centralised-160M-20240223_112700/ep0-ba18500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "160M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-1B-20240223_223131"
export RUN_UUID="centralised-1B-20240223_223131"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true wandb.setup.allow_val_change=true llm_config.scheduler.t_warmup=100ba wandb.setup.id=87n121y2 llm_config.load_path=s3://checkpoints/centralised-1B-20240223_223131/ep0-ba4000-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "1B" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-760M-20240225_103905"
export RUN_UUID="centralised-760M-20240225_103905"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba wandb.setup.id=87n121y2 llm_config.load_path=s3://checkpoints/centralised-760M-20240225_103905/ep0-ba2500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "29000ba"

export SAVE_PATH="s3://checkpoints/centralised-760M-20240225_125527"
export RUN_UUID="centralised-760M-20240225_125527"
export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba wandb.setup.id=87n121y2 llm_config.load_path=s3://checkpoints/centralised-760M-20240225_125527/ep0-ba2500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "88000ba"

#! Skip for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=100ba"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "51500ba"

#! Skip for now
# unset SAVE_PATH
# unset RUN_UUID
# export EXTERNAL_CONFIGS="llm_config.save_overwrite=true llm_config.scheduler.t_warmup=0ba"
# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "88000ba"
