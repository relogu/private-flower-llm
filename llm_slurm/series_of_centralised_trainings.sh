#!/bin/bash

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "4800ba"

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "125M" "88000ba"

. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

export EXTERNAL_CONFIGS="llm_config.scheduler.t_warmup=0ba"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "420M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-160M-20240223_112700"
export RUN_UUID="centralised-160M-20240223_112700"
export EXTERNAL_CONFIGS="wandb.setup.allow_val_change=true wandb.setup.id=rmfap7hs llm_config.load_path=s3://checkpoints/centralised-160M-20240223_112700/ep0-ba8500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "160M" "88000ba"

export SAVE_PATH="s3://checkpoints/centralised-1B-20240223_131951"
export RUN_UUID="centralised-1B-20240223_131951"
export EXTERNAL_CONFIGS="wandb.setup.allow_val_change=true wandb.setup.id=qlwnyu8o llm_config.load_path=s3://checkpoints/centralised-1B-20240223_131951/ep0-ba3500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "1B" "24800ba"

export SAVE_PATH="s3://checkpoints/centralised-1B-20240223_223131"
export RUN_UUID="centralised-1B-20240223_223131"
export EXTERNAL_CONFIGS="wandb.setup.allow_val_change=true wandb.setup.id=87n121y2 llm_config.load_path=s3://checkpoints/centralised-1B-20240223_223131/ep0-ba3500-rank0.pt"
. $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "1B" "88000ba"

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "29000ba"

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "760M" "88000ba"

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "51500ba"

# . $HOME/projects/pollen_worker/llm_slurm/centralised_training.sh "3B" "88000ba"