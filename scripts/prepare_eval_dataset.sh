#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
## This script aims to prepare the evaluation gauntlet datasets. The datasets will be donwloaded from the Lorenzo's fork of the `llm-foundry` repository

#! Check what has been set in the environmental varaiables
echo "prepare_eval_dataset.sh: HOME=$HOME, DATASET_CACHE=$DATASET_CACHE"
echo "prepare_eval_dataset.sh: the datasets will be moved to $DATASET_CACHE/eval/local_data"

#! Clone the repo and move to the 'fl' branch
mkdir -p "$HOME"/projects
cd "$HOME"/projects || exit
git clone git@github.com:relogu/llm-foundry.git
cd llm-foundry || exit
git fetch origin
git checkout --track origin/fl

#! Copy the datasets folder to the DATASET_CACHE folder
mkdir -p "$DATASET_CACHE"/eval/local_data
cp -r scripts/eval/local_data/* "$DATASET_CACHE"/eval/local_data
