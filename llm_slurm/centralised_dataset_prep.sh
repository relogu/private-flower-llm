#!/bin/bash

#! Moving to the project folder
cd /nfs-share/ls985/projects/llm-foundry
#! Activate Poetry environment
poetry shell

DATA_ROOT=/home/ls985/c4
DATA_ROOT_MDS=/home/ls985/mds-c4
DATA_ROOT_SMALL=/home/ls985/my-copy-c4
DATA_ROOT_SMALL_MDS=/home/ls985/my-mds-copy-c4

#! Download and pre-processing
# #! Full dataset
# poetry run python scripts/data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root $DATA_ROOT --splits train val --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
# poetry run python scripts/data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root $DATA_ROOT_MDS --splits train val --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' --compression zstd # Compressed
#! Small dataset
# poetry run python scripts/data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root $DATA_ROOT_SMALL --splits train_small val_small --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>'
# poetry run python scripts/data_prep/convert_dataset_hf.py --dataset c4 --data_subset en --out_root $DATA_ROOT_SMALL_MDS --splits train_small val_small --concat_tokens 2048 --tokenizer EleutherAI/gpt-neox-20b --eos_text '<|endoftext|>' --compression zstd # Compressed

#! Test data
# #! Full dataset
# poetry run python llmfoundry/data/text_data.py --local_path $DATA_ROOT --split val
# poetry run python llmfoundry/data/text_data.py --local_path $DATA_ROOT_MDS --split val # Compressed
#! Small dataset
poetry run python llmfoundry/data/text_data.py --local_path $DATA_ROOT_SMALL --split val_small
# poetry run python llmfoundry/data/text_data.py --local_path $DATA_ROOT_SMALL_MDS --split val_small # Compressed