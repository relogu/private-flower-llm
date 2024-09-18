#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091

# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"
SAVE_PATH="$PROJECT_PATH/tokenizers_logs"
mkdir -p "$SAVE_PATH"

## Evaluation

# Vocab size 32000

# DATETIME=$(date '+%Y%m%d_%H%M%S')
# python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name ur --splits validation --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_ur_validation_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sw --splits validation --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sw_validation_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name la --splits validation --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_la_validation_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sr --splits validation --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sr_validation_32000_$DATETIME.log

# Vocab size 50000

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name ur --splits validation --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_ur_validation_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sw --splits validation --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sw_validation_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name la --splits validation --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_la_validation_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sr --splits validation --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sr_validation_50000_$DATETIME.log

## Train

# Vocab size 32000

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name ur --splits train --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_ur_train_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sw --splits train --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sw_train_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name la --splits train --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_la_train_32000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sr --splits train --vocab_size 32000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sr_train_32000_$DATETIME.log

# Vocab size 50000

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name ur --splits train --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_ur_train_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sw --splits train --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sw_train_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name la --splits train --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_la_train_50000_$DATETIME.log

DATETIME=$(date '+%Y%m%d_%H%M%S')
python /nfs-share/ls985/projects/flower_llm/flower_llm/dataset/train_tokenizer.py --output_root_dir "/nfs-share/ls985/projects/flower_llm/" --path allenai/c4 --name sr --splits train --vocab_size 50000 2>&1 | tee "$SAVE_PATH"/tokenizer_training_sr_train_50000_$DATETIME.log
