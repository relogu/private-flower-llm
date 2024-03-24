#!/bin/bash

unset RUN_UUID
export RUN_UUID="fed-1B-20240322_221900-fix-nesto"
unset SAVE_PATH
. $HOME/projects/pollen_worker/llm_slurm/pollen_llm_1B.sh

# unset RUN_UUID
# export RUN_UUID="fed-75M-reset-20240323_183000-fix-nesto"
# unset SAVE_PATH
# . $HOME/projects/pollen_worker/llm_slurm/pollen_llm_75M.sh

# unset RUN_UUID
# export RUN_UUID="fed-125M-reset-20240323_213000-fix-nesto"
# unset SAVE_PATH
# . $HOME/projects/pollen_worker/llm_slurm/pollen_llm_125M.sh