#!/bin/bash
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Install Poetry environment
poetry install
#! Activate Poetry shell
poetry shell
#! Check if the appropriate cuda version is in then PATH
if ! [[$PATH == *"cuda-12.1"*]] && ! [[$LD_LIBRARY_PATH == *"cuda-12.1"*]]; then
    echo "CUDA 12.1 not in PATH or LD_LIBRARY_PATH. Exiting..."
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi
#! Check the output of `nvcc -V`
nvcc -V
#! Install `flash-attn`
poetry run pip install flash-attn==2.3.2 --no-build-isolation
#! Install `xentropy-cuda-lib`
poetry run pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v2.3.2#subdirectory=csrc/xentropy
