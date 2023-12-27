#!/bin/bash
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Install Poetry environment
if ! [[ $(poetry check --lock) ]]; then
    echo "Installing Poetry environment..."
    poetry install -q
else
    echo "Poetry environment already installed."
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate
#! Check if the appropriate cuda version is in then PATH
if ! [[ $PATH == *"cuda-12.1"* ]] && ! [[ $LD_LIBRARY_PATH == *"cuda-12.1"* ]]; then
    echo "CUDA 12.1 not in PATH or LD_LIBRARY_PATH. Exiting..."
    echo "PATH=$PATH"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    exit 1
fi
#! Check the output of `nvcc -V`
nvcc -V
#! Install `flash-attn`
if ! [[ $(poetry run pip list | grep flash-attn) ]]; then
    echo "Installing flash-attn..."
    poetry run pip install -q flash-attn==2.3.2 --no-build-isolation
fi
#! Install `xentropy-cuda-lib`
if ! [[ $(poetry run pip list | grep xentropy) ]]; then
    echo "Installing xentropy-cuda-lib..."
    poetry run pip install -q xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v2.3.2#subdirectory=csrc/xentropy
fi
