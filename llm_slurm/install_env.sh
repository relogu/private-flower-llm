#!/bin/bash
#! Moving to the project folder
cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
if [[ -e $POETRY_ENV_PATH ]]; then
    echo "Poetry environment exists."
    if ! [[ $(poetry check --lock) ]]; then
        echo "Poetry environment is not up-to-date, updating..."
        poetry lock --no-update
    fi
else
    echo "Poetry environment doesn't exist. Installing..."
    poetry config installer.max-workers 10
    poetry install -q
fi
. $POETRY_ENV_PATH/bin/activate
# Adding CUDA paths to environment variables
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#! Check the output of `nvcc -V`
NVCC_OUTPUT=$(nvcc -V)
if [[ $NVCC_OUTPUT == *"release 12.1"* ]]; then
    echo "CUDA 12.1 is detected."
else
    echo "CUDA 12.1 not detected. Please install CUDA 12.1. Exiting..."
    exit 1
fi
#! Install `flash-attn`
if ! [[ $(poetry run pip list | grep flash-attn) ]]; then
    echo "Installing flash-attn..."
    poetry run pip install -q flash-attn==2.3.2 --no-build-isolation
else
    echo "flash-attn is already installed."
fi
#! Install `xentropy-cuda-lib`
if ! [[ $(poetry run pip list | grep xentropy) ]]; then
    echo "Installing xentropy-cuda-lib..."
    poetry run pip install -q xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v2.3.2#subdirectory=csrc/xentropy
else
    echo "xentropy-cuda-lib is already installed."
fi
#! Final message
echo "Environment is ready."
