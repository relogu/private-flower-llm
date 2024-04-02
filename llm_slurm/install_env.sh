#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"
echo "PROJECT_PATH=$PROJECT_PATH"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
  echo "Error parsing options" >&2
  exit 1
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -p | --project_path )
      PROJECT_PATH="$2"; shift 2 ;;
    -- )
      shift; break ;;
    * )
      break ;;
  esac
done
#! Moving to the project folder
cd $PROJECT_PATH
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
    POETRY_ENV_PATH=$(poetry env info --path)
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
