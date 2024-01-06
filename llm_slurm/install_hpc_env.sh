#!/bin/bash
#! Add modules from scratch to be sure everything works
#! Enable the module command
. /etc/profile.d/modules.sh
#! Remove all modules still loaded
module purge
#! Load base modules
module load singularity/current
module load rhel8/slurm
module load dot
module load rhel8/global
module -s load openmpi/4.1.1/gcc-9.4.0-epagguv # This might be unable to locate
#! Load cuda 12.1
module load cuda/12.1
module load cudnn/8.9_cuda-12.1
#! Load additional modules
module load ceuadmin/gettext/0.20
module load vgl/2.5.1/64
#! Check the output of `nvcc -V`
nvcc -V
#! Entering the project folder
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
#! Downgrade python warnings (default in CSD3 is 'debug')
#! From here: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONWARNINGS
#! And here: https://docs.python.org/3/library/warnings.html#describing-warning-filters
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::ResourceWarning"
#! Check Python version
PYTHON_OUTPUT=$(python --version)
if [[ $PYTHON_OUTPUT == *"3.10.13"* ]]; then
    echo "Python 3.10.13 is detected."
else
    echo "Python 3.10.13 not detected. Please install Python 3.10.13. Exiting..."
    exit 1
fi
#! Final message
echo "Environment is ready."
