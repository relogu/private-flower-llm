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
#! Install Poetry environment
poetry install -q
#! Install `flash-attn`
poetry run pip install flash-attn==2.3.2 --no-build-isolation
#! Install `xentropy-cuda-lib`
poetry run pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v2.3.2#subdirectory=csrc/xentropy
#! Check Python version
python --version
