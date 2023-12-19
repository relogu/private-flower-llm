#!/bin/bash
#! Add modules from scratch to be sure everything works
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment
#! Remove old cuda
module unload cuda/11.4
module unload cuda/11.4.0/gcc-9.4.0-3hnxhjt
#! Load cuda 12.1
module load cuda/12.1
module load cudnn/8.9_cuda-12.1
#! Load additional modules
module load ceuadmin/gettext/0.20
module load vgl/2.5.1/64
module load intel-oneapi-compilers/2022.1.0/gcc/b6zld2mz # This might/should fail
module load intel-oneapi-mpi/2021.6.0/intel/guxuvcpm # This might/should fail
module load rhel8/default-icl
#! Check the output of `nvcc -V`
nvcc -V
#! Check Python version
python --version
#! Entering the project folder
cd $HOME/projects/pollen_worker
#! Install Poetry environment
poetry install
