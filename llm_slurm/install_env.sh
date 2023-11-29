#!/bin/bash

#! Moving to the project folder
cd /nfs-share/ls985/projects/pollen_worker
#! Install Poetry environment
poetry install
#! Activate Poetry shell
poetry shell
#! Set the CUDA version
export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#! Check the output of `nvcc -V`
nvcc -V
#! Install `flash-attn`
poetry run pip install flash-attn==2.3.2 --no-build-isolation
#! Install `xentropy-cuda-lib`
poetry run pip install xentropy-cuda-lib@git+https://github.com/HazyResearch/flash-attention.git@v2.3.2#subdirectory=csrc/xentropy
