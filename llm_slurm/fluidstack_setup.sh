#!/bin/bash
## This script aims to setup the OS for a fluidstack machine
## starting from the "Plain Ubuntu 20.04" image
#! Update and upgrade package manager
sudo apt-get update
sudo apt-get upgrade -y
#! Installing the essentials
sudo apt-get install -y build-essential zlib1g-dev libedit-dev \
    libssl-dev liblzma-dev libffi-dev libbz2-dev \
    libreadline-dev libsqlite3-dev
#! Check the output of `nvcc -V`
NVCC_OUTPUT=$(nvcc -V)
if [[ $NVCC_OUTPUT == *"release 12.1"* ]]; then
    echo "CUDA 12.1 is detected."
else
    #! Get and install CUDA 12.1.1 and its drivers
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    sudo sh cuda_12.1.1_530.30.02_linux.run --toolkit --no-man-page --driver --silent
fi
if [[ $PATH == *"cuda-12.1"* ]]; then
    echo "PATH variable is already set."
else
    #! Set the PATH env variables for the new CUDA version
    echo '# Adding CUDA 12.1 to the PATH environmental variables' >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}' >> ~/.bashrc
    export PATH=/usr/local/cuda-12.1/bin${PATH:+:${PATH}}
fi
if [[ $LD_LIBRARY_PATH == *"cuda-12.1"* ]]; then
    echo "LD_LIBRARY_PATH variable is already set."
else
    #! Set the LD_LIBRARY_PATH env variables for the new CUDA version
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
#! Set GPU persistence mode
sudo nvidia-smi -pm 1
#! Move the $HOME to the 'ephemeral storage' folder
sudo mkdir -p /ephemeral/$USER
sudo rsync -a $HOME/ /ephemeral/$USER
export HOME="/ephemeral/$USER"
echo '# Move $HOME to `ephemeral storage`' >> ~/.bashrc
echo 'export HOME="/ephemeral/$USER"' >> ~/.bashrc
cd
#! Install `pyenv`
PYENV_VER_OUTPUT=$(pyenv --version)
if [[ $PYENV_VER_OUTPUT == *"pyenv "* ]]; then
    echo "pyenv is already installed."
else
    #! Getting `pyenv`
    curl https://pyenv.run | bash
fi
if [[ $PYENV_ROOT == *"pyenv"* ]]; then
    echo "PYENV_ROOT variable is already set."
else
    #! Setting up `pyenv` to execute automatically in the shell
    echo '# Load `pyenv` automatically' >> ~/.bashrc
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    export PYENV_ROOT="$HOME/.pyenv"
    echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    [[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc
    eval "$(pyenv init -)"
    echo '# # Load pyenv-virtualenv automatically' >> ~/.bashrc
    echo '# eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
fi
#! Installing python 3.10.13
pyenv install -s 3.10.13
#! Selecting this python as global
pyenv global 3.10.13
#! Upgrade pip
pip install --upgrade pip
#! Monitoring utilities
sudo snap install bpytop
pip install nvitop
#! Install poetry
pip install poetry
#! Install cmake
pip install cmake
#! Set up git credentials
echo '[user]' > ~/.gitconfig
echo '    name = relogu' >> ~/.gitconfig
echo '    email = lollonasi97@gmail.com' >> ~/.gitconfig
#! Set up S3 credentials
mkdir ~/.aws
echo '[default]' > ~/.aws/config
echo '[default]' > ~/.aws/credentials
echo '    aws_access_key_id = jj15X7kIlfU9uHwyuTmJ' >> ~/.aws/credentials
echo '    aws_secret_access_key = rAD3IMOhooHO79BD1tY9DbxOY2bSN9MEj02XOFwP' >> ~/.aws/credentials
#! Set up wandb credentials
echo 'machine api.wandb.ai' > ~/.netrc
echo '    login user' >> ~/.netrc
echo '    password 7cc6fc5aec5d4c63f203cd1853a71eb7b9131774' >> ~/.netrc
#! Changing permissions to the 'ephemeral storage' folder
sudo chmod -R a+wr /ephemeral
#! Create the tmp folder in the 'ephemeral storage' folder
sudo mkdir -p /ephemeral/$USER/tmp
#! Clone the repo and move to the 'llm' branch
mkdir -p $HOME/projects
cd $HOME/projects
git clone https://github.com/relogu/pollen_worker.git
cd pollen_worker
git fetch origin
git checkout --track origin/llm
#! Changing permissions to the 'ephemeral storage' folder again
sudo chmod -R a+wr /ephemeral
#! Sync back conf file
sudo rsync -a $HOME/.bashrc /home/$USER/.bashrc
sudo rsync -a $HOME/.aws /home/$USER/.aws
sudo rsync -a $HOME/.netrc /home/$USER/.netrc
