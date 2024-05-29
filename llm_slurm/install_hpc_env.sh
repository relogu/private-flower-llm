#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if ! $?; then
	echo "install_hpc_env.sh: Error parsing options" >&2
	exit 1
fi

eval set -- "$OPTIONS"

while true; do
	case "$1" in
	-p | --project_path)
		PROJECT_PATH="$2"
		shift 2
		;;
	--)
		shift
		break
		;;
	*)
		break
		;;
	esac
done
echo "install_hpc_env.sh: Install env in PROJECT_PATH=$PROJECT_PATH"
#! Add modules from scratch to be sure everything works
#! Enable the module command
# shellcheck disable=SC1091
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
#! Install `pyenv`
PYENV_VER_OUTPUT=$(pyenv --version)
if [[ $PYENV_VER_OUTPUT == *"pyenv "* ]]; then
	echo "install_hpc_env.sh: pyenv is already installed."
else
	#! Getting `pyenv`
	curl https://pyenv.run | bash
fi
if [[ $PYENV_ROOT == *"pyenv"* ]]; then
	echo "install_hpc_env.sh: PYENV_ROOT variable is already set."
else
	#! Setting up 'pyenv' to execute automatically in the shell
	echo "# Load 'pyenv' automatically" >>~/.bashrc
	# shellcheck disable=SC2016
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >>~/.bashrc
	export PYENV_ROOT="$HOME/.pyenv"
	# shellcheck disable=SC2016
	echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >>~/.bashrc
	[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
	# shellcheck disable=SC2016
	echo 'eval "$(pyenv init -)"' >>~/.bashrc
	eval "$(pyenv init -)"
	echo '# # Load pyenv-virtualenv automatically' >>~/.bashrc
	# shellcheck disable=SC2016
	echo '# eval "$(pyenv virtualenv-init -)"' >>~/.bashrc
fi
#! Installing python 3.10.13
pyenv install 3.10.13
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
#! Entering the project folder
cd "$PROJECT_PATH" || exit
#! Install the poetry env no matter what
poetry install -q
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
if [[ -e $POETRY_ENV_PATH ]]; then
	echo "install_hpc_env.sh: Poetry environment exists."
	if ! [[ $(poetry check --lock) ]]; then
		echo "install_hpc_env.sh: Poetry environment is not up-to-date, updating..."
		poetry lock --no-update
	fi
else
	echo "install_hpc_env.sh: Poetry environment doesn't exist. Installing..."
	poetry config installer.max-workers 10
	poetry install -q
	POETRY_ENV_PATH=$(poetry env info --path)
fi
# shellcheck disable=SC1091
. "$POETRY_ENV_PATH"/bin/activate
#! Check the output of `nvcc -V`
NVCC_OUTPUT=$(nvcc -V)
if [[ $NVCC_OUTPUT == *"release 12.1"* ]]; then
	echo "install_hpc_env.sh: CUDA 12.1 is detected."
else
	echo "install_hpc_env.sh: CUDA 12.1 not detected. Please install CUDA 12.1. Exiting..."
	exit 1
fi
#! Install `flash-attn`
if ! poetry run pip list | grep -q flash-attn; then
	echo "install_hpc_env.sh: Installing flash-attn..."
	poetry run pip install -q flash-attn==2.3.2 --no-build-isolation
else
	echo "install_hpc_env.sh: flash-attn is already installed."
fi
#! Downgrade python warnings (default in CSD3 is 'debug')
#! From here: https://docs.python.org/3/using/cmdline.html#envvar-PYTHONWARNINGS
#! And here: https://docs.python.org/3/library/warnings.html#describing-warning-filters
export PYTHONWARNINGS="ignore::DeprecationWarning,ignore::ResourceWarning"
#! Check Python version
PYTHON_OUTPUT=$(python --version)
if [[ $PYTHON_OUTPUT == *"3.10.13"* ]]; then
	echo "install_hpc_env.sh: Python 3.10.13 is detected."
else
	echo "install_hpc_env.sh: Python 3.10.13 not detected. Please install Python 3.10.13. Exiting..."
	exit 1
fi
#! Final message
echo "install_hpc_env.sh: Environment is ready."

# TODO: Set 'TMPDIR' env var to depend on the run_uid and username
# TODO: Set 'TRITON_CACHE_DIR' not to be in an NFS filesystem
