#!/bin/bash
# shellcheck disable=SC2181
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
	echo "launch_tensorboard_s3.sh: Error parsing options" >&2
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

#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
	echo "launch_tensorboard_s3.sh: Illegal number of parameters."
	echo "Usage: launch_tensorboard_s3.sh <root_path>"
	exit 1
fi
echo "PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
if [[ -e $POETRY_ENV_PATH ]]; then
	echo "launch_tensorboard_s3.sh: Poetry environment exists."
	if ! [[ $(poetry check --lock) ]]; then
		echo "launch_tensorboard_s3.sh: Poetry environment is not up-to-date, updating..."
		poetry lock --no-update
	fi
else
	echo "launch_tensorboard_s3.sh: Poetry environment doesn't exist. Installing..."
	poetry config installer.max-workers 10
	poetry install -q
	POETRY_ENV_PATH=$(poetry env info --path)
fi
# shellcheck disable=SC1091
. "$POETRY_ENV_PATH"/bin/activate

#! Launch tensorboard
poetry run tensorboard --load_fast true --logdir "$PROJECT_PATH"/tensorboard_logs_copy

#! ssh -L <local_port>:<forward_to_host>:<port_on_forward_to_host> -N <username>@<node_name>.cl.cam.ac.uk
#! ssh -L 6006:localhost:6006 -N mauao
