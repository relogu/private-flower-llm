#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
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
#! AWS S3 object store settings
aws_access_key_id=$(grep 'aws_access_key_id' ~/.aws/credentials | awk -F' = ' '{print $2}')
aws_secret_access_key=$(grep 'aws_secret_access_key' ~/.aws/credentials | awk -F' = ' '{print $2}')
export AWS_ACCESS_KEY_ID="$aws_access_key_id"
export AWS_SECRET_ACCESS_KEY="$aws_secret_access_key"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT='http://128.232.115.0:9000'
# export S3_VERIFY_SSL=0
unset S3_VERIFY_SSL
# export S3_USE_HTTPS=0
unset S3_USE_HTTPS
# export S3_REGION=us-east-1
unset S3_REGION
export AWS_LOG_LEVEL=1
#! Launch tensorboard
poetry run tensorboard --logdir s3://checkpoints/tensorboard_logs/"$1"

#! ssh -L <local_port>:<forward_to_host>:<port_on_forward_to_host> -N <username>@<node_name>.cl.cam.ac.uk
#! ssh -L 6006:localhost:6006 -N mauao
