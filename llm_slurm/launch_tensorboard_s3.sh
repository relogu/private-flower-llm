#!/bin/bash
#! Check if at least one arguments are passed
if [[ $# -lt 1 ]]; then
    echo "Illegal number of parameters."
    echo "Usage: launch_tensorboard_s3.sh <run_name>"
    exit 1
fi
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
    POETRY_ENV_PATH=$(poetry env info --path)
fi
. $POETRY_ENV_PATH/bin/activate
#! AWS S3 object store settings
aws_access_key_id=$(grep 'aws_access_key_id' ~/.aws/credentials | awk -F' = ' '{print $2}')
aws_secret_access_key=$(grep 'aws_secret_access_key' ~/.aws/credentials | awk -F' = ' '{print $2}')
export AWS_ACCESS_KEY_ID=$aws_access_key_id
export AWS_SECRET_ACCESS_KEY=$aws_secret_access_key
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
poetry run tensorboard --logdir s3://checkpoints/tensorboard_logs/$1