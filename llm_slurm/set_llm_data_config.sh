#!/bin/bash
#! Check if the external var has been set
if [[ -z "${DATA_TMP_DIR}" ]]; then
    echo "DATA_TMP_DIR is not set. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
#! Setting the helper
if [[ $1 = "--help" ]] || [[ $1 = "-h" ]]; then
    echo "Usage: bash set_llm_data_config.sh <split> <is_local>."
    echo -e "\t<split>: 'small' or 'full'. Default: 'full'"
    echo -e "\t<n_clients>: bool. Default: False"
    echo -e "\tExample: bash set_llm_data_config.sh full true"
    echo -e "\tNOTE: the external variable DATA_TMP_DIR must be set."
    exit 1
fi
#! Set/get the variables
DATA_VERSION="full"
IS_LOCAL=false
if [[ $# -eq 0 ]]; then
    echo "set_llm_data_config.sh: Using default values for all input arguments."
elif [[ $# -eq 1 ]]; then
    DATA_VERSION=$1
elif [[ $# -eq 2 ]]; then
    DATA_VERSION=$1
    IS_LOCAL=$2
else
    echo "Invalid number of input arguments. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
#! Export the endpoint of the S3 object store
export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Set data configuration
if [[ "$DATA_VERSION" == "small" ]]; then
    if [[ "$IS_LOCAL" == true ]]; then
        export DATA_CONFIG="llm_config.data_local=/local/scratch/small-c4 llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    else
        export DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://small-c4-dataset llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
    fi
else
    if [[ "$IS_LOCAL" == true ]]; then
        export DATA_CONFIG="llm_config.data_local=/local/scratch/c4 llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    else
        export DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://c4-dataset llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
    fi
fi
