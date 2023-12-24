#!/bin/bash
#! Check if the external var has been set
if [[ -z "${DATA_TMP_DIR}" ]]; then
    echo "DATA_TMP_DIR is not set. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
#! Setting the helper
if [[ $1 = "--help" ]] || [[ $1 = "-h" ]]; then
    echo "Usage: bash set_llm_data_config.sh <split> <is_local> <is_federated>."
    echo -e "\t<split>: 'full' or 'small'. Default: 'full'"
    echo -e "\t<is_local>: bool. Default: true"
    echo -e "\t<is_federated>: bool. Default: true"
    echo -e "\tExample: bash set_llm_data_config.sh full true"
    echo -e "\tNOTE: the external variable DATA_TMP_DIR must be set."
    exit 1
fi
#! Set/get the variables
SPLIT="full"
IS_LOCAL=false
IS_FEDERATED=true
if [[ $# -eq 0 ]]; then
    echo "set_llm_data_config.sh: Using default values for all input arguments."
elif [[ $# -eq 1 ]]; then
    SPLIT=$1
elif [[ $# -eq 2 ]]; then
    SPLIT=$1
    IS_LOCAL=$2
elif [[ $# -eq 3 ]]; then
    SPLIT=$1
    IS_LOCAL=$2
    IS_FEDERATED=$3
else
    echo "Invalid number of input arguments. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
#! Checking input arguments
if [[ "$SPLIT" != "full" ]] && [[ "$SPLIT" != "small" ]]; then
    echo "Invalid input argument for <split>. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
if [[ "$IS_LOCAL" != true ]] && [[ "$IS_LOCAL" != false ]]; then
    echo "Invalid input argument for <is_local>. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
if [[ "$IS_FEDERATED" != true ]] && [[ "$IS_FEDERATED" != false ]]; then
    echo "Invalid input argument for <is_federated>. Try 'bash set_llm_data_config.sh --help/-h' for more information."
    exit 1
fi
#! Get the splits settings
if [[ "$SPLIT" == "small" ]]; then
    SPLIT_CONFIG="llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
else
    SPLIT_CONFIG="llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
fi
#! Export the endpoint of the S3 object store
export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Set data configuration
if $IS_FEDERATED ; then
    if $IS_LOCAL ; then
        export DATA_CONFIG="llm_config.data_local=/local/scratch/fed-c4 $SPLIT_CONFIG"
    else
        export DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://fed-c4 $SPLIT_CONFIG"
    fi
else
    if $IS_LOCAL ; then
        export DATA_CONFIG="llm_config.data_local=/local/scratch/c4 $SPLIT_CONFIG"
    else
        export DATA_CONFIG="llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://c4-dataset $SPLIT_CONFIG"
    fi
fi
