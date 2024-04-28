#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"

# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
	echo "set_llm_data_config.sh: Error parsing options" >&2
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
echo "set_llm_data_config.sh: PROJECT_PATH=$PROJECT_PATH"
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "set_llm_data_config.sh: Assuming the script is executing in the CSD3."
	export DATA_TMP_DIR="$HOME/rds/rds-ndl32-camlsys-DNlKPrIaphU/datasets"
elif [[ $(hostname) == *'mauao'* ]]; then
	echo "set_llm_data_config.sh: Assuming the script is executing in Mauao."
	export DATA_TMP_DIR="/local/scratch/$USER/tmp"
elif [[ $(hostname) == *'ruapehu'* ]]; then
	echo "set_llm_data_config.sh: Assuming the script is executing in ruapehu."
	export DATA_TMP_DIR="/local/scratch/$USER/tmp"
else
	echo "set_llm_data_config.sh: Assuming the script is executing NOT in the CSD3 and not in Mauao (Fluidstack)."
	export DATA_TMP_DIR="/ephemeral/$USER/tmp"
fi
mkdir -p $DATA_TMP_DIR
#! Check if the external var has been set
if [[ -z ${DATA_TMP_DIR} ]]; then
	echo "set_llm_data_config.sh: DATA_TMP_DIR is not set. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
#! Setting the helper
if [[ $1 == "--help" ]] || [[ $1 == "-h" ]]; then
	echo "set_llm_data_config.sh: Usage: bash set_llm_data_config.sh <split> <is_local> <is_federated> <n_clients>."
	echo -e "\t<split>: 'full' or 'small'. Default: 'full'"
	echo -e "\t<is_local>: bool. Default: true"
	echo -e "\t<is_federated>: bool. Default: true"
	echo -e "\t<n_clients>: integer. Default: 8"
	echo -e "\tExample: bash set_llm_data_config.sh full true"
	echo -e "\tNOTE: the external variable DATA_TMP_DIR must be set."
	exit 1
fi
#! Set/get the variables
SPLIT="full"
IS_LOCAL=false
IS_FEDERATED=true
N_CLIENTS=8
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
elif [[ $# -eq 4 ]]; then
	SPLIT=$1
	IS_LOCAL=$2
	IS_FEDERATED=$3
	N_CLIENTS=$4
else
	echo "set_llm_data_config.sh: Invalid number of input arguments. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
#! Checking input arguments
if [[ $SPLIT != "full" ]] && [[ $SPLIT != "small" ]]; then
	echo "set_llm_data_config.sh: Invalid input argument for <split>, got $SPLIT. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
if [[ $IS_LOCAL != true ]] && [[ $IS_LOCAL != false ]]; then
	echo "set_llm_data_config.sh: Invalid input argument for <is_local>, got $IS_LOCAL. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
if [[ $IS_FEDERATED != true ]] && [[ $IS_FEDERATED != false ]]; then
	echo "set_llm_data_config.sh: Invalid input argument for <is_federated>, got $IS_FEDERATED. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
if [[ $N_CLIENTS -lt 1 ]]; then
	echo "set_llm_data_config.sh: Invalid input argument for <n_clients>, got $N_CLIENTS. Try 'bash set_llm_data_config.sh --help/-h' for more information."
	exit 1
fi
#! Get the splits settings
if [[ $SPLIT == "small" ]]; then
	SPLIT_CONFIG="llm_config.eval_loader.dataset.split=val_small llm_config.train_loader.dataset.split=train_small"
else
	SPLIT_CONFIG="llm_config.eval_loader.dataset.split=val llm_config.train_loader.dataset.split=train"
fi
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'
#! Set data configuration
if $IS_FEDERATED; then
	DATA_CONFIG="shared.dataset_config.is_federated=true"
	if $IS_LOCAL; then
		export DATA_CONFIG="$DATA_CONFIG shared.dataset_config.is_local=true llm_config.data_local=/local/scratch/\{\}/c$N_CLIENTS $SPLIT_CONFIG"
	else
		export DATA_CONFIG="$DATA_CONFIG shared.dataset_config.is_local=false llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://\{\}/c$N_CLIENTS $SPLIT_CONFIG"
	fi
else
	DATA_CONFIG="shared.dataset_config.is_federated=false"
	if $IS_LOCAL; then
		export DATA_CONFIG="$DATA_CONFIG shared.dataset_config.is_local=true llm_config.data_local=/local/scratch/\{\} $SPLIT_CONFIG"
	else
		export DATA_CONFIG="$DATA_CONFIG shared.dataset_config.is_local=false llm_config.data_local=$DATA_TMP_DIR llm_config.data_remote=s3://\{\} $SPLIT_CONFIG"
	fi
fi

echo "set_llm_data_config.sh: arguments=$@ first argument=$1"

#! Remove the positional arguments
eval set --
