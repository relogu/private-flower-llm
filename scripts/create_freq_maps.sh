#!/bin/bash
# shellcheck disable=SC2090,SC2086,SC2089,SC1091
#SBATCH -c 20
#SBATCH -w ruapehu
#SBATCH --gres=gpu:0
#SBATCH --job-name=freq
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#!SBATCH --time=01:00:00
#SBATCH --dependency=afterany:11822
# Default project path
PROJECT_PATH="$HOME/projects/flower_llm"

# Parse command-line options
if ! OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@"); then
	echo "convert_hf_dataset_to_mds.sh: Error parsing options" >&2
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

#! Set/get the variables
SPLIT="full"
DATASET="the-pile"
PARITION_NAME="fed_$DATASET/c8"
LOCAL="/local/scratch/flower_llm/dataset_cache"
REMOTE="s3://fed-c4/c8"
DATASET_SUBSET="en"
MOSAICML_DATA_ROOT="/local/scratch/aai30/freq_maps_new_pile"

mkdir -p "$MOSAICML_DATA_ROOT"
if [[ $SPLIT == "full" ]]; then
	SPLIT_NAME="val train"
	echo "convert_hf_dataset_to_mds.sh: Default splits selected: $SPLIT_NAME."
else
	SPLIT_NAME="$SPLIT"
	echo "convert_hf_dataset_to_mds.sh: The selected splits are $SPLIT_NAME."
fi
echo "convert_hf_dataset_to_mds.sh: PROJECT_PATH=$PROJECT_PATH"
#! Moving to the project folder
cd "$PROJECT_PATH" || exit
#! Preparing environment
if [[ $(hostname) == *'gpu-q'* ]]; then
	echo "convert_hf_dataset_to_mds.sh: Assuming the script is executing in the CSD3."
	#! Executing the environment preparation script
	#! NOTE: Must use "." to execute, "sh" doesn't work
	. "$PROJECT_PATH"/scripts/install_hpc_env.sh
fi
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
# shellcheck disable=SC1091
. "$POETRY_ENV_PATH"/bin/activate
#! Set the data root
DATA_ROOT="$MOSAICML_DATA_ROOT/$PARITION_NAME"
echo "convert_hf_dataset_to_mds.sh: Creating the partition data root directory: $DATA_ROOT"
mkdir -p "$DATA_ROOT"
#! Get info about CPU resources available
if [ -z "${SLURM_CPUS_PER_TASK}" ]; then
	NUM_CPUS=$(nproc --all)
	export NUM_CPUS
else
	NUM_CPUS="$SLURM_CPUS_PER_TASK"
	export NUM_CPUS
fi
echo "convert_hf_dataset_to_mds.sh: Number of CPU cores available: $NUM_CPUS"
#! Export the endpoint of the S3 object store
# export S3_ENDPOINT_URL='http://mauao.cl.cam.ac.uk:9000'
#! Using directly the IP to avoid name resolution issues
export S3_ENDPOINT_URL='http://128.232.115.0:9000'

declare -a TRIPLETS=(
    "the_pile iclr2025datasets/50259/the-pile en 8 val USPTO Backgrounds" "the_pile iclr2025datasets/50259/the-pile en 8 train USPTO Backgrounds"
	"the_pile iclr2025datasets/50259/the-pile en 8 val PubMed Abstracts" "the_pile iclr2025datasets/50259/the-pile en 8 train PubMed Abstracts"
	"the_pile iclr2025datasets/50259/the-pile en 8 val Pile-CC" "the_pile iclr2025datasets/50259/the-pile en 8 train Pile-CC"
	"the_pile iclr2025datasets/50259/the-pile en 8 val Github" "the_pile iclr2025datasets/50259/the-pile en 8 train Github"
	"the_pile iclr2025datasets/50259/the-pile en 8 val NIH ExPorter" "the_pile iclr2025datasets/50259/the-pile en 8 train NIH ExPorter"
	"the_pile iclr2025datasets/50259/the-pile en 8 val EuroParl" "the_pile iclr2025datasets/50259/the-pile en 8 train EuroParl"
	"the_pile iclr2025datasets/50259/the-pile en 8 val Enron Emails" "the_pile iclr2025datasets/50259/the-pile en 8 train Enron Emails"
	"the_pile iclr2025datasets/50259/the-pile en 8 val DM Mathematics" "the_pile iclr2025datasets/50259/the-pile en 8 train DM Mathematics"
	"the_pile iclr2025datasets/50259/the-pile en 8 val PubMed Central" "the_pile iclr2025datasets/50259/the-pile en 8 train PubMed Central"
	"the_pile iclr2025datasets/50259/the-pile en 8 val Gutenberg (PG-19)" "the_pile iclr2025datasets/50259/the-pile en 8 val Gutenberg (PG-19)"
	"the_pile iclr2025datasets/50259/the-pile en 8 val FreeLaw" "the_pile iclr2025datasets/50259/the-pile en 8 train FreeLaw"
	"the_pile iclr2025datasets/50259/the-pile en 8 val Wikipedia (en)" "the_pile iclr2025datasets/50259/the-pile en 8 train Wikipedia (en)"
	"the_pile iclr2025datasets/50259/the-pile en 8 val ArXiv" "the_pile iclr2025datasets/50259/the-pile en 8 train ArXiv"
	"the_pile iclr2025datasets/50259/the-pile en 8 val StackExchange" "the_pile iclr2025datasets/50259/the-pile en 8 train StackExchange"
	"the_pile iclr2025datasets/50259/the-pile en 8 val HackerNews" "the_pile iclr2025datasets/50259/the-pile en 8 train HackerNews"
	"the_pile iclr2025datasets/50259/the-pile en 8 val PhilPapers" "the_pile iclr2025datasets/50259/the-pile en 8 train PhilPapers"
	# "the_pile fed-the-pile/c8 en 8 val USPTO Backgrounds" "the_pile fed-the-pile/c8 en 8 train USPTO Backgrounds"
	# "the_pile fed-the-pile/c8 en 8 val PubMed Abstracts" "the_pile fed-the-pile/c8 en 8 train PubMed Abstracts"
	# "the_pile fed-the-pile/c8 en 8 val Pile-CC" "the_pile fed-the-pile/c8 en 8 train Pile-CC"
	# "the_pile fed-the-pile/c8 en 8 val Github" "the_pile fed-the-pile/c8 en 8 train Github"
	# "the_pile fed-the-pile/c8 en 8 val NIH ExPorter" "the_pile fed-the-pile/c8 en 8 train NIH ExPorter"
	# "the_pile fed-the-pile/c8 en 8 val EuroParl" "the_pile fed-the-pile/c8 en 8 train EuroParl"
	# "the_pile fed-the-pile/c8 en 8 val Enron Emails" "the_pile fed-the-pile/c8 en 8 train Enron Emails"
	# "the_pile fed-the-pile/c8 en 8 val DM Mathematics" "the_pile fed-the-pile/c8 en 8 train DM Mathematics"
	# "the_pile fed-the-pile/c8 en 8 val PubMed Central" "the_pile fed-the-pile/c8 en 8 train PubMed Central"
	# "the_pile fed-the-pile/c8 en 8 val Gutenberg (PG-19)" "the_pile fed-the-pile/c8 en 8 val Gutenberg (PG-19)"
	# "the_pile fed-the-pile/c8 en 8 val FreeLaw" "the_pile fed-the-pile/c8 en 8 train FreeLaw"
	# "the_pile fed-the-pile/c8 en 8 val Wikipedia (en)" "the_pile fed-the-pile/c8 en 8 train Wikipedia (en)"
	# "the_pile fed-the-pile/c8 en 8 val ArXiv" "the_pile fed-the-pile/c8 en 8 train ArXiv"
	# "the_pile fed-the-pile/c8 en 8 val StackExchange" "the_pile fed-the-pile/c8 en 8 train StackExchange"
	# "the_pile fed-the-pile/c8 en 8 val HackerNews" "the_pile fed-the-pile/c8 en 8 train HackerNews"
	# "the_pile fed-the-pile/c8 en 8 val PhilPapers" "the_pile fed-the-pile/c8 en 8 train PhilPapers"
	# "the_pile fed-the-pile/c8 en 8 val USPTO Backgrounds" "the_pile fed-the-pile/c8 en 8 train USPTO Backgrounds"
	# "the_pile fed-the-pile/c8 en 8 val PubMed Abstracts" "the_pile fed-the-pile/c8 en 8 train PubMed Abstracts"
	# "the_pile fed-the-pile/c8 en 8 val Pile-CC" "the_pile fed-the-pile/c8 en 8 train Pile-CC"
	# "the_pile fed-the-pile/c8 en 8 val Github" "the_pile fed-the-pile/c8 en 8 train Github"
	# "the_pile fed-the-pile/c8 en 8 val NIH ExPorter" "the_pile fed-the-pile/c8 en 8 train NIH ExPorter"
	# "the_pile fed-the-pile/c8 en 8 val EuroParl" "the_pile fed-the-pile/c8 en 8 train EuroParl"
	# "the_pile fed-the-pile/c8 en 8 val Enron Emails" "the_pile fed-the-pile/c8 en 8 train Enron Emails"
	# "the_pile fed-the-pile/c8 en 8 val DM Mathematics" "the_pile fed-the-pile/c8 en 8 train DM Mathematics"
	# "the_pile fed-the-pile/c8 en 8 val PubMed Central" "the_pile fed-the-pile/c8 en 8 train PubMed Central"
	# "the_pile fed-the-pile/c8 en 8 val Gutenberg (PG-19)" "the_pile fed-the-pile/c8 en 8 val Gutenberg (PG-19)"
	# "the_pile fed-the-pile/c8 en 8 val FreeLaw" "the_pile fed-the-pile/c8 en 8 train FreeLaw"
	# "the_pile fed-the-pile/c8 en 8 val Wikipedia (en)" "the_pile fed-the-pile/c8 en 8 train Wikipedia (en)"
	# "the_pile fed-the-pile/c8 en 8 val ArXiv" "the_pile fed-the-pile/c8 en 8 train ArXiv"
	# "the_pile fed-the-pile/c8 en 8 val StackExchange" "the_pile fed-the-pile/c8 en 8 train StackExchange"
	# "the_pile fed-the-pile/c8 en 8 val HackerNews" "the_pile fed-the-pile/c8 en 8 train HackerNews"
	# "the_pile fed-the-pile/c8 en 8 val PhilPapers" "the_pile fed-the-pile/c8 en 8 train PhilPapers"
	# "c4 fed-mc4/c8/en en 8 val client" "c4 fed-mc4/c8/en en 8 train client"
	# "c4 fed-mc4/c8/it it 8 val client" "c4 fed-mc4/c8/it it 8 train client"
	# "c4 fed-mc4/c8/mr mr 8 val client" "c4 fed-mc4/c8/mr mr 8 train client"
	# "c4 fed-mc4/c8/th th 8 val client" "c4 fed-mc4/c8/th th 8 train client"
	# "c4 fed-mc4/c8/ms ms 8 val client" "c4 fed-mc4/c8/ms ms 8 train client"
	# "c4 fed-mc4/c8/fa fa 8 val client" "c4 fed-mc4/c8/fa fa 8 train client"
	# "c4 fed-mc4/c8/ar ar 8 val client" "c4 fed-mc4/c8/ar ar 8 train client"
	# "c4 fed-mc4/c8/ar hi 8 val client" "c4 fed-mc4/c8/hi hi 8 train client"
	# "c4 fed-mc4/c8/ko ko 8 val client" "c4 fed-mc4/c8/ko ko 8 train client"
	# "c4 fed-mc4/c8/ja ja 8 val client" "c4 fed-mc4/c8/ja ja 8 train client"
	# "c4 fed-mc4/c8/zh zh 8 val client" "c4 fed-mc4/c8/zh zh 8 train client"
	# "c4 fed-mc4/c8/sr sr 8 val client" "c4 fed-mc4/c8/sr sr 8 train client"
	# "c4 fed-mc4/c8/ru ru 8 val client" "c4 fed-mc4/c8/ru ru 8 train client"
	# "c4 fed-mc4/c8/da da 8 val client" "c4 fed-mc4/c8/da da 8 train client"
	# "c4 fed-mc4/c8/fr fr 8 val client" "c4 fed-mc4/c8/fr fr 8 train client"
	# "c4 fed-mc4/c8/it it 8 val client" "c4 fed-mc4/c8/it it 8 train client"
	# "c4 fed-mc4/c8/uk uk 8 val client" "c4 fed-mc4/c8/uk uk 8 train client"
	# "c4 fed-mc4/c8/bg bg 8 val client" "c4 fed-mc4/c8/bg bg 8 train client"
	# "c4 fed-c4/c128 en 128 val client" "c4 fed-c4/c128 en 128 train client"
)

mkdir -p "$MOSAICML_DATA_ROOT"

for TRIPLET in "${TRIPLETS[@]}"; do
	read -r DATASET PARTITION_NAME DATASET_SUBSET NUM_CLIENTS SPLIT_NAME PREFIX <<<"$TRIPLET"
	for ((i = 0; i < NUM_CLIENTS; i++)); do
		CLIENT_PARTITION_NAME="${PARTITION_NAME}/${PREFIX}_${i}"
		DATA_ROOT="$MOSAICML_DATA_ROOT/$CLIENT_PARTITION_NAME"
		# echo "$DATA_ROOT"
		REMOTE="s3://$CLIENT_PARTITION_NAME" # Adjusted

		LOCAL="/local/scratch/flower_llm/dataset_cache/$CLIENT_PARTITION_NAME"
		echo "Processing $CLIENT_PARTITION_NAME with data root $DATA_ROOT and remote $REMOTE"
		# Adjusting the call to the Python script:
		poetry run python -m flower_llm.dataset.create_freq_map \
			--dataset "$DATASET" \
			--data_subset "$DATASET_SUBSET" \
			--splits $SPLIT_NAME \
			--out_root "$DATA_ROOT" \
			--compression zstd \
			--concat_tokens 2048 \
			--tokenizer EleutherAI/gpt-neox-20b \
			--eos_text '<|endoftext|>' \
			--num_workers "$NUM_CPUS" \
			--local "$LOCAL" \
			--remote "$REMOTE"
	done
done
# --tokenizer_kwargs # add these if you want to pass additional kwargs to the tokenizer
# --no_wrap # set this if you want to wrap long sequences
# --bos_text # default
# --local # default
# --remote # default
# --shuffle # default
# --shuffle_seed # default
#! Remove the positional arguments
eval set --