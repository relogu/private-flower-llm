#!/bin/bash
#SBATCH -c 8
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --job-name=RO1-scale
#SBATCH --partition=normal
#SBATCH --tasks-per-node=1
#SBATCH --output=%x-%j.out
#!SBATCH --dependency=afterany:77883

# Get the timestamp and the unique run id
timestamp=$(date +%Y-%m-%d_%H%M%S)
run_uuid=$(uuidgen)
# \activate the environment and go to the pollen_worker directory
cd $HOME/projects/pollen_worker
#! Activate Poetry environment
POETRY_ENV_PATH=$(poetry env info --path)
. $POETRY_ENV_PATH/bin/activate

# Set up the redis password
redis_password=$(uuidgen)
export redis_password
# Set up the head node IP address
ip="localhost"
main_port=8380
port1=8703
port2=8704
port3=10003
port4=8705
port5=10004
port6=20000

export NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}'`

ip=$(hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

# for n_clients_per_round in "1000" "10000"; do
for n_clients_per_round in "10000"; do
  # Start Ray session
  poetry run ray start --head --node-ip-address=$ip --num-gpus=${NUM_GPUS} --num-cpus=${SLURM_CPUS_PER_TASK} \
      --port=$main_port \
      --node-manager-port=$port1 \
      --object-manager-port=$port2 \
      --ray-client-server-port=$port3 \
      --redis-shard-ports=$port4 \
      --min-worker-port=$port5 \
      --max-worker-port=$port6 \
      --redis-password="$redis_password" \
      --verbose \
      --include-dashboard=False \
      --block &

  sleep 30

  poetry run ray status

  sleep 5

  # Set the custom hydra arguments that will be passed to the server and the node manager
  CUSTOM_HYDRA_ARGS="run_uuid=$run_uuid task=openimage task.n_clients_per_round=$n_clients_per_round task.num_rounds=100 local_epochs=1 ray_address=auto ray_redis_password=$redis_password ray_node_ip_address=$ip"

  # Launch the server, uncomment the end of the line if you what separed output logs.
  poetry run python -m pollen_worker.ray_simulation $CUSTOM_HYDRA_ARGS

  # Stop Ray session
  poetry run ray stop

done

# How to use this script? Use what follows for a interactive job
# srun -w mauao -c 11 --gres=gpu:1 --partition=interactive bash slurm_ray_singlenode.sh
# Use what follows for a batch job
# sbatch lorenzo_slurm/slurm_ray_singlenode.sh