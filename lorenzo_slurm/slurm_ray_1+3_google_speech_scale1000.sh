#!/bin/bash
#! Slurm requests
#SBATCH --job-name=RG13-scale1k
#SBATCH --nodelist=ngongotaha,mauao
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=24
#SBATCH --output=%x-%j.out
#SBATCH --time=04:00:00
#SBATCH --dependency=afterany:78053,78060

# Load modules or your own conda environment here
cd $HOME/projects/pollen_worker
poetry shell

#! Set up the redis password
redis_password=$(uuidgen)
export redis_password
#! Set up Ray's ports
main_port=8379
port1=8700
port2=8701
port3=10001
port4=8702
port5=10002
port6=19999
#! Get the IP address of the head node
ip=$(hostname --ip-address)
#! If we detect a space character in the head node IP, we'll
#! convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  ip=${ADDR[1]}
else
  ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $ip"
fi
ip_head=$ip:$main_port
export ip_head
echo "IP Head: $ip_head"

export NUM_GPUS=`echo $CUDA_VISIBLE_DEVICES | awk 'BEGIN{FS=","};{print NF}'`
echo "NUM_GPUS: $NUM_GPUS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"

#! NOTE: Forcing the head node to be the current node
#! that will be used for executing the simulation script
this_hostname=$(hostname)
echo "STARTING HEAD at $this_hostname"
srun --nodes=1 --ntasks=1 -w "$this_hostname" \
  poetry run ray start --head --node-ip-address=$ip --num-gpus=1 --num-cpus=${SLURM_CPUS_PER_TASK} \
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
      --log-style="record" \
      --block &

sleep 30

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
# Number of total nodes minus 1
num_nodes=$((SLURM_JOB_NUM_NODES - 1))
# Launching one worker per node that is different from the head node
for ((i = 0; i <= num_nodes; i++)); do
    node_i=${nodes_array[$i]}
    if [ "$node_i" != "$this_hostname" ]; then
        echo "Starting WORKER $i at $node_i"
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            poetry run ray start --address "$ip_head" --redis-password="$redis_password" --num-gpus=${NUM_GPUS} --num-cpus=${SLURM_CPUS_PER_TASK} --log-style="record" --block &
    fi
done

sleep 30

poetry run ray status

# ===== Call your code below =====
# Set the custom hydra arguments that will be passed to the server and the node manager
CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=google_speech task.n_clients_per_round=1000 task.num_rounds=10 local_epochs=1 ray_address="auto" ray_redis_password=$redis_password ray_node_ip_address=$ip"

echo "LAUNCHING SIMULATION at $this_hostname"

echo "LAUNCHING SIMULATION at this_hostname"
poetry run python -m pollen_worker.ray_simulation $CUSTOM_HYDRA_ARGS 

# How to use this script? Use what follows for a interactive job
# srun --nodelist mauao,ngongotaha --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:1 --partition=interactive bash lorenzo_slurm/slurm_ray_multinode.sh
# Use what follows for a batch job
# sbatch lorenzo_slurm/slurm_ray_multinode.sh
