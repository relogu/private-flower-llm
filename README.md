# Pollen: High-throughput Simulation of Federated Learning via Resource-Aware Client Placement

This repository contains the code for the paper *"High-throughput Simulation of Federated Learning via Resource-Aware Client Placement"*.
Pollen is an open-source simulator for federated learning built on top of the popular [Flower](https://github.com/adap/flower) framework.
Pollen allows simulating large-scale federated learning settings of an unprecedented scale efficiently using heterogeneous GPU devices, server configurations, and network topologies.

## Overview

Pollen is a cutting-edge simulator designed to accelerate research in Federated Learning (FL), a privacy-focused machine learning paradigm. FL involves collaborative training of models directly on edge devices, but large-scale experimentation is often hindered by the lack of efficient simulators. Pollen addresses this challenge by introducing innovative strategies to enhance scalability and realism in FL simulations.

## Key Components

**Push-Based Client Placement System:** Pollen optimizes communication efficiency by implementing a push-based client placement system, overcoming the limitations of pull-based approaches in previous simulators.

**Concurrency Estimator:** Pollen's placement model estimates the number of concurrent clients that can be simulated on a single GPU, allowing researchers to efficiently run large-scale experiments on a single GPU.

**Resource-Aware GPU Balancing:** The simulator intelligently balances clients across servers and their GPUs using a novel online machine-learning model. This not only improves overall system efficiency but also reduces GPU idle time by up to 50%.

**Accurate Training Time Predictions:** Pollen's placement model provides precise training time predictions, allowing researchers to run extensive experiments with millions of clients. This feature significantly enhances the reliability of simulations.

## Impact on the Scientific Community

Pollen empowers researchers to simulate unprecedented large-scale federated learning settings efficiently. By overcoming the limitations of existing simulators, Pollen enables scientists to test new ideas, conduct extensive experiments, and push the boundaries of FL research in a time-efficient manner. Experimental comparisons with popular FL frameworks demonstrate significant speed-ups, making Pollen a valuable tool for advancing FL research at scale.

## Experimental Evaluation

Pollen has been rigorously evaluated on four representative FL tasks, comparing its performance against ad-hoc FL frameworks such as Flower, Flute, FedScale, and Parrot. The results showcase substantial experimental speed-ups, reducing simulation times from days or weeks.

## Getting Started

Refer to the documentation and examples in this repository to quickly get started with Pollen. Join our community and contribute to the advancement of large-scale Federated Learning research!

### Example

Follow the steps below to run a simple example using Pollen.

```bash
############## MASTER NODE ##############
#! Get the task name
TASK_NAME="openimage"/"shakespeare_memory"/"google_speech"/"reddit"
#! Select the placement policy, pollen uses the following policies: "lb" (learning-based), "rr" (round robin), "bu" (batches based)
POLICY="lb"/"rr"/"bu"
#! Get the master node
MASTER_NODE=$(hostname)
#! Set the port
PORT=6381
#! Set the run uuid
run_uuid=$(uuidgen)
#! Set the custom hydra arguments
CUSTOM_HYDRA_ARGS="num_nodes=2 run_uuid=$run_uuid task=$TASK_NAME task.n_clients_per_round=100 task.num_rounds=100 local_epochs=1 placement_policy=$POLICY flwr_address=$MASTER_NODE:$PORT"

echo "STARTING POLLEN SERVER at $MASTER_NODE"
poetry run python -m pollen_worker.launch_pollen_server $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &
#! If resources are avaliable here
echo "STARTING POLLEN NODE MANAGER at $MASTER_NODE"
poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS hydra/job_logging=none hydra/hydra_logging=none &

############## SLAVE NODE(s) ##############
#! Get the slave node
SLAVE_NODE=$(hostname)
echo "STARTING POLLEN NODE MANAGER at $SLAVE_NODE"
poetry run python -m pollen_worker.node_manager $CUSTOM_HYDRA_ARGS # Make sure the Hydra's options are the same as before
```
