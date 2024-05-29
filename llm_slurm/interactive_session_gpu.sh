#!/bin/bash
# shellcheck disable=SC2181

sintr -A LANE-SL3-GPU -p ampere -N1 --gres=gpu:1 --time=01:00:00 --qos=INTR
