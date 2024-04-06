#!/bin/bash
# Default project path
PROJECT_PATH="$HOME/projects/pollen_worker"


# Parse command-line options
OPTIONS=$(getopt -o p: --long project_path: -n 'parse-options' -- "$@")
if [ $? -ne 0 ]; then
  echo "Error parsing options" >&2
  exit 1
fi

eval set -- "$OPTIONS"

while true; do
  case "$1" in
    -p | --project_path )
      PROJECT_PATH="$2"; shift 2 ;;
    -- )
      shift; break ;;
    * )
      break ;;
  esac
done


sintr -A LANE-SL3-GPU -p ampere -N1 --gres=gpu:1 --time=01:00:00 --qos=INTR
