#!/bin/bash
set -e

MODE="$1"
if [ -z "$MODE" ]; then
  MODE=${SM_MODE:-train}
else
  shift
fi

if [ "$MODE" = "serve" ] || [ "$MODE" = "inference" ]; then
  SCRIPT=${1:-inference.py}
  export SAGEMAKER_PROGRAM=$SCRIPT
  exec serve
else
  SCRIPT=${1:-train.py}
  export SAGEMAKER_PROGRAM=$SCRIPT
  exec python -m sagemaker_training
fi
