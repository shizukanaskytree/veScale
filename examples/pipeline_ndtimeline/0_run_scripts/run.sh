#!/bin/bash

# export VESCALE_DEBUG_MODE=1
export OMP_NUM_THREADS=2
# Set the number of GPUs
NUM_GPUS=4

# Run the finetuning process with the specified parallelism sizes
torchrun \
  --standalone \
  --nproc_per_node=$NUM_GPUS \
  simple_pp.py
