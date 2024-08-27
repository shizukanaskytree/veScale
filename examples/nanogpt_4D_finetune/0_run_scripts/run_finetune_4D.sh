#!/bin/bash

# export VESCALE_DEBUG_MODE=1

# Set the number of GPUs
NUM_GPUS=4

# Data Parallelism Size (DP Size) and Tensor Parallelism Size (TP Size)
# - DP Size should be chosen based on the number of available GPUs.
# - TP Size should be chosen based on the model's structure and the available GPU memory.
# - A common approach is to balance both DP and TP to optimize performance and memory usage.

# Example: If you have 4 GPUs and want to balance between DP and TP
DP_SIZE=2
TP_SIZE=2

# Explanation:
# With 4 GPUs, setting DP_SIZE=2 and TP_SIZE=2 allows you to distribute the data across 2 GPUs
# and split the model tensor operations across the other 2 GPUs. This balances the load and
# utilizes the GPUs effectively. Adjust these sizes based on the specific needs of your model and hardware.

# Path to save checkpoints
CHECKPOINT_DIR="./nanogpt_checkpoint_dir"

# Run the finetuning process with the specified parallelism sizes
torchrun \
  --standalone \
  --nproc_per_node=$NUM_GPUS \
  finetune_4D.py \
  config/finetune_shakespeare.py \
  --compile=False \
  --dp_size=$DP_SIZE \
  --tp_size=$TP_SIZE \
  --save_checkpoint_path=$CHECKPOINT_DIR

# To resume training from a checkpoint, add --load_checkpoint_path
# Example:
# --load_checkpoint_path=${CHECKPOINT_DIR}/iter_5
