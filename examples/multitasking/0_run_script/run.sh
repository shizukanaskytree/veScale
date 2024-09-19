#!/bin/bash
export NCCL_DEBUG=OFF

torchrun --nproc_per_node=4 main.py






# # 设置调试模式（如果需要的话）
# # export VESCALE_DEBUG_MODE=1



# # 设置 GPU 数量
# NUM_GPUS=4

# # 设置数据并行和张量并行的大小
# DP_SIZE=1
# TP_SIZE=1

# # 保存检查点的目录
# CHECKPOINT_DIR="./nanogpt_checkpoint_dir"

# # 执行 torchrun 启动脚本
# echo "Launching NanoGPT fine-tuning with the following parameters:"
# echo "  - Number of GPUs: $NUM_GPUS"
# echo "  - Data Parallelism Size (DP Size): $DP_SIZE"
# echo "  - Tensor Parallelism Size (TP Size): $TP_SIZE"
# echo "  - Checkpoint Directory: $CHECKPOINT_DIR"

# torchrun \
#   --standalone \
#   --nproc_per_node=$NUM_GPUS \
#   main.py \
#   config/finetune_shakespeare.py \
#   --compile=False \
#   --dp_size=$DP_SIZE \
#   --tp_size=$TP_SIZE \
#   --save_checkpoint_path=$CHECKPOINT_DIR

# # 如果想从检查点恢复训练，取消下面的注释并设置加载路径
# # torchrun \
# #   --standalone \
# #   --nproc_per_node=$NUM_GPUS \
# #   finetune_4D.py \
# #   config/finetune_shakespeare.py \
# #   --compile=False \
# #   --dp_size=$DP_SIZE \
# #   --tp_size=$TP_SIZE \
# #   --save_checkpoint_path=$CHECKPOINT_DIR \
# #   --load_checkpoint_path=${CHECKPOINT_DIR}/iter_5

# echo "NanoGPT fine-tuning process started successfully."