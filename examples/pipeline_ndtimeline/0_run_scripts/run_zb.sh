# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_P2P_DISABLE=1

torchrun --standalone --nproc_per_node=4 zb.py
