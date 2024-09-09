

torchrun \
    --nnodes=3 \
    --nproc-per-node=4 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    ./run_open_llama_w_vescale.py \
    --dp=12 \
    --tp=2 \
    --warmup=10 --iter=40