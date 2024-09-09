### run once
# pip install sentencepiece
# cd data/shakespeare/ && python3 prepare.py && cd ../..

GPU_CNT=4
dp_size=2
tp_size=2
max_iters=10
torchrun --standalone \
    --nproc_per_node=${GPU_CNT} \
    llama_train.py \
    --dp=${dp_size} \
    --tp=${tp_size} \
    --max_iters=${max_iters} \
    --bsz 2

# bsz 2: 22,631MiB
