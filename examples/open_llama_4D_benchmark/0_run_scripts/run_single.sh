torchrun --standalone --nnodes=1 --nproc-per-node=4 ./run_open_llama_w_vescale.py --dp=2 --tp=2 --warmup=10 --iter=40