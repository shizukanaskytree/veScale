export NCCL_DEBUG=OFF

### step 1:
### root@d3cb3edeb6a9:~/vescale_prj/veScale/examples/nanogpt_4D_finetune/data/shakespeare#
# python prepare.py


### step 2:
### root@d3cb3edeb6a9:~/vescale_prj/veScale/examples/nanogpt_4D_finetune#
python base_train.py --batch_size=1 --compile=False --dataset="shakespeare"


### step 3:
pip install triton
torchrun --standalone --nproc_per_node=4 base_train.py --compile=False --batch_size=4 --dataset="shakespeare"

