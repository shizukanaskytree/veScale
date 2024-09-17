NUM_GPUS=4
torchrun \
  --standalone \
  --nproc_per_node=$NUM_GPUS \
  gpt_vescale_dataloader.py
