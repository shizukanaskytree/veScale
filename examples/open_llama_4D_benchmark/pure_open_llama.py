"""
single gpu testing.
OOM for 24GB GPU
"""

import os
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoConfig, LlamaModel

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--total_bsz", type=int, default=1)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--no-ckpt", action="store_true")
args = parser.parse_args()

bsz = args.total_bsz
s = 2

# Initialize model
if args.no_ckpt:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = AutoConfig.from_pretrained(os.path.join(dir_path, "config.json"))
    model = LlamaModel(config)
else:
    # Use pre-trained model
    model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_3b")
    model = model.model
    config = model.config

# Ensure sequence length is within max position embeddings
assert s <= config.max_position_embeddings

# Transfer model to single GPU
model = model.cuda()

# Generate input tensor
input = torch.randint(low=0, high=config.vocab_size, size=(bsz, s)).cuda()

# Set up optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Timing events for performance measurement
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# --------  warm up --------
for _ in range(args.warmup):
    optimizer.zero_grad()
    output = model(input).last_hidden_state
    loss = output.mean()
    loss.backward()
    optimizer.step()

# --------  training loop --------
start.record()
for _ in range(args.iter):
    optimizer.zero_grad()
    output = model(input).last_hidden_state
    loss = output.mean()
    loss.backward()
    optimizer.step()
end.record()

# Synchronize and calculate execution time
torch.cuda.synchronize()
exec_t = start.elapsed_time(end) / 1000 / args.iter

# Output training performance
print(f"1 iter time: {exec_t}")