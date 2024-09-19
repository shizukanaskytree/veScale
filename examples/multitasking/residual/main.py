import os
import time
import pickle
import math
import torch
import numpy as np
from torch.distributed import broadcast, all_reduce, init_process_group, get_rank, destroy_process_group
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.dmodule.api import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.dtensor.placement_types import Replicate
from vescale.pipe import construct_pipeline_split_graph
from vescale.plan import PipelineParallelPlan, PipelineSplitMethodType, TracerType
from sharding_plan import nanoGPT_plan, nanoGPT_plan_dist_dropout
from nanogpt import GPTConfig, GPT
import vescale
import sys
from vescale.initialize.deferred_init import deferred_init, is_deferred

# Global Configurations
out_dir = "out"
eval_interval = 2000
log_interval = 10
eval_iters = 200
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = "nccl"
device = "cuda"
dtype = "float16"
dp_size = 1
tp_size = 1
DDP_grads_in_fp32 = True
use_dist_dropout = True
save_checkpoint_path = "./nanogpt_checkpoint_dir"
load_checkpoint_path = ""
async_checkpoint = False
broadcast_checkpoint = False
dataset = "shakespeare"

# Define data_dir
data_dir = os.path.join("data", dataset)

def build_device_mesh(device_type="cuda", pp_size=4, dp_size=1, tp_size=1):
    mesh_shape = (pp_size, dp_size, tp_size)
    mesh_dim_names = ["PP", "DP", "TP"]
    return VESCALE_DEVICE_MESH.init_device_mesh(device_type=device_type, mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)

def get_batch(split, world_size, block_size, batch_size, device):
    data_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,)).to(device)
    if world_size > 1:
        broadcast(ix, src=0)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    if device.startswith("cuda"):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    return x, y

def estimate_loss(model, eval_iters, device, world_size):
    losses = torch.zeros(eval_iters).to(device)
    for k in range(eval_iters):
        X, Y = get_batch("val", world_size, block_size, batch_size, device)
        logits, loss = model(X, Y)
        losses[k] = loss.item() / world_size
    if world_size > 1:
        all_reduce(losses)
    return losses.mean()

def configure_optimizers(ddp_models):
    params = []
    for model in ddp_models:
        params += [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    return DistributedOptimizer(optimizer, models=ddp_models, clip_grad=grad_clip, grad_to_fp32=DDP_grads_in_fp32)

def debug_at_rank_n(rank_id):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank_id:
            message = f'debug at rank {torch.distributed.get_rank()}'
            print(f"\033[93m{message}\033[00m", flush=True)
            import debugpy
            debugpy.listen(5678)
            debugpy.wait_for_client()
            debugpy.breakpoint()
    else:
        message = 'You are not in distributed mode.'
        print(message, flush=True)

def main():
    # Initialize process group for distributed training
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl", init_method="env://")
        # debug_at_rank_n(0)

    world_size = dp_size * tp_size
    device = f"cuda:{get_rank()}" if world_size > 1 else "cuda:0"
    torch.cuda.set_device(device)

    # Initialize device mesh
    device_mesh = build_device_mesh(device_type="cuda", pp_size=4, dp_size=dp_size, tp_size=tp_size)

    # Model initialization
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, dropout=dropout)
    model = GPT(GPTConfig(**model_args)).to(device)
    print(model)

    # Construct pipeline parallel plan and use construct_pipeline_split_graph
    boundaries = [f"transformer.h.{i}" for i in range(n_layer)]  # Updated boundaries using actual model names

    pipe_config = PipelineParallelPlan(
        num_stages=4,
        split_method=PipelineSplitMethodType.MANUAL,
        smallest_unsplittable_units=boundaries,
        split_points=["transformer.h.2", "transformer.h.5", "transformer.h.8"],
        tracer_type=TracerType.TORCH_FX,
        tracer_kwargs={"shard_plan": nanoGPT_plan_dist_dropout},
        virtual_chunks=1  # Ensure we aren't adding extra partitions
    )

    # Apply pipeline parallelism using construct_pipeline_split_graph
    split_graph = construct_pipeline_split_graph(model, pipe_config, update_split_points=True)

    # Now we split the stages and parallelize them
    model_chunks = []
    plan = nanoGPT_plan_dist_dropout if use_dist_dropout else nanoGPT_plan
    for i in range(pipe_config.num_stages):
        stage = getattr(split_graph, f"stage{i}")
        stage = parallelize_module(stage, VESCALE_DEVICE_MESH.get_tensor_parallel_mesh(), nanoGPT_plan_dist_dropout, factory=False)
        assert not is_deferred(stage)
        model_chunks.append(stage)

    # Make ddp module
    ddp_models = []
    for model_chunk in model_chunks:
        ddp_models.append(
            DDP(
                model_chunk,
                VESCALE_DEVICE_MESH.get_data_parallel_mesh(),
                accumulate_allreduce_grads_in_fp32=DDP_grads_in_fp32,
                overlap_grad_reduce=True,
                use_distributed_optimizer=True,
            )
        )

    # Optimizer
    optimizer = configure_optimizers(ddp_models)

    # Training Loop
    iter_num = 0
    t0 = time.time()
    while iter_num < max_iters:
        lr = max(min_lr, learning_rate * min(1.0, iter_num / warmup_iters))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch data and forward pass
        X, Y = get_batch("train", world_size, block_size, batch_size, device)
        optimizer.zero_grad()
        stage_id = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()

        # Forward pass with only input X
        logits = ddp_models[stage_id](X)

        # Compute loss separately
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), Y.view(-1))

        loss.backward()
        optimizer.step()

        # Logging and evaluation
        if iter_num % log_interval == 0:
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {loss.item():.4f}, time {dt*1000:.2f}ms")
            t0 = time.time()

        if iter_num % eval_interval == 0:
            val_loss = estimate_loss(ddp_models[0], eval_iters, device, world_size)
            print(f"Eval: step {iter_num}, val_loss {val_loss:.4f}")
            # Save model checkpoint
            torch.save({"model": [m.state_dict() for m in ddp_models], "optimizer": optimizer.state_dict()}, f"{save_checkpoint_path}/ckpt_{iter_num}.pth")

        iter_num += 1

    # Cleanup
    if world_size > 1:
        destroy_process_group()

if __name__ == "__main__":
    # Load and override configuration from configurator.py
    config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]

    # Read configurations from 'configurator.py' file to override default settings
    exec(open('configurator.py').read())  # overrides from command line or config file

    # Collect configurations into the config dictionary
    config = {k: globals()[k] for k in config_keys}

    # Call the main function to start training
    main()
    