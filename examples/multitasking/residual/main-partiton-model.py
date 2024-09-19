import os
import torch
import torch.distributed as dist
import numpy as np
from model import GPTConfig, GPT
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.ndtimeline import init_ndtimers, flush, wait
from vescale.dtensor.placement_types import Replicate
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.pipe.pipe_emmiter import ScheduleEngine
from vescale.pipe._schedules.instruction_base import get_linear_pp_module_dep2
from vescale.plan.spec import PipelineScheduleType
from torch.distributed import broadcast

# Add dataset config
dataset = "shakespeare"
block_size = 1024
batch_size = 4
local_batch_size = 4  # For each device
data_dir = os.path.join("data", dataset)

# Function to get data batch
def get_batch(split, rank, world_size, block_size=1024, batch_size=12):
    # Load data with np.memmap
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")

    # Randomly select batch of indices
    ix = torch.randint(len(data) - block_size, (batch_size,)).to(f"cuda:{rank}")

    # Broadcast to ensure same batch is selected across GPUs
    if world_size > 1:
        broadcast(ix, src=0)

    ix_split = torch.split(ix, local_batch_size)[rank]

    # Prepare inputs and targets for language modeling task
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix_split]).cuda(rank)
    y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix_split]).cuda(rank)

    return x, y

def main():
    # Initialize communication
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)


    # Initialize global device mesh
    dist.barrier()  # Add synchronization barrier before mesh initialization
    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type="cuda",
        mesh_shape=(4, 1, 1),
        mesh_dim_names=("PP", "DP", "TP"),
    )
    dist.barrier()  # Ensure all ranks complete the initialization

    # GPT model configuration
    gpt_config = GPTConfig(
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0
    )
    model = GPT(gpt_config).to(device)


    # Define pipeline stages
    # model_list = [model]  # For simplicity, treating single model instance here
    # print(model_list)


    # Define pipeline stages by splitting the GPT model manually
    model_list = [
        torch.nn.ModuleDict({
            "wte": model.transformer.wte,  # Embedding layers stay in the first stage
            "wpe": model.transformer.wpe,
            "drop": model.transformer.drop,
            "h": torch.nn.ModuleList([model.transformer.h[i] for i in range(3)]),  # First 3 layers (0-2)
        }),
        torch.nn.ModuleDict({
            "h": torch.nn.ModuleList([model.transformer.h[i] for i in range(3, 6)]),  # Next 3 layers (3-5)
        }),
        torch.nn.ModuleDict({
            "h": torch.nn.ModuleList([model.transformer.h[i] for i in range(6, 9)]),  # Next 3 layers (6-8)
        }),
        torch.nn.ModuleDict({
            "h": torch.nn.ModuleList([model.transformer.h[i] for i in range(9, 12)]),  # Last 3 layers (9-11)
            "ln_f": model.transformer.ln_f,  # Final layer norm and lm_head in the last stage
            "lm_head": model.lm_head,
        })
    ]

    # Now, `model_list` has the GPT model split evenly into 4 parts
    # print(model_list)


    deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())

    w = gpt_config.n_embd * 2 * 4
    a = gpt_config.n_embd * 4
    mem_f = 2 * w + 2 * a  # forward weight size
    mem_w = -2 * a
    mem_b = -mem_w - mem_f

    # Initialize ScheduleEngine
    pipe_engine = ScheduleEngine(
        deps=deps,
        meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
        schedule=PipelineScheduleType.ZERO_BUBBLE,
        batches=batch_size,
        data_iterator=[iter([get_batch('train', local_rank, world_size) for _ in range(batch_size)])],
        stage_id=local_rank,
        shape=(1, 1, 3),
        dtype=torch.float32,
        f_cost=6,
        b_cost=4,
        w_cost=4,
        c_cost=1,
        f_mem=mem_f,
        b_mem=mem_b,
        w_mem=mem_w,
        max_mem=mem_f * 4 * 2,
    )

    # _, all_forward = ScheduleEngine.execute(pipe_engine)

    # # Verify consistency with ground truth
    # if local_rank == 0:
    #     loss_per_microbatch = [item[1] for item in all_forward]
    #     print(loss_per_microbatch, all_batches_out)
    #     for t1, t2 in zip(loss_per_microbatch, all_batches_out):
    #         assert torch.allclose(t1, t2), "Distributed and single GPU losses do not match"

    # Synchronize and finalize
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
