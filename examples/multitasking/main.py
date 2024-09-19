import os
import time
import math
import pickle
import inspect
import sys
import numpy as np


import torch
from torch.distributed import broadcast, all_reduce, barrier, init_process_group, destroy_process_group, get_rank
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.nn import functional as F

from vescale.dtensor.dtensor import DTensor
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale import distribute_tensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.placement_types import Replicate
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.optim.base_optimizer import BasicOptimizer, GradOptimizerHookBase
from sharding_plan import nanoGPT_plan
from vescale.dtensor.random import manual_seed
from vescale.initialize.deferred_init import deferred_init, is_deferred
from vescale.plan import (
    PipelineParallelPlan,
    PipelineScheduleType,
    PipelineSplitMethodType,
    ModeType,
    TracerType,
)
from vescale.pipe import PipeModule, construct_stage_modules, construct_pipeline_split_graph
from vescale.engine import PipeEngine
from vescale.pipe.pipe_stage import construct_pipeline_stage

from model import GPTConfig, GPT


def debug_at_rank_n(rank_id):
    """If distributed is initialized, print only on rank n."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank_id:
            message = f'debug at rank {torch.distributed.get_rank()}'
            # print(message, flush=True)
            ### print yellow color
            print(f"\033[93m{message}\033[00m", flush=True)
            import debugpy
            debugpy.listen(5678)
            debugpy.wait_for_client()
            debugpy.breakpoint()
    else:
        message = 'You are not in distributed mode.'
        print(message, flush=True)


# Dataset Class for Shakespeare Dataset
class ShakespeareDataset(Dataset):
    def __init__(self, split="train", block_size=1024, data_dir="data/shakespeare"):
        self.data_dir = data_dir
        self.block_size = block_size
        if split == "train":
            self.data_path = os.path.join(self.data_dir, "train.bin")
        else:
            self.data_path = os.path.join(self.data_dir, "val.bin")
        self.data = np.memmap(self.data_path, dtype=np.uint16, mode="r")  # Load the binary data as memory-mapped array

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx: idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1: idx + 1 + self.block_size].astype(np.int64))
        return x, y



# Function to get DataLoader for distributed setup
def get_data_loader(split, batch_size, local_batch_size, world_size, ddp_rank, block_size=1024):
    dataset = ShakespeareDataset(split, block_size=block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=ddp_rank)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4
    )
    return loader





def print_distributed_model(model: nn.Module):
    """
    打印分布式后的模型参数分布信息，包括张量形状、分片方式和设备网格。
    """
    print(model.repr_params(
        show_shape=True,       # 显示参数的形状
        show_type=True,        # 显示参数的数据类型
        show_shard=True,       # 显示参数的分片方式
        show_mesh=True,        # 显示设备网格的信息
        show_ltensor_shape=True  # 显示本地张量的形状
    ))


def main():
    world_size = int(os.environ.get("WORLD_SIZE", 4))  # Default to 4 if WORLD_SIZE is not set
    rank = int(os.environ.get("RANK", 0))  # Default to 0 if RANK is not set


    # Create the 'logs' directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Redirect stdout and stderr to a file named "logs/output_{rank}.log"
    log_filename = os.path.join(log_dir, f"output_{rank}.log")
    log_file = open(log_filename, 'w')

    # Redirect both stdout and stderr to the log file
    sys.stdout = log_file
    sys.stderr = log_file

    # print(f"Logs for rank {rank} will be written to {log_filename}")




    ### dataloader
    ### Block Size: The block size is set to 1024, which determines the length of the input sequence (the number of tokens in each example). Each training example consists of 1024 tokens.
    block_size = 1024
    ### Batch Size: In your code, you have set the batch size as 8. This means that the DataLoader will fetch 8 examples in each batch during training or inference.
    batch_size = 8


    # Model configuration parameters
    n_layer = 12       # Number of transformer layers in the GPT model
    n_head = 12        # Number of attention heads in each attention layer
    n_embd = 768       # Embedding size of the model (dimensionality of the model)
    block_size = 1024  # Length of the input sequence (number of tokens)
    vocab_size = 50304 # Vocabulary size (default GPT-2 vocabulary size)
    dropout = 0.1      # Dropout rate for regularization
    bias = False       # True: bias in Linears and LayerNorms, like GPT-2; False: a bit better and faster


    ### pipeline parallelism
    local_batch_size = batch_size
    world_size = 4
    # dp_size = 1
    # tp_size = 1
    ddp_rank = dist.get_rank() if dist.is_initialized() else 0

    backend = "nccl"  # 'nccl', 'gloo', etc.
    dist.init_process_group(backend=backend)

    ### debug at rank 0
    # debug_at_rank_n(3) # 有效


    # device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")


    torch.manual_seed(9999)

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # model.to(device)  # Ensure model is on the correct device






    pipe_plan = PipelineParallelPlan(
        mode=ModeType.GRAPH_EAGER,
        split_method=PipelineSplitMethodType.MANUAL,
        num_stages=4,
        virtual_chunks=1,
        smallest_unsplittable_units=[f"transformer.h.{i}" for i in range(n_layer)],
        split_points=["transformer.h.3", "transformer.h.6", "transformer.h.9"],
        batch_p2p_comm=False,
        overlap_p2p_comm=True,
        schedule_type=PipelineScheduleType.SIMPLE_1F1B,
        forward_only=False,
    )

    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type="cuda",
        mesh_shape=(4, 1, 1),
        mesh_dim_names=["PP", "DP", "TP"],
    )


    # stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
    #     model,
    #     pipe_config,
    #     VESCALE_DEVICE_MESH,
    #     update_split_points=True,
    # )


    ### return PipeModule
    pipe_module = construct_pipeline_stage(
        model,
        pipe_plan,
        VESCALE_DEVICE_MESH,
        lr_scheduler=None,
        update_split_points=True,
    )



    # Ensure `ddp_models` is correctly initialized
    # ddp_models = [DDP(stage_module) for stage_module in stage_modules]

    # # make ddp module
    # ddp_models = []
    # for model_chunk in model_chunks:
    #     ddp_models.append(
    #         DDP(
    #             model_chunk,
    #             VESCALE_DEVICE_MESH.get_data_parallel_mesh(),
    #             accumulate_allreduce_grads_in_fp32=True,
    #             overlap_grad_reduce=True,
    #             use_distributed_optimizer=True,
    #         )
    #     )





    # # Optimizer hyperparameters
    learning_rate = 6e-4  # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    betas = (beta1, beta2)
    device_type = "cuda"



    ### filter out those that do not require grad
    # param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    ### create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    ### i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    # decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    # nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]



    # pipe_module.parameters()
    param = [p for p in pipe_module.parameters() if p.requires_grad]
    decay_params = [p for p in param if p.dim() >= 2]
    nodecay_params = [p for p in param if p.dim() < 2]



    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

    ### Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    # extra_args = dict(fused=True) if use_fused else dict() # error fused
    extra_args = dict()

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    print(f"using fused AdamW: {use_fused}")
    ## return optimizer





    basic_optimizer = BasicOptimizer(optimizer, models=pipe_module)






    # # optimizer = model.configure_optimizers(
    # #                 weight_decay,
    # #                 learning_rate,
    # #                 (beta1, beta2),
    # #                 device_type)

    # ### if ddp
    # # basic_optimizer = BasicOptimizer(optimizer, models=stage_modules) # only for test purpose

    # # doptim = DistributedOptimizer(
    # #     # torch.optim.Adam(stage_modules.parameters(), lr=0.01),
    # #     optimizer,
    # #     models=ddp_models,
    # #     overlap_param_gather=False,
    # # )




    # pipe_module = PipeModule(
    #                 stage_modules,
    #                 optimizer,
    #                 None,
    #                 stage_dependency,
    #                 p2p_index_mapping,
    #                 pipe_config)




    # loss_fn = lambda logits, targets: \
    #     F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    def loss_fn(logits, targets):
        print(f"shape of logits: {logits.shape}, shape of targets: {targets.shape}")
        """
        Logits shape before reshaping: torch.Size([8, 1, 50304])
        Targets shape before reshaping: torch.Size([8, 1024])
        """


        """
        ideal:

        logits.shape
        torch.Size([4, 1024, 50304])

        logits.view(-1, logits.size(-1)).shape
        torch.Size([4096, 50304])

        targets.shape
        torch.Size([4, 1024])

        targets.view(-1).shape
        torch.Size([4096])
        """


        """
        ideal:
        logits.shape
        torch.Size([1, 1024, 50304])
        """
        # Reshape logits and targets
        logits = logits.view(-1, logits.size(-1))
        """
        ideal:
        logits.view(-1, logits.size(-1)).shape
        torch.Size([1024, 50304])
        """


        """
        ideal:
        targets.shape
        torch.Size([1, 1024])
        """
        targets = targets.view(-1)
        """
        ideal:
        targets.view(-1).shape
        torch.Size([1024])
        """

        print(f"after reshape: shape of logits: {logits.shape}, shape of targets: {targets.shape}")
        ### in this code: after reshape: shape of logits: torch.Size([8, 50304]), shape of targets: torch.Size([8192])

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return loss


    pipe_module.stage_modules[0].to(device)
    # print(pipe_module.stage_modules)

    engine = PipeEngine(
        pipe_module,
        VESCALE_DEVICE_MESH,
        loss_fn,
        pipe_plan,
    )






    # ### explicitly use the dataloader without function call
    # dataset = ShakespeareDataset(split="train", block_size=block_size)
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=ddp_rank)
    # train_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     sampler=sampler,
    #     pin_memory=True,
    #     num_workers=4
    # )

    # # ### test the train_loader
    # # for batch_idx, (x, y) in enumerate(get_batch(train_loader, device, world_size, tp_mesh)):
    # #     print(f"rank: {dist.get_rank()}, x shape: {x.shape}, y shape: {y.shape}")
    # #     print(f"rank: {dist.get_rank()}, type of x: {type(x)}, type of y: {type(y)}")
    # #     break



    # # make optimizer
    # # doptim = DistributedOptimizer(
    # #     torch.optim.Adam(split_graph.parameters(), lr=0.01),
    # #     models=ddp_models,
    # #     overlap_param_gather=False,
    # # )
    # # tp_mesh = VESCALE_DEVICE_MESH.get_tensor_parallel_mesh()
    # # stage_id = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()



    # def get_batch(split, bsz=batch_size, lbsz=local_batch_size):
    #     """
    #     Deterministic data loader for loss match:
    #     This data loader ensures that the mini-batch sampling has identical behavior no matter how many GPUs are used.
    #     """
    #     data_dir = "data/shakespeare"

    #     if split == "train":
    #         data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    #     else:
    #         data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    #     # Get random indices for the batch
    #     ix = torch.randint(len(data) - block_size, (bsz,)).to(device)

    #     # Broadcast indices to all processes if distributed
    #     if world_size > 1:
    #         torch.distributed.broadcast(ix, src=0, async_op=False)

    #     ix = torch.split(ix, lbsz)[ddp_rank]  # Split indices among data parallel ranks

    #     # Fetch the batch of data
    #     x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

    #     device_type = "cuda"
    #     # Move tensors to device
    #     if device_type == "cuda":
    #         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    #     else:
    #         x, y = x.to(device), y.to(device)

    #     if world_size > 1:
    #         # Convert to DTensor for distributed operation
    #         x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])
    #         y = distribute_tensor(y, VESCALE_DEVICE_MESH["TP"], [Replicate()])

    #     # Ensure returning exactly two tensors
    #     return x, y





    def get_batch(split, bsz=batch_size, lbsz=local_batch_size):
        """
        Deterministic data loader for loss match:
        This data loader ensures that the mini-batch sampling has identical behavior no matter how many GPUs are used.
        """
        data_dir = "data/shakespeare"

        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

        # Get random indices for the batch
        ix = torch.randint(len(data) - block_size, (bsz,)).to(device)

        # Broadcast indices to all processes if distributed
        if world_size > 1:
            torch.distributed.broadcast(ix, src=0, async_op=False)

        ix = torch.split(ix, lbsz)[ddp_rank]  # Split indices among data parallel ranks

        # Fetch the batch of data
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])

        # Move tensors to device
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

        # Convert tensors to DTensor for distributed operation across devices
        # if world_size > 1:
        # x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])
        # y = distribute_tensor(y, VESCALE_DEVICE_MESH["TP"], [Replicate()])

        # Ensure returning exactly two tensors
        return x, y



    # Fetch the batch of data
    x, y = get_batch("train", batch_size, local_batch_size)
    print(f"x shape: {x.shape}, y shape: {y.shape}")

    # Convert the batch into an iterator or a list as needed
    data_iterator = [(x.to(device), y.to(device))]  # Adjust depending on how your engine expects input

    # data_iterator = [x, y]  # Data has already been converted to DTensor by get_batch()


    # Pass the data to the engine for processing
    minibatch_loss, outputs = engine(data_iterator)

    # Print loss and outputs
    print(f"Minibatch Loss: {minibatch_loss}")
    print(f"Outputs: {outputs}")



    ### Close the file when done
    sys.stdout.close()






if __name__ == "__main__":
    main()
