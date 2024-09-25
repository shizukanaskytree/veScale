import os
import time
import math
import pickle
import inspect
import sys
import numpy as np
from collections import defaultdict

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
from vescale.pipe.pipe_emmiter import ScheduleEngine, StageDeps

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



def _align_num_batches(first_stage_rank, batches):
    """
    Aligns all ranks must have the same number of mini-batches as rank 0.

    Let's walk through an example to illustrate the function `_align_num_batches` in a distributed training setup.

    ### Scenario:
    - Imagine we have **3 GPUs (ranks 0, 1, and 2)**, and we're performing some kind of parallel training.
    - Each GPU might initially have a different number of mini-batches for some reason, but we want to make sure they all have the same number of batches as **rank 0**.
    - We'll use `_align_num_batches` to ensure that all GPUs (ranks) are aligned to the same number of batches as rank 0.

    ### Example:
    - **Rank 0** (first_stage_rank): 64 batches
    - **Rank 1**: 60 batches
    - **Rank 2**: 62 batches

    We want all ranks to have 64 batches (same as rank 0).

    ### Code Simulation:
    Let's simulate how the function works with this example.

    #### Initial values:
    ```python
    # Assume we are at Rank 1 (which initially has 60 batches)
    first_stage_rank = 0
    batches = 60  # Rank 1's initial number of batches

    #### What happens on each rank:

    1. **Rank 0 (first_stage_rank):**
    - Input: `batches = 64`
    - This rank broadcasts its value of `64` to all other ranks.
    - No change in `batches`, as it is already 64.

    ```python
    num_batches = torch.tensor([64], dtype=torch.int64).cuda(0)
    dist.broadcast(num_batches, src=0)  # Broadcasts 64 to ranks 1 and 2
    # is_consistent = 64 == 64 (True), no change needed
    return 64
    ```

    2. **Rank 1:**
    - Input: `batches = 60`
    - It receives the broadcasted value of `64` from rank 0.
    - It compares `60` (local `batches`) with `64` (broadcasted value), and since
    they are different, it updates its `batches` to `64`.

    ```python
    num_batches = torch.tensor([60], dtype=torch.int64).cuda(1)  # Create local tensor
    dist.broadcast(num_batches, src=0)  # Receives 64 from rank 0

    # is_consistent = 60 == 64 (False), so update batches to 64
    batches = num_batches.item()  # Now batches = 64
    return 64
    ```

    3. **Rank 2:**
    - Input: `batches = 62`
    - It receives the broadcasted value of `64` from rank 0.
    - It compares `62` (local `batches`) with `64` (broadcasted value), and since
    they are different, it updates its `batches` to `64`.

    ```python
    num_batches = torch.tensor([62], dtype=torch.int64).cuda(2)  # Create local tensor
    dist.broadcast(num_batches, src=0)  # Receives 64 from rank 0

    # is_consistent = 62 == 64 (False), so update batches to 64
    batches = num_batches.item()  # Now batches = 64
    return 64
    ```

    ### Final Result:

    - **Rank 0**: 64 batches (unchanged).
    - **Rank 1**: Now has 64 batches (updated from 60).
    - **Rank 2**: Now has 64 batches (updated from 62).

    Now all ranks are aligned with the same number of batches, ensuring consistency
    across the distributed training process.

    ### Why is this important?

    In distributed training, especially in pipeline or model parallelism, if
    different GPUs (ranks) have different numbers of mini-batches, the synchronization
    between them might break. This can lead to crashes or inefficiencies. By aligning
    the number of mini-batches across all ranks, we ensure that the GPUs can proceed
    in lockstep, avoiding synchronization issues.
    """
    num_batches = torch.tensor([batches], dtype=torch.int64).cuda(dist.get_rank())
    dist.broadcast(num_batches, src=first_stage_rank)
    is_consistent = num_batches.item() == batches
    if not is_consistent:
        batches = num_batches.item()
    return batches






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
    debug_at_rank_n(0) # 有效


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
        # split_method=PipelineSplitMethodType.AUTO,
        # split_method=PipelineSplitMethodType.UNIFORM,
        num_stages=4,
        virtual_chunks=1,
        smallest_unsplittable_units=[f"transformer.h.{i}" for i in range(n_layer)],
        split_points=["transformer.h.3", "transformer.h.6", "transformer.h.9"],
        batch_p2p_comm=False,
        overlap_p2p_comm=True,
        # schedule_type=PipelineScheduleType.SIMPLE_1F1B,
        schedule_type=PipelineScheduleType.ZERO_BUBBLE,
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






    #--------------------------------------------------------------------------#

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

    x, y = x.to(device), y.to(device)

    # Convert the batch into an iterator or a list as needed
    data_iterator = [x, y]  # Adjust depending on how your engine expects input

    # data_iterator = [x, y]  # Data has already been converted to DTensor by get_batch()

    #--------------------------------------------------------------------------#













    ### deps 可以这样获得, 也可以在下面另一个函数 build_schedule 中获得
    # deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())


    #--------------------------------------------------------------------------#
    ### def build_schedule 来自于 class PipeEngine 内

    # def build_schedule(self, minibatches, data_shape=None):
    #     """
    #     Build pipeline parallel training schedules.
    #     """

    ### 这个 minibatches 的值是多少? 谁调用了 build_schedule 函数?
    minibatches = data_iterator



    ### meshes = self.global_mesh.get_global_tensor_parallel_meshes()
    meshes = VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes()

    ### dp_rank, tp_rank = self.global_mesh.get_data_parallel_rank(), self.global_mesh.get_tensor_parallel_rank()
    dp_rank, tp_rank = VESCALE_DEVICE_MESH.get_data_parallel_rank(), VESCALE_DEVICE_MESH.get_tensor_parallel_rank()

    tp_meshes_dict = defaultdict(list)

    def _locate_tp_mesh(_rank):
        for tp_mesh in meshes:
            if _rank in tp_mesh.mesh.tolist():
                return tp_mesh
        else:
            raise ValueError("TP submesh not found.")

    for _rank in range(torch.distributed.get_world_size()):
        ### _coordinate = self.global_mesh.get_strategy_coordinate(_rank)
        _coordinate = VESCALE_DEVICE_MESH.get_strategy_coordinate(_rank)
        tp_mesh = _locate_tp_mesh(_rank)
        _dp_rank, _tp_rank = _coordinate[1], _coordinate[2]
        tp_meshes_dict[(_dp_rank, _tp_rank)].append(tp_mesh)


    new_meshes = tp_meshes_dict[(dp_rank, tp_rank)]
    meshes = new_meshes

    ### first_stage_rank = self.global_mesh.get_strategy_coordinate(local_rank=0)[0]
    first_stage_rank = VESCALE_DEVICE_MESH.get_strategy_coordinate(local_rank=0)[0]

    # FIXME: the input can either be PipeModule, or a sequence of DDP modules? In the latter case, how to get stage dependency
    ### pipe_module = self.module

    stage_dep_matrix, p2p_index_mapping = pipe_module.stage_deps, pipe_module.p2p_index_mapping
    stage_dependency = StageDeps(
        dep=stage_dep_matrix,
        meshes=meshes,
        vpp_module_list=pipe_module,
        p2p_index_mapping=p2p_index_mapping,
    )





    num_minibatches = _align_num_batches(first_stage_rank, len(minibatches))

    # TODO: insert shape inference
    # batch_p2p_comm = self.engine_plan.batch_p2p_comm

    # if on interleaved 1f1b schedule, set batch_p2p_comm to False to execute p2p communication
    ### schedule_type = self.schedule_type
    schedule_type = PipelineScheduleType.ZERO_BUBBLE

    virtual_chunks_per_stage = pipe_plan.virtual_chunks

    if schedule_type in [PipelineScheduleType.INTERLEAVED_1F1B, PipelineScheduleType.ZERO_BUBBLE]:

        ### 这里的 virtual_chunks_per_stage 是多少, 当调用函数 build_schedule 时, 传入的参数是多少?
        ### 来自于 pipe_plan: PipelineParallelPlan

        data_iterator = [iter(minibatches) for _ in range(virtual_chunks_per_stage)]
        batch_p2p_comm = False
    elif schedule_type == PipelineScheduleType.SIMPLE_1F1B:
        data_iterator = minibatches
    else:
        raise NotImplementedError(f"Schedule {schedule_type} not implemented yet.")


    #--------------------------------------------------------------------------#



    ### 用的数据相对准, 我用的是 huggingface demo 的 real example
    ### https://huggingface.co/spaces/sail/zero-bubble-pipeline-parallellism/blob/main/v_schedule.py

    ### 这个 settings 用来当做 doc 参考, 不删
    # settings = [
    #     # p,   n,     f,     b,     w,   c,    h,  a,  l
    #     (8, 24, 18522, 18086, 9337, 601, 2304, 24, 24),
    #     (8, 32, 18513, 18086, 9331, 626, 2304, 24, 24),
    #     (8, 64, 18546, 18097, 9321, 762, 2304, 24, 24),
    #     (8, 24, 29718, 29444, 19927, 527, 4096, 32, 32),
    #     (8, 32, 29802, 29428, 19530, 577, 4096, 32, 32),
    #     (8, 64, 29935, 29621, 19388, 535, 4096, 32, 32),
    #     (16, 48, 11347, 11248, 8132, 377, 5120, 40, 48),
    #     (16, 64, 11307, 11254, 8101, 379, 5120, 40, 48),
    #     (16, 128, 11325, 11308, 8109, 378, 5120, 40, 48),
    #     (32, 96, 10419, 10207, 7715, 408, 6144, 48, 64),
    #     (32, 128, 10408, 10204, 7703, 408, 6144, 48, 64),
    #     (32, 256, 10402, 10248, 7698, 460, 6144, 48, 64),
    #     (4, 8, 6, 4, 4, 1, 4096, 32, 32),
    #     (8, 24, 29444, 29718, 19927, 527, 4096, 32, 32),
    #     ( 8, 32, 16099, 16504,  7589,  540, 2304, 24, 16),
    #     (16, 48, 14407, 14380,  9676, 1610, 4096, 32, 32),
    #     (16, 64, 14412, 14393,  9688, 1621, 4096, 32, 32),
    #     (16, 128,14316, 14306,  9639, 1619, 4096, 32, 32),
    #     (24, 72,  6763,  6969,  5251,  755, 5120, 40, 48),
    #     (24, 96,  6783,  6984,  5259,  758, 5120, 40, 48),
    #     (24, 192, 6785,  6990,  5260,  770, 5120, 40, 48),
    #     (32,  96, 9458,  9748,  7288,  879, 6144, 48, 64),
    #     (32, 128, 9469,  9744,  7306,  892, 6144, 48, 64),
    #     (32, 256, 9447,  9644,  7193,  887, 6144, 48, 64),
    # ]

    s = 1024
    p, n, f, b, w, c, h, a, l = (4, 8, 6, 4, 4, 1, 4096, 32, 32)
    mem_f = 34 * h + 5 * a * s
    mem_w = - 32 * h
    mem_b = - mem_w - mem_f


    # Pack costs and memory values into a dictionary
    costs_and_mem = {
        'f_cost': f,
        'b_cost': b,
        'w_cost': w,
        'c_cost': c,
        'f_mem': mem_f,
        'b_mem': mem_b,
        'w_mem': mem_w,
        'max_mem': mem_f * 4 * 2,
    }

    data_shape = None
    dtype = pipe_plan.p2p_tensor_dtype
    overlap_p2p_comm = pipe_plan.overlap_p2p_comm




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


    global_mesh = VESCALE_DEVICE_MESH

    forward_only = pipe_plan.forward_only

    pipe_engine = ScheduleEngine(
        deps=stage_dependency,
        meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
        schedule=PipelineScheduleType.ZERO_BUBBLE,
        batches=num_minibatches,
        data_iterator=data_iterator,
        stage_id=VESCALE_DEVICE_MESH.get_pipeline_parallel_rank(),
        shape=data_shape,
        dtype=dtype,
        num_chunks=virtual_chunks_per_stage,
        input_shapes=None,
        input_shapes_unpad=None,
        # send_dtypes_map=self.module.recv_dtypes_dict,
        overlap_p2p_comm=overlap_p2p_comm,
        batch_p2p_comm=batch_p2p_comm,
        loss_fn=loss_fn,
        global_mesh=global_mesh,
        forward_only=forward_only,
        **costs_and_mem,
    )


    ### this is the right way to do init
    # pipe_engine = ScheduleEngine(
    #     deps=deps,
    #     meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
    #     schedule=PipelineScheduleType.ZERO_BUBBLE,
    #     batches=batches,
    #     data_iterator=[iter(data_iterator) for _ in range(num_chunks)],
    #     stage_id=local_rank,
    #     shape=(1, 1, 3),
    #     dtype=torch.float32,
    #     f_cost=6,
    #     b_cost=4,
    #     w_cost=4,
    #     c_cost=1,
    #     f_mem=mem_f,
    #     b_mem=mem_b,
    #     w_mem=mem_w,
    #     max_mem=mem_f * 4 * 2,
    # )























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





    pipe_module.stage_modules[0].to(device)
    # print(pipe_module.stage_modules)

    engine = PipeEngine(
        pipe_module,
        VESCALE_DEVICE_MESH,
        loss_fn,
        pipe_plan,
    )

    engine.schedule_engine = pipe_engine






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



    ### data_iterator

    # Pass the data to the engine for processing
    minibatch_loss, outputs = engine(minibatch=data_iterator, reuse_schedule=True)


    # # Print loss and outputs
    # print(f"Minibatch Loss: {minibatch_loss}")
    # print(f"Outputs: {outputs}")



    ### Close the file when done
    sys.stdout.close()






if __name__ == "__main__":
    main()
