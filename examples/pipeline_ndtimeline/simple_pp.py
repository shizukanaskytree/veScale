# test/parallel/pipeline/e2e/test_pp_accuracy_alignment.py

import os
from typing import Any, Callable, Dict, Generator, Iterator, List, Sequence, Tuple, TypeVar, cast
from functools import wraps

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
import torch.distributed as dist

from vescale.plan import (
    PipelineParallelPlan,
    PipelineScheduleType,
    ModeType,
    PipelineSplitMethodType,
)
from vescale.pipe import PipeModule, construct_stage_modules
from vescale.engine import PipeEngine
# from common_dtensor import DTensorTestBase, with_comms
from vescale.devicemesh_api import VESCALE_DEVICE_MESH

from vescale.ndtimeline import init_ndtimers, flush, wait

microbatch_size = 2
factor = 32
batch_size = microbatch_size * factor
stage = 4
RANDOM_SEED = 9999


class MLP(nn.Module):
    def __init__(self, features_in, feature_middle, features_out, value, idx=1):
        super().__init__()
        self.value = value
        self.idx = idx
        self.counter = 0
        self.fc1 = nn.Linear(features_in, feature_middle, bias=False)
        self.fc2 = nn.Linear(feature_middle, features_out, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        # torch.save(t, f"{os.environ['model_name']}_mlp{self.value}_fwd{self.counter}_out_tensor.pt")
        # self.counter += 1
        return t


class EightMLP(nn.Module):
    def __init__(self, hidden=1024, fixed_size=True):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, hidden, 1, 1)
        self.mlp2 = MLP(hidden, hidden, hidden, 2, 2)
        self.mlp3 = MLP(hidden, hidden, hidden, 1, 3)
        self.mlp4 = MLP(hidden, hidden, hidden, 2, 4)
        self.mlp5 = MLP(hidden, hidden, hidden, 1, 5)
        self.mlp6 = MLP(hidden, hidden, hidden, 2, 6)
        self.mlp7 = MLP(hidden, hidden, hidden, 1, 7)
        self.mlp8 = MLP(hidden, hidden, hidden, 2, 8)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        return x


def loss_fn(x):
    return x.mean()


def _run_engine_with_1f1b(fixed_size=True):
    os.environ["model_name"] = "pp"

    ### from torchrun, torchrun will automatically assign RANK for each process.
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE')}, RANK: {os.environ.get('RANK')}")

    # device = f"cuda:{self.rank}"
    device = f"cuda:{rank}"
    # must do this: https://pytorch.org/docs/stable/distributed.html
    torch.cuda.set_device(device)

    model = EightMLP(16, fixed_size=fixed_size).cuda()
    # model.load_state_dict(torch.load("baseline_model.pt"))

    pipe_config = PipelineParallelPlan(
        mode=ModeType.GRAPH_EAGER,
        split_method=PipelineSplitMethodType.MANUAL,
        num_stages=4,
        virtual_chunks=2,
        smallest_unsplittable_units=["mlp1", "mlp2", "mlp3", "mlp4", "mlp5", "mlp6", "mlp7", "mlp8"],
        split_points=["mlp2", "mlp4", "mlp6", "mlp8"],
        batch_p2p_comm=False,
        overlap_p2p_comm=True,
        schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
    )

    optimizer_fn_kwargs = {
        "lr": 0.01,
        "momentum": 0,
        "dampening": 0,
        "weight_decay": 0,
        "nesterov": False,
        "maximize": False,
        "foreach": None,
        "differentiable": False,
    }

    torch.manual_seed(9999)

    ### input data
    with torch.no_grad():
        batch = [torch.ones(microbatch_size, 128, 16, dtype=torch.float32).to(device) for _ in range(factor)]

    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type="cuda",
        mesh_shape=(4, 1, 1),
        mesh_dim_names=["PP", "DP", "TP"],
    )

    ### xxx profiling below xxx
    ### https://github.com/volcengine/veScale/blob/main/vescale/ndtimeline/README.md#how-to-use-ndtimeline
    init_ndtimers(
        rank=int(rank),
        local_rank=int(rank),
        enable_streamer=True,
    )
    ### xxx profiling above xxx

    stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
        model,
        pipe_config,
        VESCALE_DEVICE_MESH,
        update_split_points=True,
    )

    print(f"len of stage_modules: {len(stage_modules)}")

    _parameters = list(stage_modules[0].parameters()) + list(stage_modules[1].parameters())

    optimizer = torch.optim.SGD(_parameters, **optimizer_fn_kwargs)

    pipe_module = PipeModule(
                    stage_modules,
                    optimizer,
                    None,
                    stage_dependency,
                    p2p_index_mapping,
                    pipe_config)

    engine = PipeEngine(
        pipe_module,
        VESCALE_DEVICE_MESH,
        loss_fn, # loss function
        pipe_config,
    )

    minibatch_loss, minibatch_outputs = engine(batch)

    print(f"mini batch loss: {minibatch_loss}")
    # print(f"mini batch outputs: {minibatch_outputs}")

    ### Stage 0 -> 1 -> 2 -> 3 -> loss
    if rank == 3:
        minibatch_loss.backward()

    optimizer = engine.get_optimizer
    optimizer.step()


    # if self.rank == 0:
    #     self.save_mlp_parameter(engine.module[0].get_submodule("mlp1"), "engine_1f1b_mlp1")
    #     self.save_mlp_parameter(engine.module[1].get_submodule("mlp5"), "engine_1f1b_mlp5")
    # if self.rank == 1:
    #     self.save_mlp_parameter(engine.module[0].get_submodule("mlp2"), "engine_1f1b_mlp2")
    #     self.save_mlp_parameter(engine.module[1].get_submodule("mlp6"), "engine_1f1b_mlp6")
    # if self.rank == 2:
    #     self.save_mlp_parameter(engine.module[0].get_submodule("mlp3"), "engine_1f1b_mlp3")
    #     self.save_mlp_parameter(engine.module[1].get_submodule("mlp7"), "engine_1f1b_mlp7")
    # if self.rank == 3:
    #     self.save_mlp_parameter(engine.module[0].get_submodule("mlp4"), "engine_1f1b_mlp4")
    #     self.save_mlp_parameter(engine.module[1].get_submodule("mlp8"), "engine_1f1b_mlp8")


    dist.barrier()

    ### ndtimeline profiling done.
    ### Verify that there are no other operations or code paths that might cause the process group to be unintentionally
    ### destroyed or reset after initialization. This could happen if, for example, there is a call to destroy_process_group()
    ### somewhere in your code before the barrier().
    flush()
    wait()



def main():
    _run_engine_with_1f1b()


if __name__ == "__main__":
    main()
