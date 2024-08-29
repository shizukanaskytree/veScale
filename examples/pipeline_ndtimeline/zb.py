import os
import datetime
import sys

import torch
import torch.nn as nn  # Add this import
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.pipe.pipe_emmiter import ScheduleEngine
from vescale.dtensor.placement_types import Replicate
from vescale.pipe._schedules.instruction_base import get_linear_pp_module_dep2
from vescale.plan.spec import PipelineScheduleType
from vescale.devicemesh_api import VESCALE_DEVICE_MESH

class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features * 2, bias=False)
        torch.nn.init.uniform_(self.fc1.weight, 0, 1)
        self.fc2 = nn.Linear(n_features * 2, n_features)
        torch.nn.init.uniform_(self.fc2.weight, 0, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class FourMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden)
        self.mlp2 = MLP(hidden)
        self.mlp3 = MLP(hidden)
        self.mlp4 = MLP(hidden)

    def forward(self, x):
        return self.mlp4(self.mlp3(self.mlp2(self.mlp1(x))))


class EightMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlps = nn.ModuleList([MLP(hidden) for _ in range(8)])

    def forward(self, x):
        all_input_x = []
        for idx, mlp in enumerate(self.mlps):
            x = mlp(x)
            x.retain_grad()
            all_input_x.append(x)
            print(f"mlp: {idx} output : {x}")
        return x, all_input_x


def init_process_group():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= world_size else "gloo"

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        init_method="env://",
        timeout=datetime.timedelta(seconds=1200),
    )

    if backend == "nccl":
        torch.cuda.set_device(rank)


def destroy_process_group():
    dist.barrier()
    dist.destroy_process_group()


def test_zerobubble_engine():
    """
    Tests zero-bubble pipeline schedule with profiling.
    """
    init_process_group()

    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )

        local_rank = rank
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)

        from vescale.ndtimeline import init_ndtimers, flush, wait

        init_ndtimers(rank=int(local_rank), local_rank=int(local_rank), enable_streamer=True)
        num_chunks = 2
        n_hidden = 3
        batches = 8
        model = EightMLP(n_hidden)
        for i in range(8):
            model.mlps[i] = model.mlps[i].cuda()

        torch.distributed.barrier()

        all_batches_out = []
        if rank == 0:
            true_model = model
            for i in range(8):
                true_model.mlps[i] = true_model.mlps[i].cuda(0)
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(0)
                out, all_output_x = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                for idx, output in enumerate(all_output_x):
                    print(f"mlp{idx}.grad is {output.grad}")
                print(" ====================================== ")

        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        model_list = []

        if rank == 0:
            model_list = [model.mlps[0], model.mlps[7]]
        elif rank == 1:
            model_list = [model.mlps[1], model.mlps[6]]
        elif rank == 2:
            model_list = [model.mlps[2], model.mlps[5]]
        elif rank == 3:
            model_list = [model.mlps[3], model.mlps[4]]

        deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(data.float().cuda())

        w = n_hidden * 2 * 4
        a = n_hidden * 4
        mem_f = 2 * w + 2 * a
        mem_w = -2 * a
        mem_b = -mem_w - mem_f

        pipe_engine = ScheduleEngine(
            deps=deps,
            meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
            schedule=PipelineScheduleType.ZERO_BUBBLE,
            batches=batches,
            data_iterator=[iter(data_iterator) for _ in range(num_chunks)],
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

        _, all_forward = ScheduleEngine.execute(pipe_engine)
        if rank == 0:
            loss_per_microbatch = [item[1] for item in all_forward]
            print(loss_per_microbatch, all_batches_out)
            for t1, t2 in zip(loss_per_microbatch, all_batches_out):
                assert t1 == t2

        torch.distributed.barrier()

        flush()
        wait()
    finally:
        destroy_process_group()

# Example usage:
if __name__ == "__main__":
    test_zerobubble_engine()
