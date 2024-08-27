################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.testing._internal.common_utils import run_tests

from common_dtensor import DTensorTestBase, with_comms_device, with_comms

from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard


def initialize_distributed():
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # device = f"cuda:{rank}"
    # torch.cuda.set_device(device)

    ### Use the backend appropriate for your hardware
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    torch.distributed.init_process_group(backend=backend, world_size=world_size, rank=rank)

    # # Initialize the process group
    # dist.init_process_group(backend=backend)

    # # Set the device based on the rank
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(dist.get_rank())

    # Optionally, set the random seed for reproducibility
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(42)


def debug_at_rank_n(rank_id):
    """If distributed is initialized, print only on rank n."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank_id:
            message = f'debug at rank {torch.distributed.get_rank()}'
            # print(message, flush=True)
            print(f"\033[93m{message}\033[00m", flush=True)
            import debugpy
            debugpy.listen(5678)
            debugpy.wait_for_client()
            debugpy.breakpoint()
    else:
        message = 'You are not in distributed mode.'
        # print(message, flush=True)
        print(f"\033[93m{message}\033[00m", flush=True)


CONFIG = {"batch_size": 4, "seq_length": 4, "hidden_size": 4}


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 2)
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(config["hidden_size"] * 2, config["hidden_size"])

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize fc1 weights
        self.fc1.weight.data = torch.arange(self.fc1.weight.numel()).view(self.fc1.weight.size()).float()

        # Initialize fc2 weights
        self.fc2.weight.data = torch.arange(self.fc2.weight.numel()).view(self.fc2.weight.size()).float()

        # Initialize fc1 and fc2 biases if needed (same pattern)
        if self.fc1.bias is not None:
            self.fc1.bias.data = torch.arange(self.fc1.bias.numel()).float()
        if self.fc2.bias is not None:
            self.fc2.bias.data = torch.arange(self.fc2.bias.numel()).float()

    def forward(self, x):
        # print(f"x.shape: {x.shape}") # x.shape: [16, 4]

        ### y = x W^T + b, x shape: [16, 4], W shape: [8, 4], W^T shape: [4, 8], y shape: [16, 8]
        x = self.fc1(x) # x shape: [16, 8]
        # print(f"fc1 output shape: {x.shape}") # fc1 output shape: [16, 8]

        x = self.gelu(x) # input x shape: [16, 8]; output x shape: [16, 8]
        # print(f"gelu output shape: {x.shape}")

        ### W shape: [4, 8]
        ### y = x W^T + b, x shape: [16, 8], W shape: [4, 8], W^T shape: [8, 4], y shape: [16, 4]
        x = self.fc2(x) # input x shape: [16, 8]; output x shape: [16, 4]
        # print(f"fc2 output shape: {x.shape}")
        return x


param_sharding_plan1 = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Shard(0)],
    "fc2.weight": [Shard(1)],
    "fc2.bias": [Replicate()],
}

fwd_resharding_plan1 = {
    "fc1.input": [[Replicate()]],
    "fc2.output": [[Replicate()]],
}

param_sharding_plan2 = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Replicate()],
    "fc2.weight": [Shard(0)],
    "fc2.bias": [Replicate()],
}

fwd_resharding_plan2 = {
    "fc1.input": [[Replicate()]],
    "fc1.weight": [Replicate()],
    "fc2.weight": [Replicate()],
    "fc2.output": [[Replicate()]],
}


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config["hidden_size"], bias=False)
        self.mlp = MLP(config)

    def forward(self, x):
        return self.mlp(self.ln(x))


class DModuleTestPlans(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _run_plan(self, param_sharding_plan, fwd_resharding_plan, devce_type):
        device_mesh = DeviceMesh(devce_type, list(range(self.world_size)))
        print(f"device_mesh: {device_mesh}")

        ### Initialize distributed training

        debug_at_rank_n(0)

        print(f"Running test with param_sharding_plan: {param_sharding_plan}, fwd_resharding_plan: {fwd_resharding_plan} on devce_type: {devce_type}")

        """
        Running test with

        param_sharding_plan:
        {
            'fc1.weight': [Shard(dim=0)],
            'fc1.bias': [Replicate()],
            'fc2.weight': [Shard(dim=0)],
            'fc2.bias': [Replicate()]
        },

        fwd_resharding_plan:
        {
            'fc1.input': [[Replicate()]],
            'fc1.weight': [Replicate()],
            'fc2.weight': [Replicate()],
            'fc2.output': [[Replicate()]]
        }
        on devce_type: cpu
        """

        """
        Given the configuration:

        - **DeviceMesh**: `[0, 1, 2, 3]`, which represents four GPUs.
        - **Placement**: `[Shard(dim=0)]`, indicating that the tensor will be sharded along dimension 0 (rows).

        ### Explanation of Sharding

        The tensor for `fc1.weight` is an 8x4 matrix:
        ```
        tensor([[ 0.,  1.,  2.,  3.],
                [ 4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11.],
                [12., 13., 14., 15.],
                [16., 17., 18., 19.],
                [20., 21., 22., 23.],
                [24., 25., 26., 27.],
                [28., 29., 30., 31.]])
        ```

        With `Shard(dim=0)`, the tensor's rows (dimension 0) will be divided across the four GPUs in the device mesh.
        Since there are 8 rows and 4 GPUs, each GPU will receive an equal share of the rows.

        ### Distribution of Rows Across GPUs

        - **GPU 0** will receive the first 2 rows:
        ```
        [[ 0.,  1.,  2.,  3.],
        [ 4.,  5.,  6.,  7.]]
        ```

        - **GPU 1** will receive the next 2 rows:
        ```
        [[ 8.,  9., 10., 11.],
        [12., 13., 14., 15.]]
        ```

        - **GPU 2** will receive the next 2 rows:
        ```
        [[16., 17., 18., 19.],
        [20., 21., 22., 23.]]
        ```

        - **GPU 3** will receive the last 2 rows:
        ```
        [[24., 25., 26., 27.],
        [28., 29., 30., 31.]]
        ```

        ### Summary of Sharding

        The `fc1.weight` tensor is sharded along its rows across the four GPUs in the DeviceMesh. Each GPU handles a distinct subset of rows:

        - **GPU 0**: Rows 0-1
        - **GPU 1**: Rows 2-3
        - **GPU 2**: Rows 4-5
        - **GPU 3**: Rows 6-7

        This sharding strategy allows each GPU to operate on a portion of the tensor, enabling parallel processing during training or inference.
        """

        """
        fc2 weights: Parameter containing:
        tensor([[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.],
                [ 8.,  9., 10., 11., 12., 13., 14., 15.],
                [16., 17., 18., 19., 20., 21., 22., 23.],
                [24., 25., 26., 27., 28., 29., 30., 31.]], requires_grad=True)

        Given the configuration:
        GPU 0: Rows 0, [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.]
        GPU 1: Rows 1, [ 8.,  9., 10., 11., 12., 13., 14., 15.]
        GPU 2: Rows 2, [16., 17., 18., 19., 20., 21., 22., 23.]
        GPU 3: Rows 3, [24., 25., 26., 27., 28., 29., 30., 31.]
        """

        # create golden model (local replicate)
        mlp_golden = MLP(CONFIG)

        def print_model_weights_shapes(model):
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Shape: {param.shape}")

        print_model_weights_shapes(mlp_golden)

        ### Print the initialized weights
        print(f"fc1 weights: {mlp_golden.fc1.weight}")
        print(f"fc2 weights: {mlp_golden.fc2.weight}")

        mlp_golden.to(devce_type)
        for name, param in mlp_golden.named_parameters():
            dist.all_reduce(param, async_op=False)

        # create dmodule (by plans)
        dmlp = MLP(CONFIG)
        dmlp.to(devce_type)
        dmlp.load_state_dict(mlp_golden.state_dict())

        parallelize_module(dmlp, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})

        # create data (local replicate)
        print(f"CONFIG: {CONFIG}")
        ### CONFIG: {'batch_size': 4, 'seq_length': 4, 'hidden_size': 4}
        input_golden = torch.randn(
            CONFIG["batch_size"] * CONFIG["seq_length"], CONFIG["hidden_size"], device=devce_type, requires_grad=False
        )

        dist.all_reduce(input_golden, async_op=False)
        input_tensor = input_golden.detach().clone()
        print(f"input_tensor.shape: {input_tensor.shape}")

        # match forward
        output_tensor = dmlp(input_tensor).to_local()
        output_golden = mlp_golden(input_golden)
        self.assertTrue(torch.allclose(output_tensor, output_golden, rtol=1e-4, atol=1e-5))

        # match backward
        output_tensor.sum().backward()
        output_golden.sum().backward()
        for n, p in dmlp.named_parameters():
            if n.endswith("bias") and any(place.is_partial() for place in p.placements):
                continue  # vescalized linear
            self.assertTrue(p.grad is not None)
            with torch.no_grad():
                grad_dmlp = p.grad.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()
            grad_golden = mlp_golden.get_parameter(n).grad
            self.assertTrue(torch.allclose(grad_dmlp, grad_golden, rtol=1e-4, atol=1e-5))


    @with_comms_device(device_type="cpu")
    def test_cpu(self):
        torch.manual_seed(42)
        self._run_plan(param_sharding_plan1, fwd_resharding_plan1, "cpu")
        self._run_plan(param_sharding_plan2, fwd_resharding_plan2, "cpu")


    @with_comms_device(device_type="cuda")
    def test_cuda(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._run_plan(param_sharding_plan1, fwd_resharding_plan1, "cuda")
        # self._run_plan(param_sharding_plan2, fwd_resharding_plan2, "cuda")


    @with_comms_device(device_type="cuda")
    def test_wrong_plan(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        # create dmodule (by plans)
        mlp = MLP(CONFIG)
        with self.assertRaises(KeyError):
            parallelize_module(mlp, device_mesh, {"parameters": param_sharding_plan1, "forward": fwd_resharding_plan1})
        with self.assertRaises(KeyError):
            parallelize_module(mlp, device_mesh, {"parameter": param_sharding_plan1, "forwards": fwd_resharding_plan1})


    @with_comms
    def test_tp_plan(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        sharding_plan = {
            "parameter": {
                "mlp.fc1.weight": [Shard(0)],
                "mlp.fc1.bias": [Shard(0)],
                "mlp.fc2.weight": [Shard(1)],
                "mlp.fc2.bias": [Replicate()],
            },
            "forward": {
                "input": [[Replicate()]],
                "ln.input": [[Replicate()]],  # no SP
                "mlp.input": [[Replicate()]],
                "mlp.fc2.output": [[Replicate()]],
            },
        }

        dmodel = parallelize_module(Block(CONFIG), device_mesh, sharding_plan)
        input = torch.ones((CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["hidden_size"]), requires_grad=True)
        output = dmodel(input).to_local()
        output.sum().backward()


    @with_comms
    def test_tp_sp_plan(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        sharding_plan = {
            "parameter": {
                "mlp.fc1.weight": [Shard(0)],
                "mlp.fc1.bias": [Shard(0)],
                "mlp.fc2.weight": [Shard(1)],
                "mlp.fc2.bias": [Replicate()],
            },
            "forward": {
                "input": [[Replicate()]],
                "ln.input": [[Shard(1)]],  # SP
                "mlp.input": [[Replicate()]],
                "mlp.fc2.output": [[Replicate()]],
            },
        }

        dmodel = parallelize_module(Block(CONFIG), device_mesh, sharding_plan)
        input = torch.ones((CONFIG["batch_size"], CONFIG["seq_length"], CONFIG["hidden_size"]), requires_grad=True)
        output = dmodel(input).to_local()
        output.sum().backward()


if __name__ == "__main__":
    run_tests()
