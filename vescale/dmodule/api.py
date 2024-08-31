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
# some code is inspired by torch/distributed/tensor/parallel/api.py
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
################################################################################

from typing import Dict, Optional, Union, Any
import warnings

from torch import nn
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dmodule._dmodule import DModule
from vescale.dmodule.placements_interface import PlacementsInterface
from vescale.debug import DebugLogger

__all__ = ["parallelize_module", "is_dmodule", "PlacementsInterface"]


def parallelize_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    sharding_plan: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    is_model_sharded: bool = False,
    factory: Union[bool, Dict[nn.Module, Union[bool, Dict]]] = False,
) -> nn.Module:
    r"""
    Parallelize this `nn.Module` instance by inplace converting its parameters/buffers/activations from Tensor to DTensor:
        1. onto target `device_mesh`
        2. with target `sharding_plan`

    Args:
        device_mesh: the device mesh used in this entire DModule and its submodules

        sharding_plan:
            'parameter': the plan to specify which and how weights are sharded on device mesh during initalization.

                        Format: `{ <fully qualified name of weight/bias> : <sharding placements> }`
                            - <fully qualified name> is torch-native (see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)
                            - <fully qualified name> uses regex match in form of `<regex pattern for submodule>.<weight/bias>`
                            - <sharding placements> can be either:
                                - `None` for no op
                                - `Sequence[Placement]` for sharding spec (see `Placement` in `/vescale/dtensor/README.md`)
                                - `PlacementsInterface(Sequence[Placement], <optional flags>)` for sharding spec with DTensor flags

                        Note: Non-specified parameters in module will be converted to DTensor in `Replicate`, i.e., the "default" param plan.

                        Example:
                        >>> # find submodule "fc1"'s "weight" and convert it to DTensor in `tensor.dim=1` sharded on `device_mesh.dim=0`
                        >>> # and convert the rest parameters to DTensor in `Replicate()`
                        >>> param_plan = { "fc1.weight" : [Shard(1)] }

            'forward': the plan to specify which and how submodules' input/weight/output are resharded on device mesh during forward pass.

                        Format: `{ <fully qualified name of input/weight/output> : <resharding placements> }`
                            - <fully qualified name> is same as above
                            - <fully qualified name of input/weight/output> uses regex match in form of `<regex pattern for submodule>.<input/weight/output>`
                            - <resharding placements> can be defined in two forms:

                            1. List form: Just list all desired <sharding placements> (same as above), in a list.

                                The order of placement should follow the order the the arguments are defined.
                                And, for `*args`, the placements should be defined in the order of input.

                                Example:
                                >>> def foo(a, b, *args, c=1.0, **kwargs): pass
                                >>> fwd_plan = {
                                        "input":[
                                            [Replicate()], # placement for a
                                            [Shard(0)],  # placement for b
                                            [Shard(1)],  # placement for args[0]
                                            None,        # no op for args[1]
                                            None,       # no op for c
                                            [Partial]  # placement for the first key in kwargs.
                                        ]
                                    }
                                >>> foo(
                                        tensor0,  # will be Replicate
                                        tensor1,  # will be Shard(0)
                                        tensor2,  # will be Shard(1)
                                        tensor3,  # will be torch.Tenosr
                                        d = tensor4 # will be Partial
                                    )


                            2. Dictionary form: Use the arg name as the key, and the <sharding placements> as the value.
                                There is a special case where the key is `*args` and then the value is a list of placements.

                                Example:
                                >>> def foo(a, b, *args, c=1.0, **kwargs): pass
                                >>> fwd_plan={
                                        "input":{
                                            "a": [Replicate()], # placement for a
                                            "b": [Shard(0)],  # placement for b
                                            "args":[[Shard(1)], None],  # list of placements for args
                                            "c": None,       # placement for c
                                            "d": [Partial]  # placement for the first key in kwargs (called as d)
                                        }
                                    }
                                >>> foo(
                                        tensor0,  # will be Replicate
                                        tensor1,  # will be Shard(0)
                                        tensor2,  # will be Shard(1)
                                        tensor3,  # will be torch.Tenosr
                                        d = tensor4 # will be Partial
                                    )

        is_model_sharded (Optional): is this model (parameters/buffers) already sharded?

                                    Format:
                                        - `False` (Default): each rank holds a full model
                                        - `True`: each rank holds a only shard

                                    Note: this arg will be used for initalization internally.

        factory (Optional): whether to capture factory function (`torch.zeros`/`ones`/`empty`/`randn`/`full`/`arrange`) and convert it to DTensor during forward pass.
                This is used for resolving mixed Tensor and DTensor compute in forward, as bad practice can initailize the torch.Tensor buffer
                within `forward()` instead of within `Module.__init__()`. If this bad practice does happen, we can use this arg as a solver,
                at the cost of extra dispatching overhead.

                Format: `True` or `False` or `{ submodule_cls : { factory_func : <sharding placements> } }`
                    - `True`: all submodules and all factory funcs will be converted to DTensor in `Replicate`.
                    - `False` or `{}`: disable this factory function conversion to DTensor.
                    - `{ submodule_cls : True }`: only this `submodule_cls`'s all factory function will be converted to DTensor in `Replicate`.
                    - `{ submodule_cls : False or {} }`: exclude this `submodule_cls` for factory function conversion to DTensor.
                    - `{ submodule_cls : { factory_func : <sharding placements> } }`: only this `submodule_cls`'s `factory_func` will be converted to DTensor in `<sharding placements>`.

                Nested Case: `{ submodule_cls_outer : True/False/{..}, submodule_cls_inner : True/False/{..} }` can have `submodule_cls_inner` nested in `submodule_cls_outer`,
                            in which case we let the inner `submodule_cls_inner` overrides `submodule_cls_outer` in `True/False/{..}`, i.e., like a context manager in Python.

                Note: Currently, this factory converison:
                    - only covers `forward()`
                    - assumes same <sharding placements> for `factory_func`
                    - won't be affected by other TorchDispatchMode

    Returns:
        (Optional) this parallelized model instance


    Example:: using `plans`

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 4)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        with torch.device("cpu"):
            mlp = MLP()

        device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

        sharding_plan = {
            "parameter" : {
                "fc1.weight": [Shard(0)],
                "fc1.bias": [Shard(0)],
                "fc2.weight": [Shard(1)],
                "fc2.bias": [Replicate()],
            },
            "forward" : {
                "fc1.input": [[Replicate()]],
                "fc2.output": [[Replicate()]],
            }
        }

        dmlp = parallelize_module(mlp, device_mesh, sharding_plan)
        output = dmlp(input)


    Example:: using `is_model_sharded`

        ...
        with torch.device("cpu"):
            mlp_shard = MLPShardedPerRank()
        ...
        dmlp = parallelize_module(mlp_shard, ..., is_model_sharded = True)


    Example:: using deferred initialization

        ...
        fake_model = deferred_init(MLP)
        ...
        dmlp = parallelize_module(fake_model, device_mesh, sharding_plan)


    Example:: using factory for converting tensor buffer in forward

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

            def forward(self, x):
                x = torch.zeros(x.shape) # to be converted to DTensor zeros during runtime
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        dmlp = parallelize_module(MLP(), ..., factory=True) # or factory = { MLP: {torch.zeros: [Replicate()]} }

    Example:: using factory for nested classes

        class MLP(nn.Module):
            ...

            def forward(self, x):
                x = torch.zeros(x.shape) # to be converted to DTensor in Shard
                ...

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

            def forward(self, x):
                x = torch.zeros(x.shape) # to be converted to DTensor in Replicate
                x = self.mlp(x)
                return x

        dmlp = parallelize_module(MLP(), ..., factory={ Block : {torch.zeros: [Replicate()]}
                                                        MLP: {torch.zeros: [Shard(0)]} }) # inner class overrides

    Example:: using gradient synchronization with customized target

        ...
        dmlp = parallelize_module(model, ...})
        dmlp.finish_grad_sync()
        optimizer.step()

    """

    """
    parallelize_module 是一个函数，用于将 PyTorch 的 nn.Module 实例并行化。
        它通过将参数、缓冲区和激活从 Tensor 转换为 DTensor 来实现这一点。
        该函数支持设备网格（DeviceMesh）和分片计划（sharding_plan），并提供了一些选项来控制模型是否已经分片以及是否捕获工厂函数。

    作用
        将模型的参数、缓冲区和激活从 Tensor 转换为 DTensor。
        支持设备网格和分片计划。
        提供选项来处理已经分片的模型和工厂函数。
    """

    """
    ### 在这个 sharding_plan 中，定义了如何对模型的参数和前向传播的输入输出进行分片和复制。
    ### 具体来说：
    ### 参数分片
    ###     "fc1.weight": [Shard(0)]：表示 fc1 层的权重在第0维度上进行分片。
    ###     "fc1.bias": [Shard(0)]：表示 fc1 层的偏置在第0维度上进行分片。
    ###     "fc2.weight": [Shard(1)]：表示 fc2 层的权重在第1维度上进行分片。
    ###     "fc2.bias": [Replicate()]：表示 fc2 层的偏置在所有设备上进行复制。
    ### 前向传播
    ###     "fc1.input": [[Replicate()]]：表示 fc1 层的输入在所有设备上进行复制。
    ###     "fc2.output": [[Replicate()]]：表示 fc2 层的输出在所有设备上进行复制。
    ### 为什么这样做？
    ### 分片（Shard）：
    ###     分片可以减少每个设备上的内存占用，因为每个设备只存储一部分参数。
    ###     通过在不同维度上分片，可以更好地利用设备的并行计算能力。
    ### 复制（Replicate）：
    ###     复制可以确保所有设备都能访问相同的数据，适用于需要在多个设备上共享的参数或中间结果。
    ###     对于偏置（如 fc2.bias），复制可以简化计算，因为偏置通常较小，复制开销不大。
    sharding_plan = {
        "parameter" : {
            "fc1.weight": [Shard(0)],
            "fc1.bias": [Shard(0)],
            "fc2.weight": [Shard(1)],
            "fc2.bias": [Replicate()],
        },
        "forward" : {
            "fc1.input": [[Replicate()]],
            "fc2.output": [[Replicate()]],
        }
    }
    """

    """
    将这个 `nn.Module` 实例并行化，通过就地转换其参数/缓冲区/激活函数从 Tensor 到 DTensor：
    1. 在目标 `device_mesh` 上
    2. 使用目标 `sharding_plan`

    参数：
    - `device_mesh`：在整个 DModule 及其子模块中使用的设备网格。

    - `sharding_plan`：
        - `parameter`：指定在初始化期间哪些权重以及如何在设备网格上分片的计划。

            格式：`{ <权重/偏差的完全限定名称> : <分片布局> }`
                - `<完全限定名称, i.e., fully qualified name>` 是 PyTorch 原生的 (参见 https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)
                - `<完全限定名称>` 使用正则表达式匹配形式 `<子模块的正则表达式模式>.<权重/偏差>`
                - `<分片布局>` 可以是以下之一：
                    - `None` 表示无操作
                    - `Sequence[Placement]` 表示分片规范 (参见 `/vescale/dtensor/README.md` 中的 `Placement`)
                    - `PlacementsInterface(Sequence[Placement], <可选标志>)` 表示带有 DTensor 标志的分片规范

            注意：模块中未指定的参数将转换为 DTensor 并采用 `Replicate`，即“默认”参数计划。

            示例：
            ```python
            # 找到子模块 "fc1" 的 "weight" 并将其转换为 `tensor.dim=1` 的 DTensor，在 `device_mesh.dim=0` 上分片
            # 将其余参数转换为 `Replicate()` 的 DTensor
            param_plan = { "fc1.weight" : [Shard(1)] }
            ```

        - `forward`：指定在前向传播过程中哪些子模块的输入/权重/输出在设备网格上重分片的计划。

            格式：`{ <输入/权重/输出的完全限定名称> : <重分片布局> }`
                - `<完全限定名称>` 与上述相同
                - `<输入/权重/输出的完全限定名称>` 使用正则表达式匹配形式 `<子模块的正则表达式模式>.<输入/权重/输出>`
                - `<重分片布局>` 可以以两种形式定义：

                1. 列表形式：只需将所有所需的 `<分片布局>`（与上述相同）列出，放在一个列表中。

                    布局顺序应遵循参数定义的顺序。
                    对于 `*args`，布局应按输入顺序定义。

                    示例：
                    ```python
                    def foo(a, b, *args, c=1.0, **kwargs): pass
                    fwd_plan = {
                        "input": [
                            [Replicate()], # a 的布局
                            [Shard(0)],  # b 的布局
                            [Shard(1)],  # args[0] 的布局
                            None,        # args[1] 无操作
                            None,       # c 无操作
                            [Partial]  # kwargs 第一个键的布局
                        ]
                    }
                    foo(
                        tensor0,  # 将被设置为 Replicate
                        tensor1,  # 将被设置为 Shard(0)
                        tensor2,  # 将被设置为 Shard(1)
                        tensor3,  # 将保持为 torch.Tensor
                        d = tensor4 # 将被设置为 Partial
                    )
                    ```

                2. 字典形式：使用参数名作为键，<分片布局>作为值。
                    有一种特殊情况是键为 `*args`，值为布局列表。

                    示例：
                    ```python
                    def foo(a, b, *args, c=1.0, **kwargs): pass
                    fwd_plan = {
                        "input": {
                            "a": [Replicate()], # a 的布局
                            "b": [Shard(0)],  # b 的布局
                            "args":[[Shard(1)], None],  # args 的布局列表
                            "c": None,       # c 的布局
                            "d": [Partial]  # kwargs 第一个键（称为 d）的布局
                        }
                    }
                    foo(
                        tensor0,  # 将被设置为 Replicate
                        tensor1,  # 将被设置为 Shard(0)
                        tensor2,  # 将被设置为 Shard(1)
                        tensor3,  # 将保持为 torch.Tensor
                        d = tensor4 # 将被设置为 Partial
                    )
                    ```

    - `is_model_sharded` (可选)：模型（参数/缓冲区）是否已经分片？

        格式：
        - `False` (默认)：每个进程保存完整模型
        - `True`：每个进程只保存分片

        注意：此参数将用于初始化的内部处理。

    - `factory` (可选)：是否在前向传播期间捕获工厂函数（如 `torch.zeros`/`ones`/`empty`/`randn`/`full`/`arange`）并将其转换为 DTensor。
        这用于解决前向传播中混合 Tensor 和 DTensor 计算的问题，因为错误的做法可能会在 `forward()` 中初始化 `torch.Tensor` 缓冲区而不是在 `Module.__init__()` 中。如果确实发生这种错误做法，可以使用此参数作为解决方案，但会产生额外的调度开销。

        格式：`True` 或 `False` 或 `{ submodule_cls : { factory_func : <分片布局> } }`
            - `True`：所有子模块和所有工厂函数都将转换为 `Replicate` 的 DTensor。
            - `False` 或 `{}`：禁用此工厂函数转换为 DTensor。
            - `{ submodule_cls : True }`：仅此 `submodule_cls` 的所有工厂函数将转换为 `Replicate` 的 DTensor。
            - `{ submodule_cls : False 或 {} }`：排除此 `submodule_cls` 的工厂函数转换为 DTensor。
            - `{ submodule_cls : { factory_func : <分片布局> } }`：仅此 `submodule_cls` 的 `factory_func` 将转换为 `<分片布局>` 的 DTensor。

        嵌套情况：`{ submodule_cls_outer : True/False/{..}, submodule_cls_inner : True/False/{..} }` 可以在 `submodule_cls_outer` 中嵌套 `submodule_cls_inner`，
                在这种情况下，我们让内层的 `submodule_cls_inner` 覆盖 `submodule_cls_outer` 中的 True/False/{..}，即类似于 Python 中的上下文管理器。

        注意：目前，工厂函数转换仅覆盖 `forward()`，假定 `factory_func` 具有相同的 `<分片布局>`，且不会受到其他 TorchDispatchMode 的影响。

    返回：
    (可选) 并行化后的模型实例。

    示例:: 使用 `plans`

    ```python
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    with torch.device("cpu"):
        mlp = MLP()

    device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

    sharding_plan = {
        "parameter" : {
            "fc1.weight": [Shard(0)],
            "fc1.bias": [Shard(0)],
            "fc2.weight": [Shard(1)],
            "fc2.bias": [Replicate()],
        },
        "forward" : {
            "fc1.input": [[Replicate()]],
            "fc2.output": [[Replicate()]],
        }
    }

    dmlp = parallelize_module(mlp, device_mesh, sharding_plan)
    output = dmlp(input)
    ```

    示例:: 使用 `is_model_sharded`

    ```python
    ...
    with torch.device("cpu"):
        mlp_shard = MLPShardedPerRank()
    ...
    dmlp = parallelize_module(mlp_shard, ..., is_model_sharded=True)
    ```

    示例:: 使用延迟初始化

    ```python
    ...
    fake_model = deferred_init(MLP)
    ...
    dmlp = parallelize_module(fake_model, device_mesh, sharding_plan)
    ```

    示例:: 在前向传播中使用工厂函数转换 tensor 缓冲区

    ```python
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.fc2 = nn.Linear(8, 8)

        def forward(self, x):
            x = torch.zeros(x.shape)  # 在运行时将其转换为 DTensor 的 zeros
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    dmlp = parallelize_module(MLP(), ...,
    """

    # for performance, update debug env once here
    DebugLogger.update_vescale_debug_mode_from_env()

    if DModule.is_dmodule(module):
        warnings.warn(f"{module} is already parallelized `DModule`. Skip `parallelize_module`", UserWarning)
        return module

    # check sharding plan
    sharding_plan = DModule.check_and_sanitize_sharding_plan(sharding_plan)

    # create dmodule attributes
    DModule.initialize_attributes(module)

    # bind dmodule methods
    DModule.initialize_methods(module)

    # register mesh, plans, and more to self
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    DModule.register_sharding_plan(module, device_mesh, sharding_plan["parameter"], sharding_plan["forward"])

    # distribute params on target device mesh
    DModule.init_parameters(module, is_model_sharded)

    # install forward hooks
    DModule.init_forward(module)

    # install backward hooks
    DModule.init_backward(module)

    # post-patch submodules
    DModule.post_patch_submodules(module)

    # prepare dtensorizing factory
    DModule.prepare_factory(module, factory)

    # tag this module as parallelized dmodule
    DModule.set_dmodule(module)

    return module


is_dmodule = DModule.is_dmodule
