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

# mypy: ignore-errors
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.plan.pipeline_parallel import PipelineParallelPlan
from vescale.plan.spec import PipelineScheduleType
from vescale.pipe._schedules import (
    OneFOneBInstrcutionGenerator,
    InterleavedOneFOneBInstructionGenerator,
    ZeroBubbleVInstrcutionGenerator,
    StageDeps,
    Shape,
)
from vescale.pipe._schedules.instruction_base import VESCALE_INTRUCTION_BUILDER as builder
from vescale.pipe.p2p_communication import reset_global_counter
from vescale.devicemesh_api.api import VeDeviceMesh
from collections import OrderedDict
from typing import Callable, Iterator, List, Sequence, Union
import torch
import torch.distributed as dist
import logging
import os


logger = logging.Logger(__file__)


class PipelineEmitter:
    """Pipeline Emitter."""

    def __init__(
        self,
        deps: StageDeps,
        meshes: Sequence[DeviceMesh],
        schedule: str,
        batches: int,
        tensor_shape: Shape,
        dtype: torch.dtype,
        num_chunks: int = 1,
        input_shapes: List[Shape] = None,
        input_shapes_unpad: List[Shape] = None,
        forward_only=False,
        overlap_p2p_comm=False,
        batch_p2p_comm: bool = True,
        param_sync_overlap=False,
        grad_sync_overlap=False,
        **kwargs,
    ):
        self.deps = deps
        self.num_stage = deps.num_stage
        self.meshes = meshes
        self.batches = batches
        self.num_chunks = num_chunks
        self.overlap_p2p_comm = overlap_p2p_comm
        self.batch_p2p_comm = batch_p2p_comm
        self.param_sync_overlap = param_sync_overlap
        self.forward_only = forward_only
        self.grad_sync_overlap = grad_sync_overlap
        if schedule == PipelineScheduleType.SIMPLE_1F1B:
            self.num_meshes = meshes
            self.instruction_generator = OneFOneBInstrcutionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                forward_only=self.forward_only,
            )

        elif schedule == PipelineScheduleType.INTERLEAVED_1F1B:
            self.instruction_generator = InterleavedOneFOneBInstructionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                input_shapes=input_shapes,
                input_shapes_unpad=input_shapes_unpad,
                num_chunks=self.num_chunks,
                batch_p2p_comm=batch_p2p_comm,
                overlap_p2p_comm=overlap_p2p_comm,
                param_sync_overlap=param_sync_overlap,
                grad_sync_overlap=grad_sync_overlap,
                forward_only=forward_only,
            )

        elif schedule == PipelineScheduleType.ZERO_BUBBLE:
            self.instruction_generator = ZeroBubbleVInstrcutionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                **kwargs,
            )
        else:
            raise NotImplementedError("unsupport schedule type")
        self.instruction_list: List[List] = self.gen_instruction()

    def gen_instruction(self):
        """
        Generates instruction steps of a pipeline schedule.
        """
        return self.instruction_generator.gen_instruction()

    def get_instruction_list(self, stage: int):
        """
        Generates instruction steps of a pipeline schedule for a particular pipeline stage.

        Args:
            stage (int): pipeline stage id

        """
        return self.instruction_generator.get_instruction_list(stage)


class ScheduleEngine:
    """
    ===========================================================
    给初中生讲解 `ScheduleEngine` 类时，我们可以用简单易懂的语言来描述。
    ===========================================================

    想象一下我们在玩一个多人合作游戏，每个人都要按照一定的顺序完成任务，而这个 `ScheduleEngine` 类就是一个类似“队长”的角色，负责安排每个队员什么时候做什么任务，并确保所有任务顺利完成。

    ### `ScheduleEngine` 类可以做什么？

    1. **安排任务顺序** (`schedule`)：队长负责根据预定的计划安排大家的任务顺序。比如，可能先让第一个人做任务，然后下一个人继续，或者几个人同时开始任务。

    2. **准备工作** (`__init__` 方法)：一开始，队长需要知道有多少人（`meshes`），每个人要做什么任务（`stage_id`），以及这个任务需要用到哪些东西（`dtype` 和 `shape`）。他还会准备好任务清单（`instruction_list`），告诉大家什么时候开始任务。

    3. **传递物品** (`set_data_iterator`)：在任务过程中，队长可能需要把一些物品或信息传递给每个队员（`data_iterator`）。他会确保这些物品按照计划顺利传递。

    4. **检查任务结果** (`sync_output_loss_per_pipeline`)：在任务结束后，队长会检查每个人的任务结果，并确保这些结果被其他人共享，特别是在调试（`debug_mode`）的时候。

    5. **执行任务** (`execute` 方法)：这个方法是队长的主要任务。它负责启动所有的任务，并确保每个人都按照计划完成任务。如果某个队员完成了任务，队长还会计算整个队伍的总得分（`minibatch_loss`）。

    ### 举个例子：
    假设你和几个同学一起做一个大项目，比如制作一个机器人。每个同学负责机器人不同的部分，有的负责头，有的负责手臂，有的负责腿。`ScheduleEngine` 就像是项目经理，安排大家的工作顺序，确保每个人都知道自己要做什么，并且当你们做完自己的部分后，把所有部分组合在一起。

    在 `ScheduleEngine` 的世界里，这些任务、部分和顺序都以代码的形式出现，项目经理（`ScheduleEngine`）负责协调这些代码，让整个机器人顺利完成。


    ========================
    深入每个函数, 再讲解给初中生.
    ========================

    我们可以深入每个函数，用更细致的方式来讲解 `ScheduleEngine` 类，依然保持语言简单易懂，让初中生能明白它们的作用。

    ### 1. `__init__` 方法
    **作用**：这是 `ScheduleEngine` 类的构造函数，意思是当我们要创建一个新的 `ScheduleEngine` 对象时，这个方法会首先运行，来做一些准备工作。

    **详细讲解**：
    - 当我们创建一个新项目时，比如搭建一个机器人，这个 `__init__` 方法就像是我们开始前的准备步骤。我们要决定需要多少人（`meshes`），每个人负责什么部分（`stage_id`），需要用到什么材料（`dtype` 和 `shape`）。
    - 这个方法还会设置一个“指挥中心”（`PipelineEmitter`），来确保每个人按计划进行。
    - 它还会安排好任务清单（`instruction_list`），告诉每个人什么时候该做什么。
    - 最后，它会创建一个“通讯频道”，让大家能互相分享信息，特别是在调试的时候。

    ### 2. `set_data_iterator` 方法
    **作用**：这个方法用于把要处理的数据分配给每个团队成员。

    **详细讲解**：
    - 假设我们有一堆零件需要分发给负责机器人不同部分的同学，`set_data_iterator` 方法就像是负责把这些零件分发给大家的工具。每个同学得到的零件都必须符合他们负责部分的需求（`data_shape`）。
    - 它还会更新队长（`ScheduleEngine`）的任务清单，确保每个人都拿到自己需要的东西。

    ### 3. `get_instruction_list` 方法
    **作用**：根据队员的编号，获取这个队员的任务清单。

    **详细讲解**：
    - 假设你是某个同学，负责机器人的手臂部分。当你想知道自己下一步该做什么时，你可以问队长，他会用 `get_instruction_list` 方法告诉你你应该做的事情。

    ### 4. `sync_output_loss_per_pipeline` 方法
    **作用**：在调试模式下，这个方法确保每个队员完成任务后，把结果共享给其他队员。

    **详细讲解**：
    - 当你完成了你的部分，比如你装好了机器人的手臂，你会把它交给队长来检查。队长用这个方法确保每个人都能看到你的成果，并且能根据你的工作继续完成他们的部分。尤其是在你们一起调试这个机器人时，这个方法特别有用。

    ### 5. `_collect_microbatch_losses` 方法
    **作用**：这个方法收集每个任务的小结果，并把它们合并成一个最终结果。

    **详细讲解**：
    - 想象一下，每个同学在做自己的部分时，可能会有一些小问题需要解决，比如手臂的零件可能不太合适。这个方法就像是队长收集所有这些小问题，并计算出整个项目的总问题。这可以帮助你们更好地理解整个项目的进展。

    ### 6. `execute` 方法
    **作用**：这是整个 `ScheduleEngine` 类的核心功能，负责启动并执行所有的任务。

    **详细讲解**：
    - 当所有准备工作都完成后，队长最终会下令开始工作。这就是 `execute` 方法的作用。它会依照之前设置好的计划（`schedule`），让每个同学开始工作，完成他们负责的部分。
    - 当每个人完成任务后，队长会用这个方法检查所有的成果，确保每个部分都合适，然后把它们组合起来。
    - 如果项目进行得非常顺利，你们会看到最终的机器人组装完成并正常工作。如果有问题，队长还会用这个方法调试并解决问题。

    ### 总结
    每个方法就像是队长在不同阶段的指示，确保整个团队顺利完成任务，最终达成目标。这些方法一起工作，就能让一个复杂的项目（比如机器人）在多个同学的合作下顺利完成。

    """

    def __init__(
        self,
        deps: StageDeps,
        meshes: int,
        schedule: PipelineScheduleType,
        batches: int,
        data_iterator: Union[Iterator, List[Iterator]],
        stage_id: int,
        shape: Union[Shape, Sequence[Shape]],
        dtype: Union[torch.dtype, Sequence[torch.dtype]] = torch.float32,
        num_chunks=1,
        input_shapes: List[Shape] = None,
        input_shapes_unpad: List[Shape] = None,
        forward_only=False,
        overlap_p2p_comm=False,
        batch_p2p_comm: bool = True,
        param_sync_overlap=False,
        grad_sync_overlap=False,
        send_dtypes_map: OrderedDict = None,
        loss_fn: Callable = lambda x: torch.sum(x),
        global_mesh: VeDeviceMesh = None,
        **kwargs,
    ):
        """
        我们来细化讲解这个 `__init__` 方法中的每一个变量，帮助初中生理解它们的作用。

        ### 1. `deps: StageDeps`
        - **作用**：`deps` 是一个包含阶段依赖关系的变量，代表着每个任务之间的先后顺序。
        - **类比**：想象你们在制作机器人，每个部分的制作顺序是固定的。`deps` 就是一个清单，告诉你们必须先做哪些部分，再做哪些部分。

        ### 2. `meshes: int`
        - **作用**：`meshes` 表示我们有多少组队员在同时工作。每个组可以理解为一个“工作小队”。
        - **类比**：假设你们有3个小队，每个小队负责不同的机器人部分，比如一个小队负责头部，一个小队负责身体，另一个负责腿部。

        ### 3. `schedule: PipelineScheduleType`
        - **作用**：`schedule` 是一个时间表，决定每个小队的工作顺序和方式。
        - **类比**：就像在学校的课程表一样，`schedule` 决定了哪些小队先工作，哪些后工作，以及它们是否可以同时进行任务。

        ### 4. `batches: int`
        - **作用**：`batches` 表示任务被分成多少批次来完成。每个批次是一组需要处理的数据或任务。
        - **类比**：如果你们要组装很多个机器人，你们可以一次组装一批（比如10个机器人），而不是全部一起做。这样可以让工作更有条理。

        ### 5. `data_iterator: Union[Iterator, List[Iterator]]`
        - **作用**：`data_iterator` 是一个数据提供者，负责按顺序提供给每个小队需要的零件或材料。
        - **类比**：就像有一个发零件的同学，他会根据需要把零件发给你们每个小队，让你们能够完成自己的部分。

        ### 6. `stage_id: int`
        - **作用**：`stage_id` 表示当前这个 `ScheduleEngine` 负责的任务阶段编号。
        - **类比**：每个小队都有一个编号，比如1号小队负责第一个任务，2号小队负责第二个任务。`stage_id` 就是小队的编号。

        ### 7. `shape: Union[Shape, Sequence[Shape]]`
        - **作用**：`shape` 表示你们要处理的数据或零件的形状和大小。
        - **类比**：假设你们需要的零件有不同的形状，比如方形或圆形。`shape` 就是这些零件的描述，让你们知道要拿到什么样的零件。

        ### 8. `dtype: Union[torch.dtype, Sequence[torch.dtype]] = torch.float32`
        - **作用**：`dtype` 是数据的类型，比如整数、小数等等。
        - **类比**：想象你们需要不同类型的材料来制作机器人，比如金属、塑料。`dtype` 就是这些材料的类型。

        ### 9. `num_chunks=1`
        - **作用**：`num_chunks` 表示你们把任务分成多少小块来完成。
        - **类比**：如果一个任务太大，你们可以把它分成几块小任务来完成。`num_chunks` 就是这些小任务的数量。

        ### 10. `input_shapes: List[Shape] = None`
        - **作用**：`input_shapes` 是输入数据的形状列表，告诉每个小队他们会得到什么样的数据。
        - **类比**：如果你们需要不同形状的零件，`input_shapes` 会列出这些零件的形状，确保你们知道该怎么使用它们。

        ### 11. `input_shapes_unpad: List[Shape] = None`
        - **作用**：`input_shapes_unpad` 是未填充（padding）的输入数据形状，有时为了数据对齐，我们会在数据中添加额外的空白，这个变量告诉我们原始的数据形状。
        - **类比**：如果你们拿到的零件有一些多余的包装，这个变量会告诉你们这些包装去掉后，零件的真实形状。

        ### 12. `forward_only=False`
        - **作用**：`forward_only` 是一个布尔值，表示任务是否只进行“前向”计算，而不进行“反向”计算。
        - **类比**：如果你们只需要做任务的一部分，比如只负责安装而不需要测试，`forward_only` 就是“只安装”的指示。

        ### 13. `overlap_p2p_comm=False`
        - **作用**：`overlap_p2p_comm` 表示是否允许不同小队在执行任务时可以同时交换数据。
        - **类比**：如果两个小队在工作时需要互相传递信息，这个变量决定他们是否可以在同一时间进行交流。

        ### 14. `batch_p2p_comm: bool = True`
        - **作用**：`batch_p2p_comm` 表示是否在批次之间进行数据交换。
        - **类比**：如果你们在组装每一批机器人时需要互相沟通，这个变量决定这种沟通是否在每一批次之间都发生。

        ### 15. `param_sync_overlap=False`
        - **作用**：`param_sync_overlap` 决定是否在工作过程中同步参数（比如你们要用到的一些规则或设置）。
        - **类比**：就像在组装机器人时，你们需要确保大家使用相同的工具或标准。这个变量决定这些标准是否会同步。

        ### 16. `grad_sync_overlap=False`
        - **作用**：`grad_sync_overlap` 决定是否在工作过程中同步每个小队的进展（特别是在任务完成后）。
        - **类比**：如果你们在工作时需要互相分享进展，这个变量决定这些信息是否会同步给大家。

        ### 17. `send_dtypes_map: OrderedDict = None`
        - **作用**：`send_dtypes_map` 是一个有序字典，存储了不同数据类型的发送顺序和映射关系。
        - **类比**：想象你们有一份表格，上面列出了不同材料的传递顺序，这个变量就是那个表格。

        ### 18. `loss_fn: Callable = lambda x: torch.sum(x)`
        - **作用**：`loss_fn` 是一个计算损失的函数，也就是评估任务完成得有多好或差。
        - **类比**：这个函数就像是你们完成机器人组装后的打分标准，决定你们的机器人做得有多好。

        ### 19. `global_mesh: VeDeviceMesh = None`
        - **作用**：`global_mesh` 是一个全局网络的描述，表示所有小队的分布情况和联系方式。
        - **类比**：`global_mesh` 就像是一个地图，显示了所有小队在哪里，以及它们之间如何互相联系。

        ### 20. `**kwargs`
        - **作用**：`**kwargs` 是一种收集额外参数的方式，允许你传递额外的信息或设置。
        - **类比**：假设你们在做项目时，除了标准的材料和工具，你们还需要一些额外的东西，这些额外的需求就通过 `**kwargs` 来表达。

        ### 内部变量
        1. **`os.environ["STAGE_ID"] = str(stage_id)`**：
            - **作用**：把当前阶段的编号存储在系统环境变量中，供后续使用。
            - **类比**：就像在你的笔记本上写下你现在负责的任务编号，方便后面查看。

        2. **`self.p_emmiter`**：
            - **作用**：这是一个“指挥中心”，用来发出指令，确保每个小队都按计划行动。
            - **类比**：这个 `PipelineEmitter` 就像是负责传达指令的总指挥，他会根据之前的计划安排大家的工作。

        3. **`self.schedule`**：
            - **作用**：存储任务的时间表，决定每个小队的工作顺序。
            - **类比**：这个变量就像是你的课程表，告诉你什么时候该上哪门课。

        4. **`self.deps`**：
            - **作用**：存储任务的依赖关系，确保每个任务按顺序完成。
            - **类比**：这个变量就像是你们做事情的步骤清单，确保你们不会跳过某个步骤。

        5. **`self.instruction_list`**：
            - **作用**：存储当前阶段的任务指令清单，告诉每个小队具体要做什么。
            - **类比**：这个清单就像是你的作业列表，确保你知道每一步要做什么。

        6. **`self.stage_id`**：
            - **作用**：存储当前小队的编号，表明这个 `ScheduleEngine` 对象负责哪个任务阶段。
            - **类比**：就像在你的名牌上写下你的
        """
        os.environ["STAGE_ID"] = str(stage_id)
        self.p_emmiter = PipelineEmitter(
            deps,
            meshes,
            schedule,
            batches,
            shape,
            dtype,
            num_chunks=num_chunks,
            input_shapes=input_shapes,
            input_shapes_unpad=input_shapes_unpad,
            forward_only=forward_only,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            param_sync_overlap=param_sync_overlap,
            grad_sync_overlap=grad_sync_overlap,
            **kwargs,
        )
        self.schedule = schedule
        self.deps = deps
        self.instruction_list = self.get_instruction_list(stage_id)
        self.stage_id = stage_id
        self.shape = shape
        self.dtype = dtype
        self.chunk = num_chunks
        self.send_dtypes_map = send_dtypes_map
        builder.topo = deps
        builder.dataloader = data_iterator
        builder.loss_fn = loss_fn
        self.src_loss_rank = -1
        self.global_mesh = global_mesh
        if self.global_mesh:
            all_ranks = list(range(dist.get_world_size()))
            dp_rank = self.global_mesh.get_data_parallel_rank()
            tp_rank = self.global_mesh.get_tensor_parallel_rank()
            same_pipeline_group = [
                rank for rank in all_ranks if self.global_mesh.get_strategy_coordinate(rank)[1:] == [dp_rank, tp_rank]
            ]
            for rank in same_pipeline_group:
                if self.global_mesh.get_strategy_coordinate(rank)[0] == self.global_mesh.size(0) - 1:
                    self.src_loss_rank = rank
                    break
            # the group for all ranks in the same pipeline to share final loss outputs
            self.sync_loss_group = dist.new_group(ranks=same_pipeline_group, backend="nccl")

    def set_data_iterator(self, data_iterator: List, data_shape=None):
        """
        Assigns minibatch data to instruction builder.

        Args:
            data_iterator (List): a minibatch list of microbatch data

        """
        assert builder.dataloader
        builder.dataloader = data_iterator
        if data_shape:
            self.shape = data_shape
            builder.constant_data["shape"] = data_shape

    def get_instruction_list(self, stage_id):
        return self.p_emmiter.get_instruction_list(stage_id)

    def sync_output_loss_per_pipeline(self, loss: torch.Tensor):
        """
        A debug mode function that synchronizes minibatch loss
        with all stages of a pipeline.

        Args:
            data_iterator (List): a minibatch list of microbatch data

        """
        assert self.global_mesh, "Must initialize per-pipeline dist group before synchronizing loss!"
        if loss is None:
            loss = torch.tensor(0.0, dtype=torch.float).cuda(dist.get_rank())
        dist.broadcast(loss, src=self.src_loss_rank, group=self.sync_loss_group)

        # monkey patch torch.tensor loss backward as empty tensor to make it a dummy function
        def _empty_backward():
            return None

        loss.backward = _empty_backward
        return loss

    def _collect_microbatch_losses(self, outputs):
        # monkey patch torch.tensor loss backward as empty tensor to make it a dummy function
        def _empty_backward():
            return None

        output_losses = []
        for microbatch_output, microbatch_loss in outputs:
            if microbatch_loss is None:
                if isinstance(microbatch_output, Sequence):
                    for j in range(len(microbatch_output)):
                        if microbatch_output[j].ndim == 0 and microbatch_output[j].numel() == 1:
                            loss_value = microbatch_output[j]
                            break
                    else:
                        raise ValueError("Loss values not found.")
                else:
                    loss_value = microbatch_output
            else:
                # monkey patch microbatch loss backward as empty tensor to make it a dummy function
                loss_value = microbatch_loss
            output_losses.append(loss_value)
        if not output_losses:
            return None
        tensor_device = output_losses[0].device
        minibatch_loss = torch.tensor(sum(output_losses), device=tensor_device)
        minibatch_loss.backward = _empty_backward
        return minibatch_loss

    @staticmethod
    def execute(
        instance,
        *,
        deallocate_pipeline_outputs: bool = False,
        autocast_dtype: torch.dtype = torch.float,
        enable_autocast: bool = False,
        grad_scaler=None,
        param_sync_func=None,
        grad_sync_func=None,
        debug_mode=False,
    ):
        """
        Main entry point of executing forward and backward
        computation of a minibatch.

        Args:
            instance (ScheduleEngine): a minibatch list of microbatch data
            deallocate_pipeline_outputs (bool): deallocate tensors
            autocast_dtype (torch.dtype): autocast data types
            enable_autocast (bool): turn on to enable tensor autocast
            grad_scaler (Callable): gradient scaler
            param_sync_func (Callable): gradient synchronization function
            debug_mode (bool): turn on to generate debugging outputs

        Returns:
            A tuple of two elements:
                1). loss of this minibatch of data,
                2). a list of tuple of outputs per microbatch, where for each tuple:
                    - 2.1). the first element is output of the original model
                    - 2.2). the second element is the loss of this microbatch.
                        If loss_fn is not provided at initialization, it means loss
                        is computed in 2.1) and here will return None

        """
        reset_global_counter()
        if instance.schedule == PipelineScheduleType.SIMPLE_1F1B:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        elif instance.schedule == PipelineScheduleType.INTERLEAVED_1F1B:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
                param_sync_func=param_sync_func,
                grad_sync_func=grad_sync_func,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        elif instance.schedule == PipelineScheduleType.ZERO_BUBBLE:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        else:
            raise NotImplementedError("Unsupported Schedule!")


def validate_pipeline_schedule(plan: PipelineParallelPlan):
    """
    Validates pipeline schedule settings in Pipeline ParallelPlan.

    Args:
        plan (PipelineParallelPlan): configuration of pipeline parallel API attributes

    """
    if plan.schedule_type == PipelineScheduleType.INTERLEAVED_1F1B:
        assert plan.virtual_chunks > 1
    elif plan.schedule_type == PipelineScheduleType.SIMPLE_1F1B:
        assert plan.virtual_chunks == 1
