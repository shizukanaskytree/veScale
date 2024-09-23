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

from typing import List, Sequence, Optional, Dict
from collections import deque, defaultdict
from dataclasses import dataclass
from inspect import signature
import contextlib

import torch

from vescale.pipe._schedules.instruction_base import (
    InstructionGenerator,
    StageDeps,
    CommPacket,
    register_instruction,
    Shape,
    registed_functions,
    VESCALE_INTRUCTION_BUILDER as builder,
    switch_dtensor,
)
from vescale.pipe.p2p_communication import (
    recv_backward,
    recv_forward,
    send_backward,
    send_forward,
)
from vescale.dtensor._diff import manage_dump_file
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor, make_dtensor
from vescale.ndtimeline import ndtimeit_p2p
from vescale.ndtimeline.predefined import CROSS_MESH_RECV, CROSS_MESH_SEND
from torch.distributed._functional_collectives import send, recv
from vescale.dtensor.placement_types import Placement
from vescale.dtensor._utils import compute_global_tensor_info
from torch.distributed.distributed_c10d import _get_default_group

import logging

logger = logging.getLogger(__file__)


def maybe_tensor(tensor):
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor
    else:
        raise RuntimeError(f"Error parsing tensor {tensor}")


def cross_mesh_recv(comm, p2p_tensor):
    mapping_group = comm.cur_mesh.get_mapping_rank(comm.peer_mesh)
    if isinstance(mapping_group, int):  # equal size
        default_pg = _get_default_group()
        with ndtimeit_p2p(CROSS_MESH_RECV, default_pg, mapping_group, is_batched=False):
            tensor = torch.empty((3, 3), device=p2p_tensor.device, dtype=torch.int64)
            recv(tensor, mapping_group, default_pg)
            p_size = sum(tensor[:, 0] >= 0)
            tensor = tensor[:p_size]
            sharding_type = [Placement.serialize_from_tensor(p) for p in tensor]
            sharding = sharding_type
            if len(sharding_type) > 0:
                global_shape, global_stride = compute_global_tensor_info(p2p_tensor, comm.cur_mesh, sharding)
                p2p_tensor = make_dtensor(
                    p2p_tensor,
                    comm.cur_mesh,
                    sharding,
                    shape=torch.Size(global_shape),
                    dtype=p2p_tensor.dtype,
                    requires_grad=p2p_tensor.requires_grad,
                    stride=tuple(global_stride),
                )
                return p2p_tensor
            else:
                return p2p_tensor
    else:
        raise NotImplementedError("currently not support change mesh size")


def cross_mesh_send(comm, dt):
    mapping_group = comm.cur_mesh.get_mapping_rank(comm.peer_mesh)
    if isinstance(mapping_group, int):  # equal size
        default_pg = _get_default_group()
        with ndtimeit_p2p(CROSS_MESH_SEND, default_pg, mapping_group, is_batched=False):
            if isinstance(dt, DTensor):
                send_sharding = torch.stack(
                    [p.serialize_to_tensor(dt.device) for p in dt._spec.placements]
                    + [
                        torch.full((3,), -1, device=dt.device, dtype=torch.int64)
                        for _ in range(3 - len(dt._spec.placements))
                    ]
                )
                send(send_sharding, mapping_group, default_pg)
            else:  # tensor
                send(torch.full((3, 3), -1, device=dt.device, dtype=torch.int64), mapping_group, default_pg)
    else:
        raise NotImplementedError("currently not support change mesh size")


def cross_mesh_double(comm, fwd_tensor, p2p_tensor):
    if isinstance(fwd_tensor, DTensor):
        placements = fwd_tensor._spec.placements
        global_shape, global_stride = compute_global_tensor_info(p2p_tensor, comm.cur_mesh, placements)
        p2p_tensor = make_dtensor(
            p2p_tensor,
            comm.cur_mesh,
            placements,
            shape=torch.Size(global_shape),
            dtype=p2p_tensor.dtype,
            requires_grad=p2p_tensor.requires_grad,
            stride=tuple(global_stride),
        )
    return p2p_tensor


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    chunk: int
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    rollback: bool = False

    def get_send_comms(self, total_stages, deps):
        if self.chunk == 0:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage + 1),
                        input_id=0,
                        peer_stage=self.stage + 1,
                    )
                ]
                if self.stage != total_stages
                else []
            )
        else:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage - 1),
                        input_id=0,
                        peer_stage=self.stage - 1,
                    )
                ]
                if self.stage != 0
                else []
            )

    def get_recv_comms(self, total_stages, deps):
        if self.chunk == 0:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage - 1),
                        input_id=0,
                        peer_stage=self.stage - 1,
                    )
                ]
                if self.stage != 0
                else []
            )
        else:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage + 1),
                        input_id=0,
                        peer_stage=self.stage + 1,
                    )
                ]
                if self.stage != total_stages
                else []
            )


class CostGraph:
    def __init__(self, n_stage, n_micro, f_cost, b_cost, w_cost, c_cost, f_mem, b_mem, w_mem, max_mem=None):
        self.n_node = 6 * n_stage * n_micro
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.f_mem = f_mem
        self.b_mem = b_mem
        self.w_mem = w_mem
        self.fbw_cost = [f_cost, b_cost, w_cost]
        self.fbw_mem = [f_mem, b_mem, w_mem]
        self.max_mem = max_mem or f_mem * self.n_stage * 2

    def get_id(self, cat, chunk, stage, micro):
        return (
            cat * 2 * self.n_stage * self.n_micro + chunk * self.n_stage * self.n_micro + stage * self.n_micro + micro
        )

    """
    死皮赖脸的问 chatgpt.
    https://github.com/shizukanaskytree/veScale.git
    2024-0828-pp-timeline
    commit id: 97a8c0f7b4b14f2d6bdf09e8dee4480abd30ec0f
    """
    def try_v_schedule(self, fill_f=True, fill_b=True, approved_bubble=None):
        """
        这段代码实现了一个复杂的调度算法，用于优化深度学习任务中的计算和内存使用。代码涉及多个变量，这里结合具体数值解释每个变量的作用：

        1. **self.b_cost = 4**: 表示`B`操作的计算开销为4单位。
        2. **self.b_mem = -48**: `B`操作减少的内存量为48单位（负值表示释放内存）。
        3. **self.c_cost = 1**: 每次切换阶段的开销为1单位，表示`W`操作与`B`或`F`之间的间隙。
        4. **self.f_cost = 6**: `F`操作的计算开销为6单位。
        5. **self.f_mem = 72**: `F`操作消耗72单位的内存。
        6. **self.fbw_cost = [6, 4, 4]**: 这是`F`、`B`、`W`操作对应的计算开销数组，分别为6、4和4单位。
        7. **len(self.fbw_cost) = 3**: 说明`F`、`B`、`W`操作的种类有3种，分别对应前向计算、后向计算和权重更新。
        8. **self.fbw_mem = [72, -48, -24]**: 这是`F`、`B`、`W`操作对应的内存变化，分别为72、-48和-24，表示`F`消耗内存，`B`和`W`释放内存。
        9. **len(self.fbw_mem) = 3**: 对应3个操作的内存使用情况。
        10. **self.max_mem = 576**: 表示在整个调度过程中，单阶段的最大内存上限为576单位。
        11. **self.n_micro = 8**: 表示总共有8个微批次（microbatches）需要处理。
        12. **self.n_node = 192**: 表示总共有192个节点参与计算。
        13. **self.n_stage = 4**: 表示模型被分成4个阶段，每个阶段执行一定的计算。
        14. **self.w_cost = 4**: `W`操作（权重更新）的计算开销为4单位。
        15. **self.w_mem = -24**: `W`操作释放的内存量为24单位。

        其他变量的作用和意义：

        1. **count**: 这是一个二维列表，存储每个阶段中不同操作的执行次数。
        2. **end_time**: 这是一个长度为`n_node`的列表，用来记录每个节点结束时间，初始值为-1。
        3. **cur_time**: 记录每个阶段当前的时间进度。
        4. **mem**: 记录每个阶段当前的内存使用量。
        5. **stage_bubble**: 存储每个阶段的空闲时间（bubble），用于衡量不同阶段的计算效率。
        6. **pending_w**: 用于存储各阶段等待执行的`W`操作。
        7. **schedule**: 用于存储调度表，记录每个阶段已经执行的操作。
        8. **stage_str**: 用于显示每个阶段的执行状态字符串，用于调试或输出。

        通过这些变量，代码的核心任务是逐步调度`F`、`B`、`W`三种操作，并在内存和计算资源允许的范围内最大化执行效率。
        """
        count = []
        for i in range(self.n_stage):
            count.append([0] * 6)

        end_time = [-1] * self.n_node
        cur_time = [0] * self.n_stage
        mem = [0] * self.n_stage
        stage_bubble = [0] * self.n_stage
        pending_w = [deque() for _ in range(self.n_stage)]
        schedule = [[] for _ in range(self.n_stage)]
        stage_str = ["    " * i for i in range(self.n_stage)]

        if approved_bubble is None:
            approved_bubble = [-1] * self.n_stage
        max_approved_bubble = max(approved_bubble)

        def get_max_stage_bubble(stage=-1):
            max_stage_bubble = 0
            for bb in stage_bubble:
                max_stage_bubble = max(max_stage_bubble, bb)
            if stage >= 0:
                max_stage_bubble = max(max_stage_bubble, max_approved_bubble - approved_bubble[stage])
            return max_stage_bubble

        def put_w(stage):
            assert len(pending_w[stage]) > 0
            _, chunk_, _ = pending_w[stage].popleft()
            put(2, chunk_, stage)

        def put(cat, chunk, stage, assert_cnt=True):
            """
            ### 任务类型解释

            根据代码的说明：

            - `cat`（任务类别）分为三种类型（F, B, W），分别代表不同的计算操作（例如前向传播、反向传播和权重更新）。
            - 每个 `cat` 都分为两块 `chunk`，因此对于每个 `cat`，我们有两个任务块（chunk 0 和 chunk 1）。

            ### 分析

            假设 `count` 中的每个数值代表一个阶段中不同任务的执行情况：

            - `count[0][0] = 8`：第一个阶段中，任务 F 的第一个 chunk 执行了 8 次。
            - `count[0][1] = 6`：第一个阶段中，任务 F 的第二个 chunk 执行了 6 次。
            - `count[0][2] = 6`：第一个阶段中，任务 B 的第一个 chunk 执行了 6 次。
            - `count[0][3] = 3`：第一个阶段中，任务 B 的第二个 chunk 执行了 3 次。
            - `count[0][4] = 6`：第一个阶段中，任务 W 的第一个 chunk 执行了 6 次。
            - `count[0][5] = 3`：第一个阶段中，任务 W 的第二个 chunk 执行了 3 次。

            其他阶段（如 `count[1]`, `count[2]`, `count[3]`）的解释类似。

            在这段代码中，`count` 是一个二维列表，表示不同阶段 (`stage`) 下不同任务类型的执行次数。这个二维列表中的每个元素都是一个整数，表示特定任务在特定阶段被执行的次数。

            让我们逐步解析 `count = [[8, 6, 6, 3, 6, 3], [8, 7, 6, 3, 5, 3], [8, 7, 5, 4, 5, 3], [8, 8, 5, 4, 4, 4]]` 中的数字含义：

            ### 解释二维列表的结构

            - `count[i][j]`：表示第 `i` 个 `stage`（阶段）中，第 `j` 个任务的执行次数。

            - 每一行 `[8, 6, 6, 3, 6, 3]` 代表一个阶段（`stage`），比如：
            - `count[0]` 表示第一个阶段的执行情况，其中：
                - `count[0][0] = 8` 表示第一个任务类型的第一块（chunk 0）在第一个阶段执行了8次。
                - `count[0][1] = 6` 表示第一个任务类型的第二块（chunk 1）在第一个阶段执行了6次。
                - `count[0][2] = 6` 表示第二个任务类型的第一块在第一个阶段执行了6次。
                - `count[0][3] = 3` 表示第二个任务类型的第二块在第一个阶段执行了3次。
                - 以此类推。

            ### 总结

            - `count` 表示了在不同阶段中，各种任务类型的执行次数。
            - 每个任务类型（F, B, W）又分为两个块（chunk 0 和 chunk 1），所以每个 `stage` 中会有 6 个数字，分别表示这 6 个任务块的执行次数。

            这些信息用来确保每个阶段中的任务按照指定的次数被执行，以实现正确的调度和优化。
            """


            """
            stage_str = list:
            [
                'F1  F2  F3  F4  F5  F6  F7  f1  B1  W1  f2  B2  ... b1  w1  f5  B5  W5  b2  w2  f6  B6  W6  b3  w3  ',
                '    F1  F2  F3  F4  F5  f1  F6  f2  B1  W1  f3  ... B4  W4  F8  b2  w2  f6  B5  W5  b3  w3  f7  B6  ',
                '        F1  F2  F3  f1  F4  f2  F5  f3  B1  W1  ... b2  w2  f6  B4  W4  F8  b3  w3  f7  B5  W5  b4  ',
                '            F1  f1  F2  f2  F3  f3  F4  f4  B1  ... W3  F7  b3  w3  f7  B4  W4  F8  b4  w4  f8  B5  '
            ]
            """

            _tmp = _no_bubble = cur_time[stage] + self.fbw_cost[cat]
            _cnt = count[stage][cat * 2 + chunk]
            if _cnt >= self.n_micro:
                if not assert_cnt:
                    stage_str[stage] += "    "
                    cur_time[stage] = _tmp  # TODO
                    return
                raise AssertionError()
            assert mem[stage] + self.fbw_mem[cat] <= self.max_mem
            stage_str[stage] += "FfBbWw"[cat * 2 + chunk] + str(_cnt + 1) + " " * (3 - len(str(_cnt + 1)))
            if cat > 0 or chunk > 0:
                last_id = cat * 2 + chunk - 1
                if cat < 2:
                    assert end_time[self.get_id(last_id // 2, last_id % 2, stage, _cnt)] >= 0
                else:
                    assert end_time[self.get_id(1, chunk, stage, _cnt)] >= 0
            if chunk == 1 and cat < 2:
                if stage < self.n_stage - 1:
                    _fa_id = self.get_id(cat, chunk, stage + 1, _cnt)
                    assert end_time[_fa_id] >= 0
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            if chunk == 0 and cat < 2:
                if stage > 0:
                    _fa_id = self.get_id(cat, chunk, stage - 1, _cnt)
                    assert end_time[_fa_id] >= 0, f"{cat}, {chunk}, {stage}, {_cnt}"
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            _id = self.get_id(cat, chunk, stage, _cnt)
            if count[stage][0] > 0:
                stage_bubble[stage] += _tmp - _no_bubble
            end_time[_id] = _tmp
            cur_time[stage] = _tmp
            mem[stage] += self.fbw_mem[cat]
            # noinspection PyTypeChecker
            schedule[stage].append((cat, chunk, _cnt))
            if cat == 1:
                pending_w[stage].append((2, chunk, _cnt))
            count[stage][cat * 2 + chunk] += 1

        for i in range(self.n_stage):
            put(0, 0, i)
        for i in range(self.n_stage - 1, -1, -1):
            if i == self.n_stage - 1:
                put(0, 1, i)
                continue
            tmp = end_time[self.get_id(0, 1, i + 1, 0)] + self.c_cost
            while (
                mem[i] + self.fbw_mem[0] * (2 + i * 2) <= self.max_mem
                and cur_time[i] + self.fbw_cost[0] <= tmp
                and count[i][0] < self.n_micro
            ):
                for j in range(i + 1):
                    put(0, 0, j)
            put(0, 1, i)
        iter_chunk_ = 0
        end_tmp = 0
        for i in range(self.n_stage):
            if i == 0:
                end_tmp = cur_time[0] + self.fbw_cost[1]
                continue
            tmp = end_tmp + self.c_cost
            while (
                count[i][0] + count[i][1] < count[i - 1][0] + count[i - 1][1]
                or count[i][1] <= count[i - 1][1] < self.n_micro
            ):
                for j in range(self.n_stage - 1, i - 1, -1):
                    if count[j][iter_chunk_] < self.n_micro:
                        put(0, iter_chunk_, j)
                iter_chunk_ = 1 - iter_chunk_

        for _ in range(2 * self.n_micro):
            # check mem before putting b
            for i in range(self.n_stage):
                while mem[i] + self.fbw_mem[1] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
            b0_ranks, b1_ranks = [], []
            for i in range(self.n_stage):
                if count[i][3] >= count[i][2]:
                    b0_ranks.append(i)
                elif i == self.n_stage - 1:
                    b1_ranks.append(i)
                else:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                    if end_time[fa_id] >= 0 or count[i][2] >= self.n_micro:
                        b1_ranks.append(i)
                    else:
                        b0_ranks.append(i)
            b_ranks = []
            # put b1
            for i in reversed(b1_ranks):
                b_ranks.append((i, 1))
            # put b0
            for i in b0_ranks:
                b_ranks.append((i, 0))
            for i, _chunk_ in b_ranks:
                fa_id = -1
                if _chunk_ == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                if _chunk_ == 0 and i > 0:
                    fa_id = self.get_id(1, 0, i - 1, count[i][2])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if _chunk_ == 1:
                        put_w(i)
                    elif fill_b:
                        put_w(i)
                put(1, _chunk_, i)

            # put f
            for i in range(self.n_stage):
                if count[i][1] >= self.n_micro:
                    continue
                put_item = None
                if count[i][1] >= count[i][0]:
                    put_item = 0
                elif i == self.n_stage - 1:
                    put_item = 1
                else:
                    if end_time[self.get_id(0, 1, i + 1, count[i][1])] >= 0:
                        put_item = 1
                    elif count[i][0] < self.n_micro:
                        if i == 0:
                            put_item = 0
                        elif end_time[self.get_id(0, 0, i - 1, count[i][0])] >= 0:
                            put_item = 0
                if put_item is None:
                    continue
                # check mem before putting f
                while mem[i] + self.fbw_mem[0] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
                fa_id = -1
                if put_item == 0 and i > 0:
                    fa_id = self.get_id(0, 0, i - 1, count[i][0])
                if put_item == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(0, 1, i + 1, count[i][1])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if fill_f:
                        put_w(i)
                put(0, put_item, i)

        for i in range(self.n_stage):
            while len(pending_w[i]) > 0:
                put_w(i)

        max_bubble = get_max_stage_bubble()
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        bubble_rate = max_bubble / expected_time
        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            _schedule, _end_time, _max_bubble = self.try_v_schedule(
                fill_f=fill_f,
                fill_b=fill_b,
                approved_bubble=stage_bubble,
            )
            if _max_bubble < max_bubble:
                return _schedule, _end_time, _max_bubble
        return schedule, end_time, max_bubble

    def print_details(self, end_time, print_scaling=1):
        for stage in range(self.n_stage):
            stage_str = ["."] * int(max(end_time) / print_scaling)
            for _cat in range(3):
                for _chunk in range(2):
                    for _micro in range(self.n_micro):
                        _id = self.get_id(_cat, _chunk, stage, _micro)
                        if end_time[_id] < 0:
                            continue
                        end = int(end_time[_id] / print_scaling)
                        start = int((end_time[_id] - self.fbw_cost[_cat]) / print_scaling)
                        for j in range(start, end):
                            if j == start or j == end - 1:
                                stage_str[j] = "FfBbWw"[_cat * 2 + _chunk]
                            elif j == start + 1:
                                if _micro >= 10:
                                    stage_str[j] = str(_micro // 10)
                                else:
                                    stage_str[j] = str(_micro)
                            elif j == start + 2 and _micro >= 10:
                                stage_str[j] = str(_micro % 10)
                            else:
                                stage_str[j] = "-"
            _str = ""
            for _c in stage_str:
                _str += _c
            print(_str)

    def get_v_schedule(self, only_run_time=False):
        schedule, end_time, max_bubble = None, None, None
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        for fill_b in [True, False]:
            for fill_f in [True, False]:
                _schedule, _end_time, _max_bubble = self.try_v_schedule(fill_b=fill_b, fill_f=fill_f)
                if max_bubble is None or _max_bubble < max_bubble:
                    max_bubble = _max_bubble
                    schedule = _schedule
                    end_time = _end_time
        if only_run_time:
            return max_bubble + expected_time
        bubble_rate = max_bubble / (expected_time + max_bubble)
        msg = "%2d %3d, [%5d %5d %5d %5d], %6d -> %6.4f" % (
            self.n_stage,
            self.n_micro,
            *self.fbw_cost,
            self.c_cost,
            self.max_mem // self.f_mem,
            bubble_rate,
        )

        logger.info(msg)
        local_order = [[] for _ in range(self.n_stage)]
        comm_id = {}
        comm_id_counter = 0
        post_validation_time = 0
        for i in range(self.n_stage - 1, -1, -1):
            pv_id = min(2 * (self.n_stage - 1 - i), self.n_micro - 1)
            post_validation_time = max(
                post_validation_time, end_time[self.get_id(0, 0, i, pv_id)] - self.fbw_cost[0] - self.c_cost
            )
            for it in ["RECV_", "SEND_", ""]:
                if i == 0 and it == "SEND_":
                    continue
                if i == self.n_stage - 1 and it == "RECV_":
                    continue
                stage_ = i
                local_order[stage_].append(
                    ScheduledNode(
                        type=it + "POST_VALIDATION",
                        chunk=0,
                        stage=stage_,
                        minibatch=0,
                        start_time=post_validation_time,
                        completion_time=post_validation_time,
                    )
                )
                comm_id[local_order[stage_][-1]] = comm_id_counter
                comm_id_counter += 1
        for i in range(self.n_stage):
            for _cat_, _chunk_, _micro_ in schedule[i]:
                complete_time = end_time[self.get_id(_cat_, _chunk_, i, _micro_)]
                local_order[i].append(
                    ScheduledNode(
                        type="FBW"[_cat_],
                        chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                        stage=i,
                        minibatch=_micro_,
                        start_time=complete_time - self.fbw_cost[_cat_],
                        completion_time=complete_time,
                    )
                )
                if _cat_ == 2:  # no communication for W
                    continue
                cat_str = "FORWARD" if _cat_ == 0 else "BACKWARD"

                def communicate(send_recv, stage_):
                    # noinspection PyTypeChecker
                    local_order[stage_].append(
                        ScheduledNode(
                            type=send_recv + cat_str,
                            chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                            stage=stage_,
                            minibatch=_micro_,
                            start_time=complete_time,
                            completion_time=complete_time,
                        )
                    )
                    comm_id[local_order[stage_][-1]] = comm_id_counter

                if _chunk_ == 1 and i > 0:
                    communicate("SEND_", i)
                    communicate("RECV_", i - 1)
                if _chunk_ == 0 and i < self.n_stage - 1:
                    communicate("SEND_", i)
                    communicate("RECV_", i + 1)
                comm_id_counter += 1
        for rank in range(self.n_stage):
            # For nodes with the same timestamp on the same stage, communication will be prioritized.
            def even_breaker(x: ScheduledNode):
                # Compute nodes are always delayed.
                if x.type in ["F", "B", "W"]:
                    return comm_id_counter
                # For comm nodes, order by their unique comm id
                return comm_id[x]

            local_order[rank] = sorted(local_order[rank], key=lambda x: (x.start_time, even_breaker(x)))
            # If a recv with intersects with previous computation, reorder them so that recv
            # is executed before computation and hence can be overlapped.
            for i in range(len(local_order[rank])):
                if (
                    i > 0
                    and local_order[rank][i - 1].type in {"F", "B", "W"}
                    and local_order[rank][i].type.startswith("RECV")
                    and "POST_VALIDATION" not in local_order[rank][i].type
                    and local_order[rank][i].start_time <= local_order[rank][i - 1].completion_time
                ):
                    local_order[rank][i], local_order[rank][i - 1] = local_order[rank][i - 1], local_order[rank][i]

        local_order_with_rollback = [[] for _ in range(self.n_stage)]
        for rank in range(self.n_stage):
            rollback_comm = set()
            if rank > 0:
                for node in local_order[rank - 1]:
                    if node.type == "POST_VALIDATION":
                        break
                    if node.type == "SEND_FORWARD":
                        assert node.chunk == 0
                        rollback_comm.add(node.minibatch)
            for node in local_order[rank]:
                if node.type == "RECV_FORWARD" and node.chunk == 0 and node.minibatch in rollback_comm:
                    rollback = True
                    rollback_comm.remove(node.minibatch)
                else:
                    rollback = False
                local_order_with_rollback[rank].append(
                    ScheduledNode(
                        type=node.type,
                        chunk=node.chunk,
                        stage=node.stage,
                        minibatch=node.minibatch,
                        start_time=node.start_time,
                        completion_time=node.completion_time,
                        rollback=rollback,
                    )
                )
            assert len(rollback_comm) == 0
            msg = ""
            for node in local_order_with_rollback[rank]:
                msg += f"{node.type}-{node.minibatch}-{int(node.rollback)},"
            msg = msg[:-1] + "\n"
            logger.info(msg)

        return local_order_with_rollback


class ZeroBubbleVInstrcutionGenerator(InstructionGenerator):
    def __init__(
        self,
        deps: StageDeps,
        meshes: List[DeviceMesh],
        batches: int,
        f_cost: int,
        b_cost: int,
        w_cost: int,
        c_cost: int,
        f_mem: int,
        b_mem: int,
        w_mem: int,
        max_mem=None,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
    ):
        self.num_chunk = 2  # for ZBV, manually set num chunks be 2 for each worker
        self.deps = deps
        n_stage = deps.num_stage
        n_micro = batches
        self.cost_graph = CostGraph(n_stage, n_micro, f_cost, b_cost, w_cost, c_cost, f_mem, b_mem, w_mem, max_mem=None)
        self.num_stage = len(meshes)
        self.schema = self.cost_graph.get_v_schedule()
        self.default_shape = default_shape
        self.default_dtype = default_dtype

    def gen_instruction(self):
        self.instruction_list = [[] for _ in range(self.num_stage)]
        self.instruction_list_str = ["" for _ in range(self.num_stage)]

        for stage in range(self.num_stage):
            stage_str = ""
            for node in self.schema[stage]:
                self._set_inst(node, stage)
                stage_str += node.type + ","
            stage_str = stage_str[:-1]

        self.gen_instruction_str_list()

    def gen_instruction_str_list(self):
        instruction_lists = self.instruction_list
        stage_strs = defaultdict(str)
        for stage_id, instruction_list in enumerate(instruction_lists):
            cur_stage_str = stage_strs[stage_id]
            for inst in instruction_list:
                cur_stage_str += f"{VESCALE_INSTRUCTION_MAPPING_ZBV[inst.type]},"
            cur_stage_str = cur_stage_str[:-1]
            stage_strs[stage_id] = cur_stage_str
        builder.build_from_dict(stage_strs)

    @manage_dump_file
    def execute(
        self,
        stage_id,
        autocast_dtype=torch.float32,
        enable_autocast=False,
        grad_scaler=None,
        deallocate_pipeline_outputs=False,
    ):
        # init constant data
        builder.constant_data["autocast_dtype"] = autocast_dtype
        builder.constant_data["enable_autocast"] = enable_autocast
        builder.constant_data["grad_scaler"] = grad_scaler
        builder.constant_data["deallocate_pipeline_outputs"] = deallocate_pipeline_outputs
        builder.constant_data["total_stages"] = self.num_stage
        builder.constant_data["stagedeps"] = self.deps
        builder.constant_data["default_shape"] = self.default_shape
        builder.constant_data["default_dtype"] = self.default_dtype

        # Model chunk IDs with synchronized grads
        builder.user_data["synchronized_model_chunks"] = set()
        builder.user_data["input_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensor_grads"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["fwd_wait_handles"] = None
        builder.user_data["bwd_wait_handles"] = None
        builder.user_data["output_tensor"] = None
        builder.user_data["input_tensor"] = (None, None)
        builder.user_data["output_tensor_grad"] = None
        builder.user_data["forward_data_store"] = []
        model = self.deps.get_current_model(stage_id)

        builder.model = model
        instruction_list = self.get_instruction_list(stage_id)
        builder.stage_id = stage_id
        builder_instruction_list = builder.global_instructions_funcs[stage_id]

        assert len(instruction_list) == len(builder_instruction_list)
        # print(f"cur stage {stage_id} debug inst list: {instruction_list} len inst {len(instruction_list)}")

        for inst, fn in zip(instruction_list, builder_instruction_list):
            builder.user_data["inst"] = inst
            fn()

        return builder.user_data["forward_data_store"]


# communication


@register_instruction(name="vescale_zbv_send_forward")
def vescale_zbv_send_forward():
    inst = builder.user_data["inst"]
    output_tensors = builder.user_data["output_tensor"]

    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]

    def f(info):
        output_tensor, comm, shape = info
        send_forward(
            output_tensor=maybe_tensor(output_tensor),
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
            tensor_shape=shape,
        )
        cross_mesh_send(comm, output_tensor)

    comm_packages = inst.get_send_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])

    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    infos = zip(output_tensors, comm_packages, shapes)
    return list(map(f, infos))


@register_instruction(name="vescale_zbv_recv_forward")
def vescale_zbv_recv_forward():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    mbx = inst.minibatch

    def f(info):
        comm, shape, dtype = info
        p2p_tensor = recv_forward(
            tensor_shape=shape,
            recv_dtype=dtype,
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
        )
        p2p_tensor = cross_mesh_recv(comm, p2p_tensor)
        return p2p_tensor

    comm_packages = inst.get_recv_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    dtypes = [builder.constant_data["default_dtype"] for _ in comm_packages]
    infos = zip(comm_packages, shapes, dtypes)
    out = list(map(f, infos))
    input_tensor = out if len(out) > 0 else None
    builder.user_data["input_tensor"] = (input_tensor, mbx)
    builder.user_data["input_tensors"][chunk_id].append((input_tensor, mbx))
    return input_tensor


@register_instruction(name="vescale_zbv_send_backward")
def vescale_zbv_send_backward():
    inst = builder.user_data["inst"]
    input_tensor_grad = builder.user_data["input_tensor_grad"]
    if not isinstance(input_tensor_grad, list):
        input_tensor_grad = [input_tensor_grad]

    def f(info):
        grad, comm, shape = info
        send_backward(
            input_tensor_grad=maybe_tensor(grad),
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
            tensor_shape=shape,
        )
        cross_mesh_send(comm, grad)

    recv_comms = inst.get_recv_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in recv_comms]
    infos = zip(input_tensor_grad, recv_comms, shapes)
    return list(map(f, infos))


@register_instruction(name="vescale_zbv_recv_backward")
def vescale_zbv_recv_backward():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk

    def f(info):
        comm, shape, dtype = info
        p2p_tensor = recv_backward(
            tensor_shape=shape,
            recv_dtype=dtype,
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
        )
        p2p_tensor = cross_mesh_recv(comm, p2p_tensor)
        return p2p_tensor

    comm_packages = inst.get_send_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    dtypes = [builder.constant_data["default_dtype"] for _ in comm_packages]
    infos = zip(comm_packages, shapes, dtypes)
    out = list(map(f, infos))
    output_tensor_grad = out if len(out) > 0 else None

    builder.user_data["output_tensor_grad"] = output_tensor_grad
    builder.user_data["output_tensor_grads"][chunk_id].append(output_tensor_grad)
    return output_tensor_grad


# forward


@register_instruction(name="vescale_zbv_forward")
def vescale_zbv_forward():
    """
    ==========================================
    把 def vescale_zbv_forward() 讲解给初中生听懂
    ==========================================

    要向初中生解释这个 `vescale_zbv_forward()` 函数，可以用以下方式：

    ---

    **背景介绍**
    这个函数是一个计算机程序的一部分，用于处理深度学习模型的训练。深度学习模型就像是一个复杂的数学公式，通过学习大量的数据，来学会做某些任务，比如识别图片中的物体或翻译语言。

    ---

    **函数的作用**
    `vescale_zbv_forward()` 函数的作用是让模型在每个步骤中计算出结果，然后传递到下一步继续处理。可以把它想象成一个流水线，每一部分的工作完成后，结果就会被传递到下一部分继续加工。

    ---

    **代码逐步解释**

    1. **准备工作：**
    - 这个函数首先会从一些全局的变量中取出当前处理的任务（`inst`）、模型的某一部分（`cur_model`）、以及输入输出的数据存储位置。

    2. **判断是否在流水线的起点或终点：**
    - `is_pp_first_stage` 和 `is_pp_last_stage` 用来检查当前处理的部分是否是整个流程的开始或结束。如果是开始部分，意味着它需要从外部输入数据；如果是结束部分，意味着它需要输出最终结果。

    3. **准备输入数据：**
    - 函数会从已存储的数据中找到当前要处理的输入数据（`input_tensor`）。如果是流程的开始部分，并且之前没有数据存储，它会添加一个空的占位符（`None`）。

    4. **检查并启动自动类型转换（Autocast）：**
    - `enable_autocast` 是一个优化技巧，可以让计算更高效。这个部分会决定是否要开启这种优化。

    5. **实际的计算：**
    - 函数会调用模型的计算功能（`cur_model(*args, **kwargs)`），这就像是让模型在这个步骤中做数学运算。得到的结果会存储到 `output_tensor` 里。

    6. **处理结果：**
    - 如果这是流程的最后一步（`is_pp_last_stage`），函数会计算损失值（一个衡量模型表现好坏的指标），并将结果存储起来。
    - 如果还没到最后一步，它会把结果传给下一步继续处理。

    7. **结果的保存和传递：**
    - 无论当前是不是最后一步，函数都会把计算的结果（`output_tensor`）存储到全局变量中，以便后续步骤可以使用。

    ---

    **总结**
    `vescale_zbv_forward()` 就像是在工厂的流水线中，把每一个步骤完成后继续传递给下一个步骤。它确保了每一部分都能正确地收到前面的结果，并进行下一步的计算。最终，在整个流程结束后，得到完整的计算结果。
    """


    """
    ======================
    难点是解释清楚抽象的变量名
    ======================

    理解和解释变量名确实是代码理解中的一个难点，尤其是当这些变量名很抽象的时候。让我们把这些抽象的变量名转换成更容易理解的比喻，帮助初中生更好地理解它们的作用。

    ---

    ### 变量名解释

    1. **`inst`**（指令/任务）：
    想象一下，你正在执行一个任务，而 `inst` 就是用来记录这个任务的详细信息。比如，如果你在完成一个拼图，`inst` 就记录了你目前在拼的那一块。

    2. **`chunk_id`**（块编号）：
    这个可以看作是任务的一个小部分的编号。如果你的任务是拼图，那么 `chunk_id` 可能是表示你正在拼的那一块的编号。

    3. **`stage_id`**（阶段编号）：
    想象你有一个任务是分多个步骤完成的，而 `stage_id` 就是表示你现在在第几步。比如，做蛋糕时，先搅拌面糊是第一步，烘烤是第二步。

    4. **`mbx`**（小批次编号）：
    在模型训练中，数据通常分成许多小块来处理，`mbx` 表示当前处理的是第几块数据。可以想象成你在完成一个大任务时，把它分成小部分逐个完成。

    5. **`cur_model`**（当前模型）：
    `cur_model` 就像是你的助手，它是专门为你当前任务设计的工具，用来处理任务中的一部分。

    6. **`user_data`**（用户数据）：
    这是一些额外的信息或数据，程序会在执行任务时用到。就像在做实验时，你可能会有一张纸，上面写着你要记录的结果或注意事项。

    7. **`forward_data_store`**（前向数据存储）：
    这个是用来存储每一步计算结果的地方。可以想象成你在做实验时用的笔记本，每完成一步，你就把结果记下来。

    8. **`input_tensors` 和 `output_tensors`**（输入/输出张量）：
    这些是模型在每一步中需要处理的数据。可以想象成材料（输入）和成品（输出）。比如，面粉是输入，蛋糕是输出。

    9. **`autocast_dtype` 和 `enable_autocast`**（自动类型转换的数据类型和开关）：
    这就像是一个让工作更轻松的工具，它可以自动调整工具的精度，使得你在做任务时能更快、更高效。

    10. **`is_pp_first_stage` 和 `is_pp_last_stage`**（是否为流水线的第一阶段/最后阶段）：
        就像在一个生产线上，`is_pp_first_stage` 是检查你是否在开始点，而 `is_pp_last_stage` 是检查你是否在结束点。如果你在开始点，你需要把原材料准备好；如果在结束点，你需要把最终产品输出。

    11. **`prepare_data_fn`**（准备数据的函数）：
        这个是一个帮助函数，负责在你开始处理之前把所有需要的数据准备好。就像在做实验前，你会先把所有的器材和材料准备好。

    12. **`output_tensor`**（输出数据）：
        这是每次计算后得到的结果。就像你完成一小步后，得到的中间结果。

    13. **`loss_tensor`**（损失数据）：
        在机器学习中，`loss_tensor` 用来衡量模型的表现，就像你在比赛后知道自己得了多少分，用它来调整自己的表现。

    ---

    ### 用这些比喻讲解函数

    通过这些比喻，你可以这样解释函数的操作过程：

    - `inst` 是你要完成的任务，而 `chunk_id` 和 `stage_id` 就是帮助你跟踪当前任务的进度和具体步骤。
    - 你会用 `cur_model` 这个助手来处理 `input_tensors` （输入数据），并得到 `output_tensors` （输出数据）。
    - 在任务的开始和结束阶段，函数会做一些额外的准备或收尾工作，比如在开始时准备输入，在结束时计算得分（`loss_tensor`）。
    - 如果需要，`autocast` 就像一个能让你工具变得更高效的开关，开启后可以让计算更快、更省力。

    这样解释后，抽象的变量名变得更加直观和易于理解，也更符合初中生的认知水平。
    """



    """
    ====================================
    ### 讲解 `vescale_zbv_forward()` 函数
    ====================================

    这个函数负责处理一个深度学习模型中的计算步骤。为了让你更好地理解，我们可以把它想象成一个在工厂里完成任务的流程。每一步都有一个具体的任务，处理完后，结果会被传递给下一步继续处理。

    #### **开始任务**

    首先，函数会从一个叫 `inst` 的地方取出你要完成的任务。这个任务被分成很多小部分，每部分都有一个编号（`chunk_id`）和一个阶段（`stage_id`）。你可以把它想象成在完成一个复杂的拼图，每一块都有它自己的编号和步骤。

    #### **准备助手和数据**

    接着，函数会把这些信息交给一个助手（`cur_model`）来处理。为了完成任务，助手需要一些原材料，这些材料被保存在 `input_tensors` 里，而加工后的成品会放到 `output_tensors` 里。

    #### **流水线的起点和终点**

    函数还会检查你是不是在流水线的起点（`is_pp_first_stage`）或终点（`is_pp_last_stage`）。如果你在起点，意味着你需要准备好材料（输入数据）。如果你在终点，意味着你需要把加工好的成品（输出数据）交出去。

    #### **优化工具**

    为了让工作更高效，函数还会检查一个叫 `autocast` 的工具。如果这个工具被开启（`enable_autocast`），它会自动调整工具的精度，让你的计算更快、更省力。

    #### **实际的计算**

    现在，一切准备就绪，函数会用 `cur_model` 这个助手来处理输入的数据（`input_tensor`），并得到加工后的成品（`output_tensor`）。如果你在流水线的终点，助手还会计算得分（`loss_tensor`），用来衡量整个任务的完成情况。

    #### **结果的传递**

    最后，无论你是不是在终点，函数都会把计算的结果（`output_tensor`）保存起来，以便下一步继续使用。如果你在流水线的终点，结果会被传递给下一个任务，继续完成整个流程。

    ---

    ### 总结

    `vescale_zbv_forward()` 函数就像在工厂的流水线上完成任务。它负责把任务的每一步处理好，并且在每一步中确保所需的材料准备好，结果正确传递。通过这个函数，整个流程的计算可以顺利进行，最终得到满意的结果。
    """


    """
    好的，我们结合代码来形象地解释每个部分，这样你可以更好地理解代码和它对应的操作。

    ### 1. 机器人操作员的安排

    ```python
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    mbx = inst.minibatch
    cur_model = builder.model[chunk_id]
    ```

    这里，操作员首先要了解当前的任务在哪个阶段（`stage_id`）和使用哪个机器人部件（`chunk_id`）。他还需要知道当前要处理的数据是哪一小部分（`mbx`），并根据这些信息获取当前要使用的“机器人部件”（`cur_model`）。

    ### 2. 获取工具和数据

    ```python
    user_data = builder.user_data
    forward_data_store = user_data["forward_data_store"]
    input_tensors = user_data["input_tensors"]
    output_tensors = user_data["output_tensors"]

    constant_data = builder.constant_data
    autocast_dtype = constant_data["autocast_dtype"]
    enable_autocast = constant_data["enable_autocast"]
    ```

    操作员从储藏室（`user_data` 和 `constant_data`）中取出他需要的工具和材料，比如输入的“能量块”（`input_tensors`）和输出的“工具包”（`output_tensors`）。如果任务需要特殊模式（`enable_autocast`），他也会提前准备好。

    ### 3. 检查任务阶段

    ```python
    is_pp_first_stage = stage_id == 0 and chunk_id == 0
    is_pp_last_stage = stage_id == 0 and chunk_id == 1
    ```

    操作员现在要检查这是不是任务的第一步（`is_pp_first_stage`），如果是，他需要做一些额外的准备工作。如果是最后一步（`is_pp_last_stage`），他也要特别处理结果。

    ### 4. 找到合适的输入

    ```python
    input_tensor = None
    for cur_item in input_tensors[chunk_id]:
        if cur_item is not None and cur_item[1] == mbx:
            input_tensor = cur_item[0]

    if not is_pp_first_stage:
        assert input_tensor is not None
    ```

    操作员在工具箱里翻找，确保他找到了正确的“能量块”（`input_tensor`）来驱动机器。如果这不是第一步，那必须确认工具已经准备好。

    ### 5. 是否需要特殊模式

    ```python
    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    ```

    如果任务需要开启特殊模式（`autocast`），比如夜视模式，操作员会在这个“上下文管理器”（`context_manager`）里开启这个模式，否则就使用普通模式。

    ### 6. 准备任务

    ```python
    def prepare_data():
        model_chunk_id = builder.user_data["model_chunk_id"]
        ground_truth = []
        if builder.user_data["is_pp_first_stage"]:
            true_input_tensor = next(builder.dataloader[model_chunk_id])
            # keep the input tensor in builder
            if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]) + 1:
                true_input_tensor.requires_grad_()
                builder.user_data["input_tensors"][chunk_id].pop()
                builder.user_data["input_tensors"][chunk_id].append((true_input_tensor, mbx))
        else:
            local_tensors = next(builder.dataloader[model_chunk_id])
            if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
                ground_truth.append(local_tensors[-1])
            elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
                ground_truth.append(local_tensors["labels"])
            true_input_tensor = builder.user_data["p2p_tensors"]
            if isinstance(true_input_tensor, Sequence):
                true_input_tensor = true_input_tensor[0]

        return (true_input_tensor,), {}, ground_truth
    ```

    在这个部分，操作员调用了一个“准备数据”的函数。他从“数据加载器”（`builder.dataloader`）中取出当前任务需要的“真材实料”（`true_input_tensor`），这些是机器人接下来需要处理的数据。

    ### 7. 执行任务

    ```python
    args, kwargs, ground_truth = builder.user_data["prepare_data_fn"]()
    builder.user_data["ground_truth"] = ground_truth
    output_tensor = cur_model(*args, **kwargs)
    ```

    现在，操作员把准备好的数据传给机器人部件（`cur_model`），让它开始执行任务。任务完成后，机器人会返回一个“结果”（`output_tensor`），就像搬完东西后，机器人告诉你“我完成了”。

    ### 8. 任务完成后的处理

    ```python
    if is_pp_last_stage:
        output_tensor, loss_tensor = registed_functions["vescale_zbv_loss_fn"](output_tensor)
        forward_data_store.append((output_tensor, loss_tensor))
        output_tensor = output_tensor if builder.loss_fn is None else loss_tensor
    ```

    如果这是任务的最后一步，操作员会检查机器人给出的结果，并计算损失（`loss_tensor`），确保任务的效果是预期的。

    ### 9. 准备下一步

    ```python
    if stage_id + 1 == builder.constant_data["total_stages"] and chunk_id == 0:
        builder.user_data["input_tensor"] = (output_tensor, mbx)
        builder.user_data["input_tensors"][chunk_id + 1].append((output_tensor, mbx))
    ```

    如果任务还没有结束，操作员会把这次任务的结果传给下一个机器人部件，为下一步做好准备。

    ### 10. 最终收尾

    ```python
    builder.user_data["output_tensors"][chunk_id].append(output_tensor)
    user_data["output_tensor"] = output_tensor
    ```

    最后，操作员把所有结果汇总起来，完成整个任务的处理。

    通过这些操作，机器人能够一步步地完成复杂的任务，而操作员则确保每个步骤都顺利进行，最终得出满意的结果。
    """


    """
    ================================================
    让我们通过一个生动的例子来解释这个 `prepare_data` 函数。
    ================================================

    想象一下，我们要准备一道菜，而这个函数就是厨师在厨房里准备食材的步骤。

    ### 厨师的准备工作

    1. **确定使用的食材和工具：**

    ```python
    model_chunk_id = builder.user_data["model_chunk_id"]
    ground_truth = []
    ```

    首先，厨师会从冰箱里拿出需要的食材（`model_chunk_id`），并准备一个空盘子（`ground_truth`），这个盘子会用来盛装最终做好的菜。

    2. **判断是不是第一次做这道菜：**

    ```python
    if builder.user_data["is_pp_first_stage"]:
    ```

    接着，厨师要检查这是不是第一次做这道菜（`is_pp_first_stage`）。如果是第一次，那么他需要从头开始准备所有的食材。

    3. **从冰箱里取出主材料：**

    ```python
    true_input_tensor = next(builder.dataloader[model_chunk_id])
    ```

    厨师打开冰箱（`builder.dataloader`），取出主要的食材（`true_input_tensor`），比如蔬菜或肉类，这是这道菜的核心部分。

    4. **检查并调整食材：**

    ```python
    if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]) + 1:
        true_input_tensor.requires_grad_()
        builder.user_data["input_tensors"][chunk_id].pop()
        builder.user_data["input_tensors"][chunk_id].append((true_input_tensor, mbx))
    ```

    厨师会检查食材是否准备得当（`input_tensors` 和 `output_tensors`）。如果发现冰箱里的食材还没准备好（比如还需要洗或切），他会对食材进行处理（`requires_grad_()`），并把处理好的食材放回到食材列表中。

    5. **如果不是第一次做菜：**

    ```python
    else:
        local_tensors = next(builder.dataloader[model_chunk_id])
        if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
            ground_truth.append(local_tensors[-1])
        elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
            ground_truth.append(local_tensors["labels"])
        true_input_tensor = builder.user_data["p2p_tensors"]
        if isinstance(true_input_tensor, Sequence):
            true_input_tensor = true_input_tensor[0]
    ```

    如果这道菜已经做过几次了，厨师会从冰箱里直接取出一些已经准备好的配菜（`local_tensors`），并将这些配菜放到盘子里（`ground_truth`）。他还会检查之前是否有未用完的主材料（`p2p_tensors`），如果有，也会继续使用。

        这段代码的逻辑是处理“如果这不是第一次做菜”的情况。具体来说，它涉及如何处理和准备已经存在的数据。让我们逐步理解这个过程。

        ### 1. 从数据加载器获取数据

        ```python
        local_tensors = next(builder.dataloader[model_chunk_id])
        ```

        在这一步，代码从数据加载器（`builder.dataloader`）中获取一组数据（`local_tensors`），这相当于从冰箱里拿出一盘已经准备好的配菜。这个 `local_tensors` 可能包含多个元素，比如一组输入数据或者标签。

        ### 2. 检查数据的类型和内容

        ```python
        if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
            ground_truth.append(local_tensors[-1])
        ```

        接下来，代码检查 `local_tensors` 是不是一个序列（`Sequence`），比如列表或元组。如果是，并且序列中有多个元素，代码就会把序列中的最后一个元素（`local_tensors[-1]`）添加到 `ground_truth` 这个列表中。这个过程就像是把配菜中的某个特别重要的部分，比如配菜的调味料，单独挑出来放到盘子里（`ground_truth`）。

        ### 3. 处理字典类型的数据

        ```python
        elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
            ground_truth.append(local_tensors["labels"])
        ```

        如果 `local_tensors` 不是一个序列，而是一个字典（`Dict`），代码会检查字典里有没有“标签”（`labels`）这个关键字。如果有，它就会把这个标签添加到 `ground_truth` 中。这相当于从一盘混合的菜里找出其中的标签部分，并放到 `ground_truth` 盘子里。

        ### 4. 获取之前保存的输入数据

        ```python
        true_input_tensor = builder.user_data["p2p_tensors"]
        ```

        然后，代码从 `builder.user_data` 中取出之前保存的输入数据（`true_input_tensor`）。这个数据可能是之前的任务留下的“剩菜”，需要继续使用。

        ### 5. 检查输入数据的类型

        ```python
        if isinstance(true_input_tensor, Sequence):
            true_input_tensor = true_input_tensor[0]
        ```

        最后，代码检查这个 `true_input_tensor` 是不是一个序列。如果是，它就会取出序列中的第一个元素作为最终的输入数据。这一步相当于从一碗“剩菜”里挑出最主要的一块食材，准备继续用在接下来的菜肴制作中。

        ### 总结

        这一部分代码主要是处理那些已经存在的数据——可能是之前步骤的“剩余物”或者已经准备好的数据。它通过检查数据的类型和内容，决定如何将它们添加到接下来的处理流程中。这确保了即使在非初始阶段，程序也能正确地处理和利用现有的数据，继续完成任务。

    6. **最终准备工作：**

    ```python
    return (true_input_tensor,), {}, ground_truth
    ```

    最后，厨师将所有准备好的食材整理好，返回主要的食材（`true_input_tensor`），一个空字典（因为暂时没有额外的配料），以及已经放到盘子里的配菜（`ground_truth`）。

    ### 总结

    这个 `prepare_data` 函数就像是厨师在厨房里准备食材的过程。它会根据当前的任务是新任务还是已进行过的任务，来决定如何从冰箱里取出食材、调整准备，并最终整理出一个准备好烹饪的食材列表。这些食材随后会被送到锅里（在函数外的部分）进行烹饪，最终呈现出一道美味的菜肴。
    """
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    mbx = inst.minibatch
    cur_model = builder.model[chunk_id]

    user_data = builder.user_data
    forward_data_store = user_data["forward_data_store"]
    input_tensors = user_data["input_tensors"]
    output_tensors = user_data["output_tensors"]

    constant_data = builder.constant_data
    autocast_dtype = constant_data["autocast_dtype"]
    enable_autocast = constant_data["enable_autocast"]

    is_pp_first_stage = stage_id == 0 and chunk_id == 0
    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    # forward step
    if is_pp_first_stage:
        if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]):
            input_tensors[chunk_id].append(None)

    # find corresponding input tensor
    input_tensor = None
    for cur_item in input_tensors[chunk_id]:
        if cur_item is not None and cur_item[1] == mbx:
            input_tensor = cur_item[0]

    if not is_pp_first_stage:
        assert input_tensor is not None

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:

        def prepare_data():
            model_chunk_id = builder.user_data["model_chunk_id"]
            ground_truth = []
            if builder.user_data["is_pp_first_stage"]:
                true_input_tensor = next(builder.dataloader[model_chunk_id])
                # keep the input tensor in builder
                if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]) + 1:
                    true_input_tensor.requires_grad_()
                    builder.user_data["input_tensors"][chunk_id].pop()
                    builder.user_data["input_tensors"][chunk_id].append((true_input_tensor, mbx))
            else:
                local_tensors = next(builder.dataloader[model_chunk_id])
                if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
                    ground_truth.append(local_tensors[-1])
                elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
                    ground_truth.append(local_tensors["labels"])
                true_input_tensor = builder.user_data["p2p_tensors"]
                if isinstance(true_input_tensor, Sequence):
                    true_input_tensor = true_input_tensor[0]

            return (true_input_tensor,), {}, ground_truth

        builder.user_data["model_chunk_id"] = chunk_id
        builder.user_data["p2p_tensors"] = input_tensor
        builder.user_data["is_pp_first_stage"] = is_pp_first_stage
        builder.user_data["is_pp_last_stage"] = is_pp_last_stage
        builder.user_data["prepare_data_fn"] = prepare_data
        args, kwargs, ground_truth = builder.user_data["prepare_data_fn"]()
        builder.user_data["ground_truth"] = ground_truth
        output_tensor = cur_model(*args, **kwargs)

    if is_pp_last_stage:
        output_tensor, loss_tensor = registed_functions["vescale_zbv_loss_fn"](output_tensor)
        forward_data_store.append((output_tensor, loss_tensor))
        output_tensor = output_tensor if builder.loss_fn is None else loss_tensor

    if stage_id + 1 == builder.constant_data["total_stages"] and chunk_id == 0:
        # turn around the forward direction
        builder.user_data["input_tensor"] = (output_tensor, mbx)
        builder.user_data["input_tensors"][chunk_id + 1].append((output_tensor, mbx))

    builder.user_data["output_tensors"][chunk_id].append(output_tensor)
    user_data["output_tensor"] = output_tensor


# backward


@register_instruction(name="vescale_zbv_backward_b")
def vescale_zbv_backward_b():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    grad_scaler = builder.constant_data["grad_scaler"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]

    input_tensors = builder.user_data["input_tensors"]
    output_tensors = builder.user_data["output_tensors"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]

    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    if is_pp_last_stage:
        if len(output_tensor_grads[chunk_id]) == 0:
            output_tensor_grads[chunk_id].append(None)
    input_tensor = input_tensors[chunk_id].pop(0)[0]
    output_tensor = output_tensors[chunk_id][0]
    output_tensor_grad = output_tensor_grads[chunk_id][0]

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # extract loss value from output tensors
    if isinstance(output_tensor[0], Sequence):
        for j in range(len(output_tensor[0])):
            if output_tensor[0][j].ndim == 0 and output_tensor[0][j].numel() == 1:
                loss_value = output_tensor[0][j]
                break
        else:
            loss_value = output_tensor[0][-1]
    else:
        loss_value = output_tensor[0]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        loss_value = grad_scaler(loss_value)
    # FIXME: For virtual pipeline, there may exist frozen layer without grad;
    # Need to verify if this solution is correct
    if not loss_value.requires_grad:
        return None

    if deallocate_pipeline_outputs:
        assert 0
        # custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        input_tensor_grad = switch_dtensor(torch.autograd.grad)(
            loss_value,
            input_tensor,
            grad_outputs=output_tensor_grad[0],
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )[0]

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    def f(input_tensor):
        if input_tensor is not None:
            assert isinstance(input_tensor, (torch.Tensor, DTensor)), input_tensor
            input_tensor.grad = None

        nonlocal output_tensor

        if not isinstance(output_tensor, Sequence):
            output_tensor = [output_tensor]

        if (output_tensor is None) or (not deallocate_pipeline_outputs):
            return
        assert isinstance(
            output_tensor, [torch.Tensor, DTensor]
        ), f"expected Tensor, found {type(output_tensor).__name__}."
        assert output_tensor._base is None, "counter-productive to free a view of another tensor."
        if isinstance(output_tensor, [torch.Tensor, DTensor]):
            output_tensor._local_tensor.data = torch.empty(
                (1,),
                device=output_tensor.device,
                dtype=output_tensor.dtype,
            )
        else:
            output_tensor.data = torch.empty(
                (1,),
                device=output_tensor.device,
                dtype=output_tensor.dtype,
            )
        return

    if not isinstance(input_tensor, Sequence):
        map(f, [input_tensor])
    else:
        map(f, input_tensor)

    if stage_id + 1 == builder.constant_data["total_stages"] and chunk_id == 1:
        # turn around the forward direction
        builder.user_data["output_tensor_grad"] = input_tensor_grad
        builder.user_data["output_tensor_grads"][chunk_id - 1].append(output_tensor_grad)

    builder.user_data["input_tensor_grad"] = input_tensor_grad


@register_instruction(name="vescale_zbv_backward_w")
def vescale_zbv_backward_w():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    cur_model = builder.model[chunk_id]
    grad_scaler = builder.constant_data["grad_scaler"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]

    output_tensors = builder.user_data["output_tensors"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]

    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    if is_pp_last_stage:
        if len(output_tensor_grads[chunk_id]) == 0:
            output_tensor_grads[chunk_id].append(None)
    output_tensor = output_tensors[chunk_id].pop(0)
    output_tensor_grad = output_tensor_grads[chunk_id].pop(0)

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor = grad_scaler(output_tensor[0])
    # FIXME: For virtual pipeline, there may exist frozen layer without grad;
    # Need to verify if this solution is correct
    if not output_tensor[0].requires_grad:
        return None

    # Gather params
    nps = {}
    for key, value in cur_model.named_parameters():
        nps[key] = value

    if deallocate_pipeline_outputs:
        assert 0
    else:
        params_grad = switch_dtensor(torch.autograd.grad)(
            output_tensor[0],
            nps.values(),
            grad_outputs=output_tensor_grad[0],
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

    # Manually set each params grad
    for param, grad in zip(nps.values(), params_grad):
        param.grad = grad


# validation


@register_instruction(name="vescale_zbv_post_validation")
def vescale_zbv_post_validation():
    pass


@register_instruction(name="vescale_zbv_recv_post_validation")
def vescale_zbv_recv_post_validation():
    pass


@register_instruction(name="vescale_zbv_send_post_validation")
def vescale_zbv_send_post_validation():
    pass


# loss


@register_instruction(name="vescale_zbv_loss_fn")
def vescale_zbv_loss_fn(output_tensor):
    loss_func = builder.loss_fn
    if loss_func is None:
        return output_tensor, None
    temp_tensor = output_tensor
    args_spec = signature(loss_func)
    args_len = len(args_spec.parameters.keys())
    if args_len == 1:
        output_tensor = loss_func(output_tensor)
    else:
        ground_truth = builder.user_data["ground_truth"]
        loss_fn_inputs = [output_tensor] + ground_truth
        output_tensor = loss_func(*loss_fn_inputs)
        assert args_len == len(loss_fn_inputs), "Mismatch of loss function #args and #actual inputs!"
    builder.user_data["output_tensor"] = output_tensor
    return temp_tensor, output_tensor


VESCALE_INSTRUCTION_MAPPING_ZBV = {
    "RECV_FORWARD": "vescale_zbv_recv_forward",
    "SEND_FORWARD": "vescale_zbv_send_forward",
    "F": "vescale_zbv_forward",
    "B": "vescale_zbv_backward_b",
    "W": "vescale_zbv_backward_w",
    "RECV_BACKWARD": "vescale_zbv_recv_backward",
    "SEND_BACKWARD": "vescale_zbv_send_backward",
    "RECV_POST_VALIDATION": "vescale_zbv_recv_post_validation",
    "SEND_POST_VALIDATION": "vescale_zbv_send_post_validation",
    "POST_VALIDATION": "vescale_zbv_post_validation",
}

if __name__ == "__main__":
    settings = [
        # p,   n,     f,     b,     w,   c,    h,  a,  l
        # (8, 24, 18522, 18086, 9337, 601, 2304, 24, 24),
        # (8, 32, 18513, 18086, 9331, 626, 2304, 24, 24),
        # (8, 64, 18546, 18097, 9321, 762, 2304, 24, 24),
        # (8, 24, 29718, 29444, 19927, 527, 4096, 32, 32),
        # (8, 32, 29802, 29428, 19530, 577, 4096, 32, 32),
        # (8, 64, 29935, 29621, 19388, 535, 4096, 32, 32),
        # (16, 48, 11347, 11248, 8132, 377, 5120, 40, 48),
        # (16, 64, 11307, 11254, 8101, 379, 5120, 40, 48),
        # (16, 128, 11325, 11308, 8109, 378, 5120, 40, 48),
        # (32, 96, 10419, 10207, 7715, 408, 6144, 48, 64),
        # (32, 128, 10408, 10204, 7703, 408, 6144, 48, 64),
        # (32, 256, 10402, 10248, 7698, 460, 6144, 48, 64),
        (4, 8, 6, 4, 4, 1, 4096, 32, 32),
        # (8, 24, 29444, 29718, 19927, 527, 4096, 32, 32),
        # ( 8, 32, 16099, 16504,  7589,  540, 2304, 24, 16),
        # (16, 48, 14407, 14380, 9676, 1610, 4096, 32, 32),
        # (16, 64, 14412, 14393, 9688, 1621, 4096, 32, 32),
        # (16, 128, 14316, 14306, 9639, 1619, 4096, 32, 32),
        # (24, 72, 6763, 6969, 5251, 755, 5120, 40, 48),
        # (24, 96, 6783, 6984, 5259, 758, 5120, 40, 48),
        # (24, 192, 6785, 6990, 5260, 770, 5120, 40, 48),
        # (32, 96, 9458, 9748, 7288, 879, 6144, 48, 64),
        # (32, 128, 9469, 9744, 7306, 892, 6144, 48, 64),
        # (32, 256, 9447, 9644, 7193, 887, 6144, 48, 64),
    ]
    s = 1024

    # h, a, s = 4096, 32, 1024
    # cost_f, cost_b, cost_w, cost_c = 29718, 29444, 19927, 527
    for p, n, f, b, w, c, h, a, _ in settings:
        mem_f = 34 * h + 5 * a * s
        mem_w = -32 * h
        mem_b = -mem_w - mem_f
        for m_offset in range(p + 1):
            graph = CostGraph(
                n_stage=p,
                n_micro=n,
                f_cost=f,
                b_cost=b,
                w_cost=w,
                c_cost=c,
                f_mem=mem_f,
                b_mem=mem_b,
                w_mem=mem_w,
                max_mem=mem_f * (p * 2 + m_offset),
            )
            graph.get_v_schedule()
            break
