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

import enum
from dataclasses import dataclass
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Sequence, Callable
import torch
from torch.distributed.distributed_c10d import get_rank
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Placement
from vescale.pipe.pipe_stage import PipeModule
from typing import List, Tuple, Union, Optional, Dict, Any
import logging
import functools
import numpy as np
from optree import tree_map
from vescale.dtensor.dtensor import DTensor
from vescale.plan.spec import PipelineP2PSpec

Shape = Union[List[int], torch.Size]

logger = logging.getLogger(__name__)
registed_functions = {}


def switch_dtensor(func: Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        def to_tensor(x):
            if isinstance(x, DTensor):
                return x.to_local()
            return x

        new_args = tree_map(to_tensor, args)
        new_kwargs = tree_map(to_tensor, kwargs)
        out = func(*new_args, **new_kwargs)
        return out

    return wrap


def register_instruction(name):
    assert name is not None, "The Instruction must have name"
    if name in registed_functions:
        msg = f"{name} allready in registed instruction"
        logger.warning(msg)

    def _register_instruction(func):
        def wrap(*args, **kwargs):
            return func(*args, **kwargs)

        registed_functions.update({name: func})
        return wrap

    return _register_instruction


@dataclass
class CommPacket:
    cur_mesh: DeviceMesh
    peer_mesh: DeviceMesh
    input_id: int
    peer_stage: int
    peer_sharding: List[Placement] = None
    cur_sharding: List[Placement] = None
    is_kwargs: bool = False


class StageDeps:
    """
    要理解这个需要掌握什么数据结构?
    ==========================

    To understand the `StageDeps` class and the code surrounding it, there are several key data structures and concepts you should be familiar with. These data structures are essential for handling the complexities of distributed systems, especially in the context of pipeline parallelism and distributed training.

    ### Key Data Structures and Concepts

    1. **Graphs (Directed Acyclic Graphs - DAGs):**
    - **Graph:** A collection of nodes (or vertices) connected by edges. In this context, each node represents a stage in the model, and the edges represent dependencies between stages.
    - **Directed Graph:** A graph where the edges have a direction, meaning they go from one node to another in a specific order. This is important for determining the sequence in which stages must be processed.
    - **Acyclic Graph:** A graph with no cycles, meaning there’s no way to start at one node and follow the edges back to the same node.

    2. **Adjacency Matrix:**
    - This is a 2D array or matrix used to represent a graph. In this context, `dep` is an adjacency matrix where `dep[i][j] = 1` means there is a directed edge from node `i` to node `j`. This matrix is crucial for defining the dependencies between the different stages of the model.

    3. **Dynamic Programming:**
    - Some of the methods, like `get_pre_stage()` and `get_post_stage()`, involve searching and processing stages based on their dependencies. Understanding dynamic programming helps in efficiently handling such processes where results from previous computations are reused.

    4. **Communication Graphs:**
    - **CommPacket:** This data structure is used to encapsulate information about communication between stages. It involves sending and receiving data across different stages, often on different devices.
    - The communication graph is built to manage how data flows between different stages (nodes) of the pipeline and across different devices (meshes).

    5. **Heaps or Priority Queues (for sorting tasks):**
    - In methods like `construct_communication_graph`, where the `prior_list` is sorted, understanding heaps or priority queues helps in grasping how stages are prioritized and processed.

    6. **Distributed Systems:**
    - Familiarity with distributed systems concepts is essential. This includes understanding how data and computations are distributed across multiple devices, how communication occurs between these devices, and how consistency is maintained.

    7. **Pipeline Parallelism:**
    - **Pipeline Parallelism:** This is a specific type of parallelism where different stages of a computation pipeline are processed in parallel across different devices. Understanding how a pipeline is constructed, executed, and how data flows through it is crucial.
    - The `StageDeps` class is heavily involved in managing pipeline parallelism, making sure that data dependencies are respected, and communication between stages is correctly handled.

    8. **Hash Tables (Dictionaries):**
    - Dictionaries (`Dict` in Python) are used throughout the code for mapping and storing communication packets, stages, and other data structures. Understanding how hash tables work will help in comprehending how data is efficiently stored and retrieved.

    9. **Queue and Stack Data Structures:**
    - These data structures are fundamental in managing the order of operations, especially when working with directed graphs where processing stages in the correct sequence is crucial.

    ### Summary

    To fully understand and work with the `StageDeps` class and the related code, you should be comfortable with the following data structures:

    - **Graphs and Adjacency Matrices** for representing dependencies.
    - **Dynamic Programming** for optimizing dependency resolution.
    - **Communication Graphs** for managing inter-device communication.
    - **Heaps or Priority Queues** for sorting and prioritizing tasks.
    - **Distributed Systems Concepts** for understanding the broader context of how these data structures are applied in parallel computing environments.
    - **Pipeline Parallelism** for specific parallelism strategies used in distributed training.

    These structures and concepts form the backbone of how distributed training pipelines are managed and executed in a system like the one you're working with.


    给初中生讲解这些函数，我会尽量用简单易懂的语言，并使用日常生活中的例子来帮助理解。
    ===================================================================

    ### 1. `__init__` 函数
    ```python
    def __init__(self, dep, meshes, vpp_module_list, p2p_index_mapping=None):
    ```
    - **作用:** 这是一个特殊的函数，当你创建一个 `StageDeps` 对象时，它会自动执行，用来初始化一些数据。
    - **例子:** 想象你要开一家店，店里需要准备货架（`dep`）、店员（`meshes`）和商品列表（`vpp_module_list`）。`__init__` 就是你开店前把这些东西都准备好。

    ### 2. `construct_communication_graph` 函数
    ```python
    def construct_communication_graph(self):
    ```
    - **作用:** 这个函数负责建立一个“通讯图”，决定每个阶段（店员）应该和谁沟通，怎么沟通。
    - **例子:** 在店里工作时，店员需要知道他们要和谁合作，比如谁负责上货、谁负责收银。`construct_communication_graph` 就是在安排这些合作方式。

    ### 3. `generate_one_forward_mapping` 函数
    ```python
    def generate_one_forward_mapping(self):
    ```
    - **作用:** 这个函数创建了一种简单的工作顺序，让店员们知道应该先做什么，再做什么。
    - **例子:** 想象你有一组店员，他们按照顺序工作：先有人上货，然后有人整理货架，最后有人打扫卫生。`generate_one_forward_mapping` 就是为这些工作排出一个顺序。

    ### 4. `parsing_forward_mapping` 函数
    ```python
    def parsing_forward_mapping(self):
    ```
    - **作用:** 这个函数检查已有的工作顺序，并确保它们是合理的。如果有多条分支的工作（像多人同时进行不同的任务），它会处理这些情况。
    - **例子:** 如果你有两位店员同时整理不同的货架，`parsing_forward_mapping` 会确保他们不会发生冲突，分配好他们的任务。

    ### 5. `get_send_comms` 函数
    ```python
    def get_send_comms(self, i):
    ```
    - **作用:** 这个函数返回某个阶段（店员）需要发送的所有消息或任务。
    - **例子:** 如果一个店员需要把货物从仓库搬到货架上，`get_send_comms` 就会告诉他要搬什么货物，搬到哪儿。

    ### 6. `get_recv_comms` 函数
    ```python
    def get_recv_comms(self, i):
    ```
    - **作用:** 这个函数返回某个阶段（店员）需要接收的所有消息或任务。
    - **例子:** 想象一个店员站在收银台，他需要知道顾客递来的钱是多少。`get_recv_comms` 就是告诉店员他会收到什么样的信息或任务。

    ### 7. `get_local_comms` 函数
    ```python
    def get_local_comms(self, i):
    ```
    - **作用:** 这个函数返回某个阶段（店员）在内部需要处理的任务，而不是和其他人之间的沟通。
    - **例子:** 一个店员在整理货架的时候，不需要和其他店员沟通，`get_local_comms` 就是告诉他在自己岗位上需要做的事情。

    ### 8. `num_stage` 函数
    ```python
    @property
    def num_stage(self):
    ```
    - **作用:** 这个函数返回总共有多少个阶段（店员）在工作。
    - **例子:** 如果你的店里有4个店员在工作，`num_stage` 就会告诉你店员的数量是4个。

    ### 9. `is_first` 函数
    ```python
    def is_first(self, s_id):
    ```
    - **作用:** 这个函数检查某个阶段（店员）是不是第一个开始工作的人。
    - **例子:** 在一组店员中，`is_first` 会告诉你谁是第一个上班的店员。

    ### 10. `is_last` 函数
    ```python
    def is_last(self, s_id):
    ```
    - **作用:** 这个函数检查某个阶段（店员）是不是最后一个完成工作的。
    - **例子:** 如果你的店里有一个店员负责关店门，`is_last` 就会告诉你他是不是最后一个离开的人。

    ### 11. `get_pre_stage` 函数
    ```python
    def get_pre_stage(self, i, ignore_virtual=True):
    ```
    - **作用:** 这个函数返回在某个阶段之前的所有阶段（店员），也就是他需要依赖的工作。
    - **例子:** 如果店员B需要等店员A整理完货架才能上货，`get_pre_stage` 会告诉你店员A是B之前的阶段。

    ### 12. `get_post_stage` 函数
    ```python
    def get_post_stage(self, i, ignore_virtual=True):
    ```
    - **作用:** 这个函数返回在某个阶段之后的所有阶段，也就是接下来要做的工作。
    - **例子:** 如果店员A整理完货架后，店员B接着上货，`get_post_stage` 会告诉你B是A之后的阶段。

    ### 13. `get_first_stage` 函数
    ```python
    def get_first_stage(self):
    ```
    - **作用:** 这个函数返回所有不依赖其他阶段的第一个阶段。
    - **例子:** 如果你的店里有几组独立的工作，`get_first_stage` 会告诉你哪些店员是各组的第一个开始工作的人。

    ### 14. `get_last_stage` 函数
    ```python
    def get_last_stage(self):
    ```
    - **作用:** 这个函数返回所有最后完成工作的阶段。
    - **例子:** `get_last_stage` 会告诉你谁是最后一个完成工作的店员。

    ### 15. `get_current_model` 函数
    ```python
    def get_current_model(self, i):
    ```
    - **作用:** 这个函数返回当前阶段正在处理的模块或任务。
    - **例子:** 想象你有很多货物需要整理，`get_current_model` 会告诉你当前店员正在整理哪一类货物。

    ### 16. `is_pipeline_first_stage` 函数
    ```python
    def is_pipeline_first_stage(self, i):
    ```
    - **作用:** 这个函数检查某个阶段是否是流水线中的第一个阶段。
    - **例子:** 如果你的店里有一条生产线，`is_pipeline_first_stage` 会告诉你哪个阶段是生产线的第一个步骤。

    ### 17. `is_pipeline_last_stage` 函数
    ```python
    def is_pipeline_last_stage(self, i):
    ```
    - **作用:** 这个函数检查某个阶段是否是流水线中的最后一个阶段。
    - **例子:** 如果你的店里有一条生产线，`is_pipeline_last_stage` 会告诉你哪个阶段是生产线的最后一个步骤。

    ### 18. `is_vpp_first_stage` 函数
    ```python
    def is_vpp_first_stage(self, i, chunk_id):
    ```
    - **作用:** 这个函数检查某个阶段在特定的分块中是否是第一个阶段。
    - **例子:** 想象你有一批货分成几部分运到店里，`is_vpp_first_stage` 会告诉你某个店员是否是这部分货的第一个处理者。

    ### 19. `is_vpp_last_stage` 函数
    ```python
    def is_vpp_last_stage(self, i, chunk_id):
    ```
    - **作用:** 这个函数检查某个阶段在特定的分块中是否是最后一个阶段。
    - **例子:** `is_vpp_last_stage` 会告诉你某个店员是否是这部分货的最后一个处理者。

    ### 20. `get_num_chunks` 函数
    ```python
    def get_num_chunks(self):
    ```
    - **作用:** 这个函数返回当前工作被分成了多少块。
    - **例子:** 如果你的货物被分成了几批运到店里，`get_num_chunks` 会告诉你总共有多少批货物。

    ### 21. `get_current_mesh` 函数
    ```python
    def get_current_mesh(self, i):
    ```
    - **作用:** 这个函数返回当前阶段在哪个设备（或者说工作区域）上运行。
    - **例子:** 想象你的店里有几个不同的工作区域，`get_current_mesh` 会告诉你某个店员在哪个区域工作。

    ### 22. `__str__` 函数
    ```python
    def __str__(self):
    ```
    - **作用:** 这个函数把整个 `StageDeps` 对象的状态转换成一个字符串，方便打印出来看。
    - **例子:** 这就像是给你店里所有工作情况写一份报告，里面详细列出了每个店员在做什么，以及他们是怎么合作的。

    这些函数共同作用，使得 `StageDeps` 类能够有效地管理复杂的分布式计算任务，确保各个阶段


    给我你比喻的例子和代码对应表格
    =========================

    以下是将代码函数与比喻例子对应的表格，帮助你更直观地理解每个函数的作用：

    | **函数名**                          | **代码作用**                                                                 | **比喻例子**                                                           |
    |-----------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------|
    | `__init__`                        | 初始化数据结构，准备阶段依赖关系、设备网格等。                                    | 开店前准备货架、店员和商品列表。                                                |
    | `construct_communication_graph`   | 建立阶段之间的通信图，决定每个阶段如何沟通。                                      | 安排店员如何合作，比如谁负责上货，谁负责收银。                                         |
    | `generate_one_forward_mapping`    | 创建一个简单的工作顺序。                                                        | 排出店员的工作顺序，比如先上货、再整理、最后打扫。                                        |
    | `parsing_forward_mapping`         | 检查并处理已有的工作顺序，特别是多条分支的情况。                                       | 确保两位店员同时整理不同货架时不会发生冲突。                                            |
    | `get_send_comms`                  | 返回某个阶段需要发送的消息或任务。                                                | 告诉店员他需要从仓库搬哪些货物到货架。                                                 |
    | `get_recv_comms`                  | 返回某个阶段需要接收的消息或任务。                                                | 告诉收银员他会收到顾客递来的钱是多少。                                                  |
    | `get_local_comms`                 | 返回某个阶段内部需要处理的任务。                                                  | 告诉店员他在整理货架时需要做的事情。                                                   |
    | `num_stage`                       | 返回总共有多少个阶段。                                                          | 告诉你店里有多少个店员在工作。                                                      |
    | `is_first`                        | 检查某个阶段是否是第一个开始工作的人。                                              | 告诉你谁是第一个上班的店员。                                                       |
    | `is_last`                         | 检查某个阶段是否是最后一个完成工作的。                                              | 告诉你谁是最后一个完成工作的店员。                                                   |
    | `get_pre_stage`                   | 返回在某个阶段之前的所有阶段。                                                   | 告诉你哪个店员需要在另一个店员完成任务后才能开始工作。                                         |
    | `get_post_stage`                  | 返回在某个阶段之后的所有阶段。                                                   | 告诉你哪个店员需要在另一个店员完成任务后接着工作。                                         |
    | `get_first_stage`                 | 返回所有第一个开始工作的阶段。                                                    | 告诉你哪些店员是各组的第一个开始工作的人。                                              |
    | `get_last_stage`                  | 返回所有最后完成工作的阶段。                                                    | 告诉你哪些店员是最后一个完成工作的人。                                                 |
    | `get_current_model`               | 返回当前阶段正在处理的模块或任务。                                                | 告诉你当前店员正在整理哪一类货物。                                                   |
    | `is_pipeline_first_stage`         | 检查某个阶段是否是流水线中的第一个阶段。                                             | 告诉你生产线的第一个步骤是哪一个。                                                   |
    | `is_pipeline_last_stage`          | 检查某个阶段是否是流水线中的最后一个阶段。                                             | 告诉你生产线的最后一个步骤是哪一个。                                                  |
    | `is_vpp_first_stage`              | 检查某个阶段在特定的分块中是否是第一个阶段。                                           | 告诉你店员是否是某批货物的第一个处理者。                                                |
    | `is_vpp_last_stage`               | 检查某个阶段在特定的分块中是否是最后一个阶段。                                           | 告诉你店员是否是某批货物的最后一个处理者。                                              |
    | `get_num_chunks`                  | 返回当前工作被分成了多少块。                                                      | 告诉你总共有多少批货物运到店里。                                                    |
    | `get_current_mesh`                | 返回当前阶段在哪个设备或区域上运行。                                               | 告诉你某个店员在哪个区域工作。                                                     |
    | `__str__`                         | 把整个 `StageDeps` 对象的状态转换成字符串，方便查看。                                    | 给你店里所有工作情况写一份详细报告。                                                 |

    这个表格通过将每个函数的实际代码作用与比喻相对应，使得理解这些复杂的概念更加直观和易于理解。

    """
    def __init__(
        self,
        dep: np.ndarray,
        meshes: List[DeviceMesh],
        vpp_module_list: Union[List, PipeModule],
        p2p_index_mapping: Optional[Dict[int, List[PipelineP2PSpec]]] = None,
    ):
        """
        vpp_module_list 中的 vpp 是什么?
        ==============================


        在 `vpp_module_list` 中的 `vpp` 可能指的是 **Virtual Pipeline Parallelism**（虚拟流水线并行），这是在分布式计算和模型并行化中的一种技术。

        ### 虚拟流水线并行 (VPP)
        虚拟流水线并行是一种将大型神经网络模型分成多个子模块，并在不同的设备（例如多个GPU）上并行执行的技术。通过这种方式，可以有效地利用硬件资源，减少计算的等待时间，同时提高整体计算的效率。

        - **Virtual（虚拟）**: 这里指的是将一个大任务分成多个虚拟的子任务，每个子任务可以独立执行。
        - **Pipeline（流水线）**: 指的是这些子任务之间的依赖关系，这些任务按顺序依次执行，每个任务完成后，其结果会传递给下一个任务。
        - **Parallelism（并行）**: 指的是多个子任务可以在不同的硬件设备上同时执行，以提高计算效率。

        在 `vpp_module_list` 中，这些 `vpp` 模块表示的是流水线并行中的各个子模块。每个模块可能代表神经网络的一部分，这些部分在执行时会在不同的设备上进行分布式计算，以实现并行处理。

        因此，`vpp_module_list` 可以被理解为一个列表，里面包含了所有需要进行流水线并行处理的模型模块。这些模块将被分配到不同的设备上，并按照预定义的顺序和依赖关系进行计算，从而实现高效的并行训练。











        要讲解这段代码给初中生理解，我们可以将它与现实生活中的情境进行比喻。
        =======================================================

        我会一步步解释每一行代码的意义，就像是在描述一个团队如何合作完成任务。









        ### 1. 初始化（`__init__` 函数）
        首先，这段代码是在创建一个新的“任务管理者”对象，这个对象负责组织和安排不同的任务，确保团队中的每个人都知道该做什么，什么时候做。

        ### 2. `dep: np.ndarray`
        ```python
        self.D = dep
        ```
        - **比喻:** 这就像是任务清单（`dep`），里面写着谁需要先做什么，再做什么。任务管理者需要这份清单来安排大家的工作顺序。

        ### 3. `meshes: List[DeviceMesh]`
        ```python
        self.meshes = meshes
        ```
        - **比喻:** 这是团队成员的列表（`meshes`），每个人都有自己擅长的工作领域。任务管理者需要知道团队中有哪些成员，以及他们各自的特长。
            ```py
            meshes
            [DeviceMesh:([0]), DeviceMesh:([1]), DeviceMesh:([2]), DeviceMesh:([3])]
            ```

        ### 4. `vpp_module_list: Union[List, PipeModule]`
        ```python
        self.M = vpp_module_list
        ```
        - 存储需要并行化处理的模型模块列表。项目的不同步骤或部分，例如一个模型的各层，这些部分将在不同的团队成员（设备）之间分布并行执行。
            ```py
            vpp_module_list
            [MLP(...), MLP(...)]
            ```

        ### 5. `is_vpp`
        ```python
        self.is_vpp = self.get_num_chunks() > 1
        ```
        - **比喻:** 任务管理者检查一下手头的任务是否被分成了多个部分（`chunks`），如果是的话，他需要特别注意，因为这意味着任务更复杂，需要更多的协调。

        ### 6. `mapping: Dict`
        ```python
        self.mapping: Dict = {}
        ```
        - **比喻:** 这是一个空的地图（`mapping`），任务管理者将用它来记录任务之间的关系，比如谁需要先做，谁需要接着做。

        ### 7. `p2p_index_mapping`
        ```python
        if p2p_index_mapping is None:
            self.mapping = defaultdict(list)
            self.generate_one_forward_mapping()
        else:
            self.mapping = p2p_index_mapping
            self.parsing_forward_mapping()
        ```
        - **比喻:** 如果没有现成的工作关系图（`p2p_index_mapping`），任务管理者就自己生成一个简单的顺序表（`generate_one_forward_mapping`）；如果有现成的图，他就使用这个图来安排任务（`parsing_forward_mapping`）。

        ### 8. 接收、发送和本地数据表格
        ```python
        self.recv_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.send_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.local_dataloader_list: Dict[Any, List[CommPacket]] = defaultdict(list)
        ```
        - **比喻:** 任务管理者准备了三个表格：
        1. **接收表格（`recv_tables`）:** 记录每个团队成员需要从别人那里接收什么任务或信息。
        2. **发送表格（`send_tables`）:** 记录每个团队成员需要把什么任务或信息发送给别人。
        3. **本地表格（`local_dataloader_list`）:** 记录每个团队成员自己需要完成的任务，不涉及和别人交换信息。

        ### 9. `construct_communication_graph`
        ```python
        self.construct_communication_graph()
        ```
        - **比喻:** 最后，任务管理者使用以上所有信息来创建一个“通讯图”（`communication_graph`），确保每个人都知道该向谁发送或接收信息，以及自己独立完成哪些任务。

        ### 总结

        这段代码就是在模拟一个任务管理者如何组织团队，安排任务，确保每个成员都知道该做什么，如何与其他人合作。通过设置接收表、发送表和本地任务表，并生成或使用任务关系图，任务管理者可以有效地协调团队中的每个人，使整个团队能够顺利完成任务。


        以下是将代码函数与比喻相对应的表格，帮助你理解每一行代码的作用及其对应的现实生活比喻：
        =======================================================================

        | **代码片段**                                                                 | **代码作用**                                                          | **比喻例子**                                                                 |
        |-----------------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------|
        | `self.D = dep`                                                              | 存储任务的依赖关系矩阵，用于决定任务的执行顺序。                                    | 任务清单，记录谁需要先做什么，再做什么。                                                |
        | `self.meshes = meshes`                                                      | 存储设备网格列表，表示不同的团队成员或计算资源。                                 | 团队成员的列表，记录每个人的特长和工作领域。                                               |
        | `self.M = vpp_module_list`                                                  | 存储模块列表或资源，用于完成任务。                                             | 手头的资源或工具列表，比如锤子、钉子，用于完成特定任务。                                        |
        | `self.is_vpp = self.get_num_chunks() > 1`                                   | 判断任务是否被分成多个部分，设置标志变量。                                      | 检查任务是否复杂，是否需要特别的协调。                                                    |
        | `self.mapping: Dict = {}`                                                   | 初始化一个空的映射表，用于记录任务之间的关系。                                     | 一个空的地图，用于记录任务之间的依赖关系，比如谁先做，谁接着做。                                      |
        | `if p2p_index_mapping is None: ...`                                         | 根据是否有现成的任务关系图，决定是生成一个简单的顺序表还是使用已有的关系图。                    | 如果没有现成的工作关系图，就自己生成一个顺序表；如果有现成的图，就使用它来安排任务。                            |
        | `self.recv_tables: Dict[int, List[CommPacket]] = defaultdict(list)`         | 初始化接收表格，记录每个阶段需要从别人接收的任务或信息。                                 | 接收表格，记录每个团队成员需要从别人那里接收的任务或信息。                                         |
        | `self.send_tables: Dict[int, List[CommPacket]] = defaultdict(list)`         | 初始化发送表格，记录每个阶段需要发送给别人的任务或信息。                                 | 发送表格，记录每个团队成员需要把任务或信息发送给谁。                                             |
        | `self.local_dataloader_list: Dict[Any, List[CommPacket]] = defaultdict(list)`| 初始化本地任务表格，记录每个阶段自己需要完成的任务。                                   | 本地任务表，记录每个团队成员自己需要完成的任务，不涉及与别人交换信息。                                   |
        | `self.construct_communication_graph()`                                      | 构建通信图，确定各阶段的通信和数据交换方式。                                        | 创建“通讯图”，确保每个成员都知道该向谁发送或接收信息，以及自己独立完成哪些任务。                         |

        这个表格通过将每个代码片段的实际功能与生活中的比喻相对应，使得理解这些复杂的编程概念变得更加简单和直观。

        """
        self.D = dep
        self.M = vpp_module_list
        self.meshes = meshes
        self.is_vpp = self.get_num_chunks() > 1
        self.mapping: Dict = {}
        if p2p_index_mapping is None:
            self.mapping = defaultdict(list)
            self.generate_one_forward_mapping()
        else:
            self.mapping = p2p_index_mapping
            self.parsing_forward_mapping()

        self.recv_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.send_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.local_dataloader_list: Dict[Any, List[CommPacket]] = defaultdict(list)
        self.construct_communication_graph()

    def construct_communication_graph(self):
        for i in range(self.num_stage):
            cur_mesh = self.get_current_mesh(i)
            cur_mapping = self.mapping[i]  # get the index mapping i
            prior_list = []
            local_data_list = []
            # stage_id: [input_idx, ...]
            for p2p_spec in cur_mapping:
                prev_stage_id = p2p_spec.peer_stage_idx
                input_id = p2p_spec.peer_output_idx
                if prev_stage_id != i:  # not from self
                    prior_list.append((self.get_current_mesh(prev_stage_id), prev_stage_id, input_id))
                else:  # from self stage
                    local_data_list.append(input_id)

            prior_list = sorted(prior_list, key=lambda item: (item[1], item[2]))
            for device, pre, input_id in prior_list:
                sr = CommPacket(
                    cur_mesh=cur_mesh, peer_mesh=device, input_id=input_id, peer_stage=pre
                )  # input is single
                self.recv_tables[i].append(sr)
            for input_id in local_data_list:
                sr = CommPacket(
                    cur_mesh=cur_mesh,
                    peer_mesh=None,
                    input_id=input_id,
                    peer_stage=None,
                )
                self.local_dataloader_list[i].append(sr)

        # construct out degree
        for i in range(self.num_stage):
            prior_list = []
            for j in range(self.num_stage):
                if i == j:  # don't check self , no cycle
                    continue
                j_recvs = self.recv_tables[j]
                for recv in j_recvs:
                    if recv.peer_stage == i:  # is i send to j
                        send = CommPacket(
                            cur_mesh=recv.peer_mesh,
                            peer_mesh=recv.cur_mesh,
                            input_id=recv.input_id,
                            peer_stage=j,
                        )
                        prior_list.append(send)
            # sort by input_id stage id is unneeded
            sorted(prior_list, key=lambda item: item.input_id)
            self.send_tables[i] = prior_list

    def generate_one_forward_mapping(self):
        """
        要向初中生解释这段代码，并结合数据结构的概念，我会用一个简单的例子和直观的比喻来说明每一步的作用。
        ================================================================================

        让我们结合 `PipelineP2PSpec` 这个数据结构来更好地理解 `generate_one_forward_mapping` 函数的作用。

        ### 什么是 `PipelineP2PSpec`？

        `PipelineP2PSpec` 是一个数据类，用于描述两个阶段之间的通信方式。它有两个主要属性：

        - **`peer_stage_idx`**: 表示这个阶段需要从哪个阶段接收数据，换句话说，就是它依赖的前一个阶段的索引。
        - **`peer_output_idx`**: 通常为0，表示来自前一个阶段的输出。

        这就像是在定义每个任务需要从哪个之前的任务获取信息或者资源。

        ### 将 `PipelineP2PSpec` 和 `generate_one_forward_mapping` 结合起来解释

        ```python
        def generate_one_forward_mapping(self):
            for i in range(self.num_stage):
                cur_mapping = self.mapping[i]
                pre_stages = self.get_pre_stage(i, ignore_virtual=False)
                assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
                for pre in pre_stages:
                    cur_mapping.append(PipelineP2PSpec(pre, 0))

                if self.is_pipeline_first_stage(i):
                    cur_mapping.append(PipelineP2PSpec(i, 0))
        ```

        ### 代码逐行解释与比喻

        #### 1. `PipelineP2PSpec` 数据结构
        ```python
        @dataclass
        class PipelineP2PSpec:
            peer_stage_idx: int
            peer_output_idx: int = 0
        ```
        - **作用**：`PipelineP2PSpec` 用来表示一个阶段（任务）从另一个阶段（任务）接收数据的方式。
        - **比喻**：想象你在写一本小说，每一章的内容可能会引用前一章的情节发展。`peer_stage_idx` 就是指你引用的前一章的编号，而 `peer_output_idx` 则是你引用的具体内容（通常是上一章的结尾）。

        #### 2. `generate_one_forward_mapping` 的作用与 `PipelineP2PSpec` 的结合
        ```python
        def generate_one_forward_mapping(self):
            for i in range(self.num_stage):
                cur_mapping = self.mapping[i]
        ```
        - **作用**：我们遍历每个阶段，准备为每个阶段生成一个任务清单（`cur_mapping`）。
        - **比喻**：想象你要为每一章小说准备一个草稿，这个草稿会列出它引用的前一章以及其他要引用的内容。

        ```python
                pre_stages = self.get_pre_stage(i, ignore_virtual=False)
                assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
        ```
        - **作用**：这里，我们获取当前阶段依赖的前一个阶段，并确保每个阶段最多只依赖一个前面的阶段。
        - **比喻**：在写小说时，你确保每一章最多只引用前面的一章，这样故事的情节不会混乱。

        ```python
                for pre in pre_stages:
                    cur_mapping.append(PipelineP2PSpec(pre, 0))
        ```
        - **作用**：如果当前阶段依赖一个前面的阶段，我们就创建一个 `PipelineP2PSpec` 对象，并把它加到当前阶段的任务清单中。
        - **比喻**：如果这一章引用了前一章的内容，你就在草稿中标记出来：“这一章的情节基于前一章的结尾。”

        ```python
                if self.is_pipeline_first_stage(i):
                    cur_mapping.append(PipelineP2PSpec(i, 0))
        ```
        - **作用**：如果当前阶段是流水线中的第一个阶段（即没有依赖），我们也把它加到任务清单中。
        - **比喻**：如果这是小说的第一章，它不需要引用前面的内容，你就直接开始写它的草稿。

        ### 总结

        结合 `PipelineP2PSpec` 这个数据结构，我们可以更好地理解 `generate_one_forward_mapping` 函数的作用。这个函数的主要任务是为每个阶段生成一个“任务草稿”，这些草稿描述了每个阶段需要引用哪些前面的内容，以及如何引用。通过 `PipelineP2PSpec`，每个阶段都知道该如何从前面的阶段获取信息，并在自己的任务中使用这些信息。这就像在写一本连贯的小说，每一章都清楚地知道它与前一章的联系，这样整个故事才能连贯流畅地进行。
        """
        for i in range(self.num_stage):
            cur_mapping = self.mapping[i]
            pre_stages = self.get_pre_stage(i, ignore_virtual=False)
            assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
            for pre in pre_stages:
                cur_mapping.append(PipelineP2PSpec(pre, 0))

            if self.is_pipeline_first_stage(i):
                cur_mapping.append(PipelineP2PSpec(i, 0))

    def parsing_forward_mapping(self):
        # 1: [(0,0), (1,0), (0,2)]
        for i in range(self.num_stage):
            if i not in self.mapping:
                cur_indexing = []
                pre_stages = self.get_pre_stage(i, ignore_virtual=False)
                assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
                for pre in pre_stages:
                    cur_indexing.append(PipelineP2PSpec(pre, 0))
                if self.is_pipeline_first_stage(i):
                    cur_indexing.append(PipelineP2PSpec(i, 0))
                self.mapping.update({i: cur_indexing})

    def get_send_comms(self, i):
        return self.send_tables[i]

    def get_recv_comms(self, i):
        return self.recv_tables[i]

    def get_local_comms(self, i):
        return self.local_dataloader_list[i]

    @property
    def num_stage(self):
        return len(self.D)

    def is_first(self, s_id):
        pre = self.D[:, s_id]
        non_zero = np.count_nonzero(pre)
        if non_zero == 0:
            return True
        return False

    def is_last(self, s_id):
        post = self.D[s_id]
        non_zero = np.count_nonzero(post)
        if non_zero == 0:
            return True
        return False

    def get_pre_stage(self, i, ignore_virtual=True):
        pre = self.D[:, i]
        stage_ids = np.where(pre == 1)[0].tolist()
        if self.is_first(i) and self.is_vpp and not ignore_virtual:
            last_stages = list(filter(self.is_last, range(self.num_stage)))
            return last_stages
        else:
            return stage_ids

    def get_post_stage(self, i, ignore_virtual=True):
        post = self.D[i]
        stage_ids = np.where(post == 1)[0].tolist()

        if self.is_last(i) and self.is_vpp and not ignore_virtual:
            first_stages = list(filter(self.is_first, range(self.num_stage)))
            return first_stages
        else:
            return stage_ids

    def get_first_stage(self):
        stages = []
        for i in range(self.num_stage):
            pre_stages = self.get_pre_stage(i)
            if len(pre_stages) == 0:  # in-degree is 0
                stages.append(i)
        return stages

    def get_last_stage(self):
        stages = []
        for i in range(self.num_stage):
            post_stages = self.get_post_stage(i)
            if len(post_stages) == 0:  # out-degree is 0
                stages.append(i)
        return stages

    def get_current_model(self, i):
        return self.M

    def is_pipeline_first_stage(self, i):
        pre = self.get_pre_stage(i)
        return len(pre) == 0  # first stage has no input

    def is_pipeline_last_stage(self, i):
        post = self.get_post_stage(i)
        return len(post) == 0  # last stage has no output

    def is_vpp_first_stage(self, i, chunk_id):
        return self.is_pipeline_first_stage(i) and chunk_id == 0

    def is_vpp_last_stage(self, i, chunk_id):
        return self.is_pipeline_last_stage(i) and (chunk_id == (self.get_num_chunks() - 1))

    def get_num_chunks(self):
        if isinstance(self.M, list):
            return len(self.M)
        else:
            return self.M.virtual_chunks

    def get_current_mesh(self, i):
        return self.meshes[i]

    def __str__(self):
        tmp = "\n\n"
        tmp += f"stages: {self.num_stage}, deps:{self.D}\n"
        for i in range(self.num_stage):
            tmp += f"\n===================stage:{i} start=======================\n"
            tmp += "recv : \n"
            for comm in self.recv_tables[i]:
                tmp += f"\t\t recv from {comm.peer_stage} with input:{comm.input_id} comm:{comm}\n"
            tmp += "send : \n"
            for comm in self.send_tables[i]:
                tmp += f"\t\t send to {comm.peer_stage} with  input:{comm.input_id} comm:{comm}\n"
            tmp += "local_dataloader_list : \n"
            for comm in self.local_dataloader_list[i]:
                tmp += f"\t\t local_dataloader with  input:{comm.input_id} comm:{comm}\n"

            tmp += f"===================stage:{i} end=======================\n\n"
        return tmp


def get_linear_pp_module_dep2(module_list: List, device_mesh_list: List[DeviceMesh]):
    """
    code context: test/parallel/pipeline/instruction/test_schedule.py
    ============

    ```py
    # initialize global device mesh
    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type="cuda",
        mesh_shape=(4, 1, 1),
        mesh_dim_names=("PP", "DP", "TP"),
    )
    global local_rank
    local_rank = self.rank
    device = f"cuda:{local_rank}"
    # must do this: https://pytorch.org/docs/stable/distributed.html
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
    all_batches_out = []
    if self.rank == 0:
        ### 获得单机下的 ground truth
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

    ### 获得分布式调度的损失值
    fwd_plan = {
        ".input": [[Replicate()]],
        ".output": [[Replicate()]],
    }
    model_list = []

    if self.rank == 0:
        model_list = [model.mlps[0], model.mlps[7]]
    elif self.rank == 1:
        model_list = [model.mlps[1], model.mlps[6]]
    elif self.rank == 2:
        model_list = [model.mlps[2], model.mlps[5]]
    elif self.rank == 3:
        model_list = [model.mlps[3], model.mlps[4]]

    print(f"model:\n{model}")
    for i in range(8):
        print(f"model.mlps[{i}]:\n{model.mlps[i]}")

    deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())
    data_iterator = []
    for i in range(batches):
        data = torch.zeros(1, 1, n_hidden) + i
        data_iterator.append(data.float().cuda())

    w = n_hidden * 2 * 4
    a = n_hidden * 4
    mem_f = 2 * w + 2 * a  # forward weight size
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

    ### 验证分布式调度的损失值与单机计算结果的一致性
    if self.rank == 0:
        loss_per_microbatch = [item[1] for item in all_forward]
        print(loss_per_microbatch, all_batches_out)
        for t1, t2 in zip(loss_per_microbatch, all_batches_out):
            self.assertEqual(t1, t2)

    # Synchronize before finishing
    torch.distributed.barrier()

    flush()
    wait()
    ```

    Explain
    =======

    Let's go through the code and focus on explaining the relevant parts, especially the `dep[i][i + 1] = 1` line and the context in which it's used.

    ### Context of the Code

    This code is designed for distributed training using a pipeline parallelism setup. The main idea is to split a model across multiple devices (like GPUs) so that different parts of the model (referred to as "stages") can be processed simultaneously on different devices, speeding up the training process.

    ### Breakdown of Key Concepts

    1. **Device Mesh:**
    - The device mesh is a grid or array of devices (GPUs in this case) that will be used to distribute the computation. The mesh defines how the devices are organized and how data will be passed between them.
    - In this code, the mesh is initialized with a shape of `(4, 1, 1)`, meaning there are 4 devices aligned in a single dimension (pipeline parallelism).

    2. **Model Stages:**
    - The model is broken down into multiple smaller parts or "stages," each of which will be assigned to a different GPU. This is done in the `model_list`, where different layers (MLPs) of the model are allocated to different ranks (devices).

    3. **Direct Graph (`dep[i][i + 1] = 1`):**
    - This line is part of a function that sets up dependencies between different stages of the model during training.
    - **`dep`** is a matrix (a 2D array) where each row and column represents a stage of the model.
    - **`dep[i][i + 1] = 1`** means that stage `i` must be completed before stage `i + 1` can begin. This is similar to a "direct graph" where each node (stage) is connected to the next, forming a chain or sequence.

    4. **Pipeline Execution:**
    - The code sets up a `ScheduleEngine` to handle the execution of the model across the different devices. This engine uses the dependency matrix (`dep`) to ensure that stages are executed in the correct order, following the dependencies defined by the direct graph.
    - The pipeline parallelism strategy here ensures that while one stage is being executed on one GPU, the next stage can start on another GPU, leading to more efficient use of resources.

    ### Explaining `dep[i][i + 1] = 1` in Detail

    Let's visualize this with an example:

    - Imagine you have 4 stages of a model, named Stage 0, Stage 1, Stage 2, and Stage 3.
    - These stages need to be executed in order, where Stage 0's output is used as the input for Stage 1, Stage 1's output for Stage 2, and so on.

    The dependency matrix `dep` initially looks like this (with all zeros):

    ```
    0 0 0 0
    0 0 0 0
    0 0 0 0
    0 0 0 0
    ```

    When the code `dep[i][i + 1] = 1` runs, it updates the matrix to show that each stage directly leads to the next:

    ```
    0 1 0 0
    0 0 1 0
    0 0 0 1
    0 0 0 0
    ```

    - **Row 0, Column 1 (`dep[0][1]`)**: Stage 0 must finish before Stage 1 can start.
    - **Row 1, Column 2 (`dep[1][2]`)**: Stage 1 must finish before Stage 2 can start.
    - **Row 2, Column 3 (`dep[2][3]`)**: Stage 2 must finish before Stage 3 can start.

    This matrix helps the `ScheduleEngine` understand the correct order of operations during the distributed training process, ensuring that the model is trained correctly across all GPUs.

    ### Conclusion

    In summary, the line `dep[i][i + 1] = 1` is crucial in setting up the dependencies between different stages of a model in a pipeline parallelism setup. It ensures that the model's stages are processed in the correct order across multiple GPUs, making the training process efficient and correct. The entire setup is part of a larger system to validate that the distributed training results match those from a single-machine setup, which is a critical aspect of ensuring consistency in large-scale machine learning models.

    """
    stage_len = len(device_mesh_list)  # for forward
    dep = np.zeros((stage_len, stage_len), dtype=np.int64)
    for i in range(stage_len - 1):
        dep[i][i + 1] = 1  # direct graph
    return StageDeps(dep, device_mesh_list, module_list)


@dataclass
class Status:
    batch_idx: int = 0
    stage_id: int = 0
    chunk_id: int = 0
    f_b: "str" = ""  # forward or backward
    stg: "str" = ""  # stage for 1f1b
    k: int = 0

    def __str__(self):
        return f"b:{self.batch_idx}, c:{self.chunk_id}, {self.stg + '-' + self.f_b}"


class PipelineSchema(metaclass=ABCMeta):
    """
    we define this class to abstract the pipeline execute
    Args:
        dep: the dependency for adjacency martrix
        meshes: the list for stage of

    """

    def __init__(self, num_stage: int, meshes: Union[List[DeviceMesh], int], batches: int = 1):
        self.num_stage = num_stage
        self.meshes = meshes
        self.batches = batches
        self._schedules: List[List[Tuple]] = self._gen_schedule()

    @property
    @abstractmethod
    def name(self):
        """print schedule name"""
        raise NotImplementedError()

    @abstractmethod
    def _gen_schedule(self):
        """generator the pipelinne schedule for engine"""
        raise NotImplementedError("not impl")

    def __str__(self):
        """print the pipeline clock work"""
        stream = "\n"
        d = " ".join([f"d{d:<24}" for d in range(self.num_mesh)])
        stream += f"T k :{d:<24} \n"
        for time, scheds in enumerate(self.schedules):
            sched_str = " ".join([f"{str(sched):<24}" for sched in scheds])
            stream += f"T {time:<2}: {sched_str} \n"
        return stream

    @property
    def schedules(self):
        """return schedules"""
        return self._schedules

    @property
    def num_mesh(self):
        """return the num mesh of tp group"""
        if isinstance(self.meshes, Sequence):
            return len(self.meshes)
        elif isinstance(self.meshes, int):
            return self.meshes
        else:
            raise NotImplementedError("unsupport device mesh list")

    @property
    def num_clock(self):
        """return num schedule for the num clock"""

        return len(self._schedules)


@dataclass
class BaseInstruction(metaclass=ABCMeta):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError("unsupport run command")

    @property
    def name(self):
        return "base_instruction"

    def dump(self):
        return f"{get_rank()}: {self}"


class InstructionGenerator(metaclass=ABCMeta):
    def __init__(
        self,
        deps: StageDeps,
        meshes: int,
        batches: int,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
        batch_shape_lists: Optional[List[Any]] = None,
        batch_dtype_lists: Optional[List[Any]] = None,
        forward_only=False,
        num_chunk=1,
    ):
        self.deps = deps
        self.meshes = meshes
        self.num_chunk = num_chunk
        self.batches = batches
        self.default_shape = default_shape
        self.default_dtype = default_dtype
        self.batch_shape_lists = batch_shape_lists
        self.batch_dtype_lists = batch_dtype_lists
        self.forward_only = forward_only
        self.instruction_list: List = []

    """
    generate instruction
    """

    @abstractmethod
    def gen_instruction(self):
        raise NotImplementedError("not implement")

    """
    get current stage instruction
    """

    def get_instruction_list(self, stage: int):
        return self.instruction_list[stage]

    """
        update with batch idx, stage idx
    """

    def _set_inst(self, inst: BaseInstruction, s: int):
        self.instruction_list[s].append(inst)

    """
        set instruction type
    """

    def execute(self, *args, **kwargs):
        raise NotImplementedError("not implement")


class InstructionBuilder:
    global_instructions_funcs = defaultdict(list)
    global_instructions_str = defaultdict(list)

    constant_data = defaultdict()
    user_data = defaultdict()
    loss_fn: Callable = torch.sum
    dataloader: Any
    topo: StageDeps
    model: Callable
    stage_id: int
    _pos = 0
    _stack = None

    def build_from_dict(self, instructions: Dict):
        assert isinstance(instructions, dict), "instructions should be dict"
        for stage_id, instruction_list in instructions.items():
            cur_stage_ins_list = instruction_list
            if isinstance(cur_stage_ins_list, str):
                instructions_funcs = cur_stage_ins_list.split(",")
            else:
                instructions_funcs = cur_stage_ins_list

            mapped_functions = [registed_functions[x] for x in instructions_funcs]

            self.global_instructions_funcs[stage_id] = mapped_functions
            self.global_instructions_str[stage_id] = instructions_funcs

    def draw_instructions(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        # draw rectangle
        stage_nums = len(self.global_instructions_str.keys())
        for stage_id, instuctions_strs in self.global_instructions_str.items():
            for id, stage_str in enumerate(instuctions_strs):
                ax.add_patch(plt.Rectangle((id, -1 * stage_id), 1, 1, fill=False, edgecolor="black", lw=2))
                ax.text(id + 0.5, -1 * stage_id + 0.5, stage_str, ha="center", va="center")

        for stage_id in range(stage_nums):
            ax.text(-0.5, -1 * stage_id + 0.5, stage_id, ha="center", va="center")
        # set max xlim and ylim
        max_stages = max(len(x) for x in self.global_instructions_str.values())
        ax.set_xlim(0, max_stages)
        ax.set_ylim(-1 * stage_nums + 1, 1)
        ax.axis("off")
        plt.savefig("instructions.png")

    @property
    def pos(self):
        return self._pos

    @property
    def last(self):
        return self._stack

    def run(self, stage_id: int):
        output = []
        for pos, fn in enumerate(self.global_instructions_funcs[stage_id]):
            self._pos = pos
            out = fn()
            self._stack = out
            output.append(out)
        return output

    def export(self, stage_id, *args, **kwargs):
        func_lists = self.global_instructions_funcs[stage_id]

        class Model(torch.nn.Module):
            def __init__(self, func_lists, model):
                super().__init__()
                self.func_lists = func_lists
                self.model = model

            def forward(self, *args, **kwargs):
                for f in self.func_lists:
                    # TODO: handle this to make forward inst work.
                    if f.__name__ == "forward":
                        activation = self.model(*args, **kwargs)
                        args = (activation,)
                    else:
                        args, kwargs = f(*args, **kwargs)
                return args, kwargs

        model = Model(func_lists, self.model)
        graph = torch.export.export(model, args)
        return graph


class CompilePPCollectiveKind(enum.Enum):
    SEND = 1
    RECV = 2
    BORADCAST = 3  # for cross mesh collective
    UNKNOWN = 4


class CompilePPCollectiveOperator:
    def __init__(
        self,
        kind: CompilePPCollectiveKind,
        src: int = None,
        dst: List[int] = None,
        is_backward: bool = False,
    ) -> None:
        assert kind in (
            CompilePPCollectiveKind.BORADCAST,
            CompilePPCollectiveKind.SEND,
            CompilePPCollectiveKind.RECV,
        )
        self.kind = kind
        self.is_backward = is_backward

        if self.kind is CompilePPCollectiveKind.SEND:
            assert dst is not None and isinstance(dst, int)
        elif self.kind is CompilePPCollectiveKind.RECV:
            assert src is not None and isinstance(src, int)
        else:
            assert src is not None and isinstance(src, int)
            assert dst is not None and isinstance(dst, List[int])
            assert src in dst

        self.src = src
        self.dst = dst
        pass

    def __hash__(self) -> int:
        if isinstance(self.dst, List[int]):
            dst = tuple(self.dst)
        else:
            dst = self.dst
        return hash((self.kind, self.src, dst, self.is_backward))


VESCALE_INTRUCTION_BUILDER = InstructionBuilder()
