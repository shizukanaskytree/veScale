################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import logging
import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import (
    ProcessGroup,
    _find_pg_by_ranks_and_tag,
    _get_default_group,
    _get_group_size,
    _get_group_tag,
    get_process_group_ranks,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
)

from vescale.debug import DebugLogger

logger = logging.getLogger(__name__)

# only import numpy typing when type checking
if TYPE_CHECKING:
    try:
        from numpy.typing import ArrayLike
    except ImportError:
        logger.warning("DeviceMesh requires numpy >= 1.21 to be installed for type checking")


class _MeshEnv:
    def __init__(self) -> None:
        self.mesh_stack: List[DeviceMesh] = []
        self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

    def get_current_mesh(self) -> "DeviceMesh":
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh is currently active!")
        return self.mesh_stack[-1]

    def create_child_mesh(self, device_mesh: "DeviceMesh", mesh_dim: int, mesh_dim_name: str) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        cur_rank = device_mesh.get_rank()
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))

        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_1d,
                mesh_dim_names=(mesh_dim_name,),
                _init_process_groups=False,
            )
            if cur_rank in mesh_1d:
                res_sub_mesh = sub_mesh

        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
        # Assign the current DeviceMesh as the parent of the child DeviceMesh.
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def create_submesh_along_multi_dims(
        self, device_mesh: "DeviceMesh", mesh_dims: List[int], cur_rank: int = None
    ) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        # check dims
        dim_size = [-1]
        for dim in mesh_dims:
            if dim >= device_mesh.ndim:
                raise RuntimeError("Mesh dim in sub groups out of range!")
            dim_size.append(device_mesh.mesh.size(dim))
        mesh_tensor = device_mesh.mesh
        for dim in mesh_dims:
            mesh_tensor = mesh_tensor.swapdims(-1, dim)
        if cur_rank is None:
            cur_rank = device_mesh.get_rank()
        pg_ranks_by_dims = mesh_tensor.reshape(dim_size)
        for mesh_nd in pg_ranks_by_dims:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_nd,
                _init_process_groups=False,
            )
            if cur_rank in mesh_nd:
                res_sub_mesh = sub_mesh
        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[dim] for dim in mesh_dims]
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def create_submesh_group(self, device_mesh: "DeviceMesh", mesh_dim: int) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        # check dims
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))
        res = []
        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_1d,
                _init_process_groups=False,
            )
            sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
            # Assign the current DeviceMesh as the parent of the child DeviceMesh.
            self.child_to_parent_mapping[sub_mesh] = device_mesh
            res.append(sub_mesh)
        return res

    def get_parent_mesh(self, device_mesh: "DeviceMesh") -> Optional["DeviceMesh"]:
        return self.child_to_parent_mapping.get(device_mesh, None)

    def get_parent_mesh_dim(self, device_mesh: "DeviceMesh") -> Optional[int]:
        """
        Return the index of the mesh dim in the parent mesh.
        The device_mesh passed in needs to be sliced out from a parent mesh.
        """
        parent_mesh = self.get_parent_mesh(device_mesh)
        child_mesh_dim_names = device_mesh.mesh_dim_names
        if parent_mesh and child_mesh_dim_names:
            assert len(child_mesh_dim_names) == 1, "The child mesh can only be a 1D mesh."
            child_mesh_dim_name = child_mesh_dim_names[0]
            if parent_mesh.mesh_dim_names:
                return parent_mesh.mesh_dim_names.index(child_mesh_dim_name)
        return None

    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        # ProcessGroup can't tell us this info so we have to infer it, assume
        # homogeneous hardware for now
        return get_world_size() // _MeshEnv.num_devices_per_host(device_type)


mesh_resources: _MeshEnv = _MeshEnv()


def _get_device_handle(device_type: str = "cuda"):
    """
    Get the module corresponding to the device_type which is cuda or cuda-like device.
    For example, when the device_type is cuda, the module `torch.cuda` is returned.
    Return None when there is no corresponding module for device_type, otherwise
    return the corresponding module.
    """
    return getattr(torch, device_type, None)


class DeviceMesh:
    """
    DeviceMesh represents a mesh of devices (given by `device_type`), where layout
    of devices could be represented as a n-d dimension array `mesh`, and each value
    of the `mesh` is the global rank in the default process group.

    DeviceMesh could be used to describe the layout of devices across the cluster
    via `mesh_dim_names`, and serves as a proxy for communication among the device lists
    within the cluster.

    By default (`pg` is `None`), we use the default ProcessGroup in this DeviceMesh class
    to implement proper communications. Note that we also add collective wrappers in this
    class. This is used to decouple detailed communication backend with the underlying
    DTensor implementation.

    By giving an existing ProcessGroup `pg`, we construct a device mesh from this `pg`,
    instead of the default ProcessGroup.

    Here are the expected behaviors:
    | `mesh` | `pg`  | result                               | catch
    ---------------------------------------------------------------------------------------------
    |  None  | None  | raise error!                         |
    |  EXIST | None  | use `mesh` + default ProcessGroup    |
    |  None  | EXIST | use `pg`'s ranks + `pg` ProcessGroup | 1D mesh only
    |  EXIST | EXIST | use `pg`'s ranks + `pg` ProcessGroup | `mesh` must equal to `pg`'s ranks

    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like, meta.
        mesh (ndarray): could be a multi-dimension array or an integer tensor that
            describes the layout of devices, the ids are global ids of the default process group.
        mesh_dim_names (Optional[Tuple[str]]): A tuple of mesh dim names to be assigned to each
            dimension of the multi-dimensional array that describes the layout of devices. Its
            length must match the length of `mesh_shape`. Each string in mesh_dim_names must be unique.
        pg (Optional[ProcessGroup]): the given ProcessGroup. See above for expected behaviors.

    Returns:
        A :class:`DeviceMesh` object

    Example (2 host with 4 GPUs each):
        ```
        # The following program runs on each process/rank in SPMD manner.
        # initialize device mesh as (2, 4) to represent the topology
        # of cross-host(dim 0), and within-host (dim 1)
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ])
        ```
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7)

    Note:
        DeviceMesh can be used as a context manager.


    `DeviceMesh` 代表设备的网格（由 `device_type` 指定），其中设备的布局可以表示为一个n维数组 `mesh`，并且 `mesh` 的每个值是默认进程组中的全局rank。

    `DeviceMesh` 可用于通过 `mesh_dim_names` 描述集群中设备的布局，并作为集群中设备列表之间通信的代理。

    默认情况下（当 `pg` 为 `None` 时），我们在这个 `DeviceMesh` 类中使用默认的 `ProcessGroup` 来实现通信。需要注意的是，我们在此类中添加了集体通信的封装器，以便将详细的通信后端与底层的 `DTensor` 实现解耦。

    如果提供了一个现有的 `ProcessGroup` `pg`，我们将从这个 `pg` 构建一个设备网格，而不是使用默认的 `ProcessGroup`。

    预期行为如下：

    - `mesh` 和 `pg` 都为空：抛出错误！
    - `mesh` 存在，`pg` 为空：使用 `mesh` 和默认 `ProcessGroup`
    - `mesh` 为空，`pg` 存在：使用 `pg` 的ranks和 `pg` 的 `ProcessGroup`（仅支持1D网格）
    - `mesh` 和 `pg` 都存在：使用 `pg` 的ranks和 `pg` 的 `ProcessGroup`，此时 `mesh` 必须与 `pg` 的ranks相等

    参数说明：
    - `device_type` (str): 网格的设备类型。目前支持：cpu, cuda/cuda-like, meta。
    - `mesh` (ndarray): 可以是多维数组或整数张量，描述设备的布局，这些id是默认进程组的全局id。
    - `mesh_dim_names` (Optional[Tuple[str]]): 一个元组，每个维度的名称将被分配给描述设备布局的多维数组的每个维度。它的长度必须与 `mesh_shape` 的长度匹配。`mesh_dim_names` 中的每个字符串必须是唯一的。
    - `pg` (Optional[ProcessGroup]): 指定的 `ProcessGroup`。详见预期行为。

    返回值：
    - 一个 `DeviceMesh` 对象。

    示例（2台主机，每台有4个GPU）：

    ```python
    # 以下程序在每个进程/rank中以SPMD方式运行。
    # 初始化设备网格为(2, 4)，以表示跨主机（第0维）和主机内（第1维）的拓扑结构。
    mesh = DeviceMesh(device_type="cuda",
                    mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                    ])
    ```

    在 `mesh` 的第一维度上进行归约将跨列(0, 4), (1, 5), (2, 6), (3, 7)进行归约，而在 `mesh` 的第二维度上进行归约将跨行(0, 1, 2, 3)和(4, 5, 6, 7)进行归约。

    注意：
    `DeviceMesh` 可以作为上下文管理器使用。
    """

    device_type: str
    mesh: Optional[Union[torch.Tensor, "ArrayLike"]]
    mesh_dim_names: Optional[Tuple[str, ...]]

    def __init__(
        self,
        device_type: str,
        mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        pg: Optional[ProcessGroup] = None,
        _validate_mesh: bool = True,
        _init_process_groups: bool = True,
    ) -> None:
        """
        device_type
        'cuda'

        mesh
        tensor([[0, 1],
                [2, 3]])

        mesh_dim_names
        ['DP', 'TP']

        pg
        None

        _validate_mesh
        True

        _init_process_groups
        True
        """
        # for performance, update debug env once here
        DebugLogger.update_vescale_debug_mode_from_env()
        # check args
        if mesh is None and pg is None:
            raise ValueError("Either `mesh` or `pg` must be provided!")
        if mesh is not None and pg is not None:
            pg_mesh_tensor = torch.tensor(get_process_group_ranks(pg), dtype=torch.int, device="cpu")
            mesh_tensor = (
                mesh.detach().cpu()
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, dtype=torch.int, device="cpu")
            )
            if not torch.equal(mesh_tensor, pg_mesh_tensor):
                raise ValueError(f"mesh({mesh_tensor}) and pg({pg_mesh_tensor}) must have the same content!")
        if pg is not None:
            self.mesh = torch.tensor(get_process_group_ranks(pg), dtype=torch.int, device="cpu")
            warnings.warn("Construction from given ProcessGroup is only supported for 1D mesh currently.")
            # TO FIX: use `mesh` to reshape `pg_mesh_tensor` for nD mesh tensor
        if mesh is not None:
            self.mesh = (
                mesh.detach().cpu()
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, dtype=torch.int, device="cpu")
            )

        self.device_type = device_type
        self.mesh_dim_names = mesh_dim_names

        # private field to pre-generate DeviceMesh's hash
        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape))

        # step 1: try to create default world pg.
        if pg is None:
            pg = self._get_or_create_default_group()
        else:
            # TODO: this logic only applies when device_type is cuda
            pg_world_size = get_world_size(group=pg)
            device_handle = _get_device_handle(self.device_type)
            num_devices_per_host = device_handle.device_count()
            if pg_world_size > num_devices_per_host and pg_world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{pg_world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
                )
        if self.device_type == "cuda":

            def _get_current_device():
                try:
                    if torch.cuda.is_available():
                        return torch.cuda.current_device()
                    else:
                        return None
                except AssertionError as e:
                    return None

            device_handle = _get_device_handle(self.device_type)
            num_devices_per_host = device_handle.device_count()
            local_rank = get_rank() % num_devices_per_host
            if local_rank != _get_current_device():
                warnings.warn("Remember to set cuda device id to local rank!!!")
                device_handle = _get_device_handle(self.device_type)
                device_handle.set_device(local_rank)

        # step 2: validate the mesh before following usage.
        if _validate_mesh:
            self._validate_mesh(pg)

        # step 3: get coordinate of current global rank on the mesh.
        # The world pg is used for device mesh identity (rank) on each
        # process (we need to know if the current global rank is in the mesh or not)
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None

        # step 4: init multi subprocess group for the mesh object.
        if _init_process_groups:
            self._init_process_groups(pg)

    def _get_or_create_default_group(self):
        default_initialized = is_initialized()
        if not default_initialized:
            """
            init_process_group 的结果是初始化一个进程组。这个函数通常用于分布式计算环境中，确保所有参与计算的进程能够相互通信。具体来说，
            它会设置一个默认的进程组，使得后续的分布式操作可以在这个进程组中进行。

            在代码中，init_process_group 被调用时会执行以下操作：

            1. 初始化默认进程组，使得所有进程可以相互通信。
            2. 设置进程组的世界大小（即参与计算的总进程数）。
            3. 确保每个进程都有一个唯一的 rank（进程编号）。

            在 _get_or_create_default_group 方法中，如果 is_initialized() 返回 False，则会调用 init_process_group() 来初始化默认进程组。
            这样可以确保在使用分布式计算功能之前，进程组已经被正确初始化。

            总结来说，init_process_group 的结果是创建并初始化一个默认的进程组，使得分布式计算环境中的进程可以相互通信。
            """
            init_process_group()

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
            )

        """
        device_handle 在代码中的作用是管理和设置设备（例如 GPU）的相关操作。它主要用于确保在分布式计算环境中，每个进程都能正确地选择和使用设备。
        以下是 device_handle 的具体作用：

        获取设备句柄：通过 _get_device_handle(self.device_type) 获取当前设备类型（例如 CUDA）的句柄。
        获取设备数量：使用 device_handle.device_count() 获取每个主机上可用的设备数量。
        设置设备：使用 device_handle.set_device(get_rank() % num_devices_per_host) 设置当前进程使用的设备。

        作用总结
            设备句柄获取：通过 _get_device_handle 获取设备句柄。
            设备数量获取：通过 device_handle.device_count() 获取每个主机上的设备数量。
            设备设置：通过 device_handle.set_device 设置当前进程使用的设备，确保在分布式环境中，每个进程都能正确地使用设备。

        代码逻辑
            获取设备句柄 device_handle。
            如果设备句柄存在且进程组未初始化，获取每个主机上的设备数量 num_devices_per_host。
            检查世界大小 world_size 是否与设备数量兼容。
            设置当前进程使用的设备。

        通过这些操作，device_handle 确保了在分布式计算环境中，每个进程都能正确地选择和使用设备，从而实现高效的计算资源管理。
        """
        device_handle = _get_device_handle(self.device_type)
        # TODO: if user want to pass pg_options, offer a way to do it
        if not default_initialized and device_handle:
            # automatically set the current cuda/cuda-like device base on num of gpu devices available in each host
            # NOTE: This device selection would only work for homogeneous hardware.
            num_devices_per_host = device_handle.device_count()
            if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
                )
            device_handle.set_device(get_rank() % num_devices_per_host)


        """
        _get_default_group 函数用于获取由 init_process_group 创建的默认进程组。如果默认进程组尚未初始化，该函数会引发一个 ValueError 异常。
        """
        return _get_default_group()

    def _validate_mesh(self, pg: ProcessGroup):
        """
        _validate_mesh 方法的作用是验证 DeviceMesh 对象中的 mesh 属性是否符合特定的要求和约束。具体来说，它执行以下几项验证：

        1. 唯一性验证：检查 mesh 中的值是否唯一。如果 mesh 中存在重复值，则抛出 RuntimeError 异常。
        2. 大小验证：检查 mesh 的大小是否超过进程组的大小。如果 mesh 的元素数量大于进程组的大小，则抛出 RuntimeError 异常。
        3. 一致性验证：确保所有调用的进程传入的 mesh 参数一致。通过将 mesh 转换为指定设备类型并进行全局收集，然后逐一比较每个进程的 mesh，如果发现不一致，则抛出 RuntimeError 异常。

        这些验证确保了 DeviceMesh 的正确性和一致性，避免在分布式计算中出现不一致或错误的配置。
        """
        # validate rank uniqueness in mesh tensor
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}")
        # validate size
        if self.mesh.numel() > _get_group_size(pg):
            raise RuntimeError(
                f"DeviceMesh should not be bigger than world (group) size, but found {self.mesh.numel()} and {_get_group_size(pg)}"
            )
        # validate that all calling ranks pass in the same `mesh` argument.
        self_mesh = self.mesh.to(self.device_type).contiguous()

        """
        在代码片段中，mesh_tensor 是通过 funcol.all_gather_tensor 函数从所有进程收集的 self_mesh 张量。它的作用如下：

        收集张量：funcol.all_gather_tensor 函数会从所有进程中收集 self_mesh 张量，并将它们拼接成一个大的张量 mesh_tensor。
        这个大张量包含了所有进程的 self_mesh 张量。

        验证一致性：mesh_tensor 被分割成多个子张量 mesh_tensor_chunked，每个子张量对应一个进程的 self_mesh。
        通过比较这些子张量，可以验证所有进程传入的 mesh 参数是否一致。
        """
        mesh_tensor = funcol.all_gather_tensor(self_mesh, gather_dim=0, group=pg)

        mesh_tensor_chunked = torch.chunk(mesh_tensor, _get_group_size(pg))
        # aten.equal not supported for meta device
        if self.device_type == "meta":
            return
        for other_rank, other_mesh in enumerate(mesh_tensor_chunked):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh argument:"
                    f"rank {get_rank()} has mesh {self_mesh} while rank {get_process_group_ranks(pg)[other_rank]}"
                    f"has mesh {other_mesh}!"
                )

    def _init_process_groups(self, pg: ProcessGroup):
        """
        _init_process_groups 是 DeviceMesh 类中的一个方法，用于初始化与每个网格维度相关的进程组。它的作用如下：

        0. 初始化维度组信息：创建一个空的 dim_group_infos 列表，用于存储每个维度的组标签和进程列表。

        1. 处理一维网格：如果网格是一维的且其元素数量与进程组大小相同，则直接将给定的进程组添加到 dim_group_infos 中。

        2. 处理多维网格：对于多维网格，方法会：

            遍历每个维度。
            交换当前维度和最后一个维度，然后重新调整形状以展平其他维度。
            为每个维度创建子进程组，并将这些组添加到 dim_group_infos 中。

        3. 错误处理：如果某个维度有多个进程组，或者当前进程不在子组中，会引发运行时错误。
        """

        """
        函数的目的, 函数的输入,输出,最小但不是一般性的例子:
        ===========================================
        **函数的目的：**
        `_init_process_groups` 函数的目的是初始化与每个网格维度相关的进程组，确保在分布式计算环境中，设备网格的每个维度只有一个进程组，从而避免资源冲突、确保计算一致性，并简化通信逻辑。

        **函数的输入：**
        - `pg`（类型：`ProcessGroup`）：输入的进程组对象，表示需要基于此进程组进行初始化的设备网格。

        **函数的输出：**
        - 无直接返回值。函数运行后，会将初始化后的维度组信息存储在对象的 `_dim_group_infos` 属性中。

        **最小但不是一般性的例子：**

        假设我们有一个一维网格，并且进程组大小与网格元素数量相同。此时，函数会直接将给定的进程组添加到维度组信息中。

        ```python
        class DeviceMesh:
            def __init__(self, mesh):
                self.mesh = mesh

            def get_rank(self):
                return 0  # 简化返回当前rank

            def _init_process_groups(self, pg: ProcessGroup):
                # 示例中的函数实现

        # 假设 mesh 是一个一维网格，且包含 4 个元素
        mesh = torch.tensor([0, 1, 2, 3])
        pg = ProcessGroup(4)  # 假设进程组包含 4 个进程

        device_mesh = DeviceMesh(mesh)
        device_mesh._init_process_groups(pg)

        # 输出应包含一个进程组，且该组对应 mesh 的所有元素
        print(device_mesh._dim_group_infos)
        ```

        在此例子中，假设网格 `mesh` 是一维且包含 4 个元素，进程组 `pg` 的大小与网格元素数量一致，
        函数会将进程组直接添加到维度组信息中，最终 `_dim_group_infos` 会包含一个进程组，且该组对应网格的所有元素。
        """

        """
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

        一个多少个进程组?
        ==============

        根据你提供的代码和上下文，在使用的分布式计算环境中，进程组的数量和结构可以由几个因素决定，主要包括设备网格的形状、并行策略以及任务划分。让我为你分析一下：

        ### 1. **设备网格的形状 (mesh shape)**
        你设置的设备网格形状为 `(4, 1, 1)`，这意味着设备网格在三个维度上有以下分布：
        - **第一个维度 (PP, Pipeline Parallelism)**: 4 个设备（4 个 GPU）
        - **第二个维度 (DP, Data Parallelism)**: 1 个设备
        - **第三个维度 (TP, Tensor Parallelism)**: 1 个设备

        ### 2. **进程组的数量**
        在这个网格形状下，不同维度的进程组数量计算如下：

        1. **全局进程组**:
        - 通常会有一个包含所有进程的全局进程组（`ptd:1`），该组用于整个设备网格的全局通信或同步。

        2. **每个维度的进程组**:
        - **第一个维度（4 个设备）**:
            - 由于这个维度有 4 个设备，因此可以划分为 4 个独立的进程组。每个进程组包含一个设备。这些进程组可能分别标记为 `ptd:2` 到 `ptd:5`。
        - **第二个维度（1 个设备）**:
            - 这个维度只有 1 个设备，因此不会进一步划分进程组，通常不会创建额外的进程组。
        - **第三个维度（1 个设备）**:
            - 这个维度同样只有 1 个设备，也不会创建额外的进程组。

        ### 3. **总的进程组数量**
        在这种配置下，可能的进程组数量可以估算如下：

        - **全局进程组**: 1 个（包含所有设备）
        - **第一个维度**: 4 个独立的进程组
        - **第二个维度**: 0 个（因为只有 1 个设备）
        - **第三个维度**: 0 个（因为只有 1 个设备）

        **总计**：可能会有 **5 个进程组**。

        ### 4. **实际观察的进程组数量**
        根据你提供的 `dim_group_infos` 输出，我们可以看到如下模式：
        - **全局进程组** `ptd:1`: 涵盖所有设备。
        - **其他进程组**: `ptd:2`, `ptd:3`, `ptd:4`, `ptd:5`, `ptd:6`, `ptd:7`, `ptd:8`, `ptd:9`，这些进程组涵盖了不同的设备组合，通常对应于不同维度或特定任务。

        这些进程组标签的数量和编号显示了在不同维度和任务划分下的进程组划分方式。

        ### **总结**
        在你的配置下，一个设备网格可能包含 **5 到 9 个进程组**，具体数量取决于设备网格的形状、维度划分和具体的并行策略。全局进程组负责全局通信，而其他进程组负责不同维度或任务中的通信。

        ------------------------------------------------------------------------

        在这些输出中，每个 `dim_group_infos` 列表包含了几个元组，每个元组都代表一个与特定维度相关的进程组及其包含的进程编号。以下是对每个 `dim_group_infos` 的详细介绍：

        ### 1. **`dim_group_infos: [('ptd:1', [0, 1, 2, 3]), ('ptd:3', [1]), ('ptd:7', [1])]`**
        - `('ptd:1', [0, 1, 2, 3])`：
            - 这个标签 `ptd:1` 代表一个全局进程组，包含 GPU `0, 1, 2, 3`，表示这个进程组覆盖了设备网格中的所有 GPU。通常用于协调所有 GPU 的全局通信或同步操作。
        - `('ptd:3', [1])`：
            - 标签 `ptd:3` 代表一个更小的进程组，仅包含 GPU `1`。这意味着在特定的维度或任务中，进程 `1` 被分配到一个单独的进程组，用于处理特定的任务。
        - `('ptd:7', [1])`：
            - 标签 `ptd:7` 也是一个只包含 GPU `1` 的进程组。与 `ptd:3` 类似，这可能是用于不同维度上的另一组计算任务。

        ### 2. **`dim_group_infos: [('ptd:1', [0, 1, 2, 3]), ('ptd:5', [3]), ('ptd:9', [3])]`**
        - `('ptd:1', [0, 1, 2, 3])`：
            - 同样，`ptd:1` 是一个全局进程组，包含所有 GPU，执行全局同步或通信任务。
        - `('ptd:5', [3])`：
            - 标签 `ptd:5` 代表一个包含 GPU `3` 的进程组。这表示在特定维度或任务中，GPU `3` 被单独分配到这个进程组，可能用于单独的计算任务。
        - `('ptd:9', [3])`：
            - 标签 `ptd:9` 也是一个仅包含 GPU `3` 的进程组，类似于 `ptd:5`，但用于不同的维度或任务。

        ### 3. **`dim_group_infos: [('ptd:1', [0, 1, 2, 3]), ('ptd:4', [2]), ('ptd:8', [2])]`**
        - `('ptd:1', [0, 1, 2, 3])`：
            - 再次，`ptd:1` 是全局进程组，包含所有 GPU。
        - `('ptd:4', [2])`：
            - 标签 `ptd:4` 代表一个仅包含 GPU `2` 的进程组，表示这个 GPU 在特定的维度或任务中被单独分配到这个组中。
        - `('ptd:8', [2])`：
            - 标签 `ptd:8` 是另一个只包含 GPU `2` 的进程组，可能用于不同的维度或任务。

        ### 4. **`dim_group_infos: [('ptd:1', [0, 1, 2, 3]), ('ptd:2', [0]), ('ptd:6', [0])]`**
        - `('ptd:1', [0, 1, 2, 3])`：
            - 这个全局进程组 `ptd:1` 包含所有 GPU。
        - `('ptd:2', [0])`：
            - 标签 `ptd:2` 代表一个仅包含 GPU `0` 的进程组，表示这个 GPU 在特定维度或任务中单独操作。
        - `('ptd:6', [0])`：
            - 标签 `ptd:6` 是另一个仅包含 GPU `0` 的进程组，用于不同的任务或维度。

        ### 5. **重复出现的模式：**
        - `dim_group_infos` 中的 `ptd:1` 标签经常出现，这个标签表示一个全局的进程组，覆盖了所有的 GPU。在分布式计算中，这样的全局组常常用于协调各个 GPU 之间的全局操作或同步任务。
        - `ptd:3`, `ptd:4`, `ptd:5` 等标签每次只包含一个 GPU。这表明在特定的维度或任务中，这些 GPU 被单独分配到不同的进程组进行计算。这种模式可能是为了更细粒度的并行计算或任务分配。

        ### **总结：**
        这些 `dim_group_infos` 列表展示了如何在多维设备网格中创建并管理进程组。全局进程组用于全局通信，而单一 GPU 的进程组用于特定的任务或维度。这种结构使得计算能够在不同的维度和任务之间高效地并行化，从而充分利用多 GPU 系统的计算能力。
        """

        # group tag/ranks associated with each mesh dimension, each mesh dimension should
        # have one sub-group per rank
        dim_group_infos: List[Tuple[str, List[int]]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == _get_group_size(pg):
            # if the mesh is the same as the given group, we just append the given
            # pg to the first dim groups.
            dim_group_infos.append((_get_group_tag(pg), get_process_group_ranks(pg)))
        else:
            # create sub pgs base on the mesh argument specified
            ### 这段代码的主要功能是基于给定的网格（mesh）参数创建子进程组（subgroups）。下面是逐步解释
            ### self.mesh.ndim 表示网格的维度数。这个循环会遍历每一个维度。
            for dim in range(self.mesh.ndim):
                # swap the current dim to the last dim
                # then reshape to flatten out other dims

                ### 交换当前维度和最后一个维度，然后重塑网格
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                ### swapdims(-1, dim) 将当前维度 dim 和最后一个维度交换。reshape(-1,  self.mesh.size(dim))
                ### 将网格重塑为一个二维数组，其中每一行代表一个子网格。

                # multi-dim mesh, create subgroups by looping over the pg_ranks
                # for each dim and append the groups
                ### 为每个维度创建子进程组
                for dim_mesh in pg_ranks_by_dim:
                    ### pg_ranks_by_dim 是一个二维数组，循环遍历每一行（即每个子网格），并将其转换为列表 subgroup_ranks。
                    subgroup_ranks = dim_mesh.tolist()
                    # call new_group regardless of the current rank in the
                    # pg or not, it's required that all ranks participate
                    # in subgroup construction

                    ### 创建新的子进程组：
                    dim_group = new_group(ranks=subgroup_ranks)
                    ### 调用 new_group 函数创建一个新的进程组，成员是 subgroup_ranks 中的进程。

                    # only add to dim_groups if the current rank in the subgroup
                    ### 检查当前进程是否在子进程组中, 如果当前进程的 rank 在 subgroup_ranks 中，则继续执行。
                    if self.get_rank() in subgroup_ranks:
                        ### 确保每个维度只有一个进程组
                        """
                        确保每个维度只有一个进程组的原因主要是为了避免资源冲突和确保计算的一致性。以下是详细解释：

                        资源管理：
                            每个进程组（process group）在分布式计算中通常会占用一定的资源（如通信带宽、内存等）。如果一个维度有多个进程组，可能会导致资源的竞争和浪费。
                        计算一致性：
                            在分布式计算中，确保每个维度只有一个进程组可以避免数据的不一致性。多个进程组可能会导致同一维度的数据被不同的进程组处理，从而产生不一致的计算结果。
                        简化通信：
                            只有一个进程组可以简化进程间的通信。多个进程组会增加通信的复杂性，可能需要额外的同步和协调机制。
                        避免死锁：
                            多个进程组可能会导致死锁的风险。确保每个维度只有一个进程组可以减少这种风险，确保进程间的通信顺畅。
                        代码逻辑：
                            从代码逻辑上看，确保每个维度只有一个进程组可以简化代码的实现和维护。多个进程组会增加代码的复杂性，可能需要额外的逻辑来处理不同进程组之间的关系。
                        在代码中，通过以下检查来确保每个维度只有一个进程组：
                        """
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, but got {self.get_rank} "
                                f"in {subgroup_ranks}!"
                            )
                            """
                            这段代码检查 dim_group_infos 的长度是否大于当前维度 dim，
                            如果是，则抛出异常，表示每个维度只能有一个进程组。
                            这种检查机制确保了上述的资源管理、计算一致性、简化通信、避免死锁等目标。
                            """
                        dim_group_infos.append((_get_group_tag(dim_group), subgroup_ranks))
        self._dim_group_infos = dim_group_infos

    def __enter__(self) -> "DeviceMesh":
        # set this mesh as the current mesh in mesh env
        mesh_resources.mesh_stack.append(self)
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # pop this mesh from mesh env
        mesh_resources.mesh_stack.pop()

    def __repr__(self) -> str:
        return f"DeviceMesh:({self.mesh.tolist()})"

    def __hash__(self):
        # ideally, we should use object id as hash, because different device mesh objects
        # give different subprocess group, so different device meshes.
        # in practice of sharding propagation,
        # we only care about different mesh tensor (value, shape).
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self.mesh) == id(other.mesh):  # short-cut eq
            return True
        if self.device_type != other.device_type:
            return False
        return self.mesh.shape == other.mesh.shape and self._flatten_mesh_list == other._flatten_mesh_list

    def __getitem__(self, mesh_dim_name: str) -> "DeviceMesh":
        """
        Slice the current DeviceMesh based on the mesh_dim_name given to create a child
        DeviceMesh.

        Args:
            mesh_dim_name (str): the name of the mesh dimension of the parent DeviceMesh
            to create a child DeviceMesh for.
        Returns:
            A :class:`DeviceMesh` object

        Example (2 host with 4 GPUs each):
        ```
        # Below is a DeviceMesh with mesh_shape of (2, 4) and mesh_dim_name of ("dp", "tp")
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ],
                          mesh_dim_names=["dp", "tp"])
                          )
        ```
        Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).
        Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).
        Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).
        Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).
        Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).
        Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).
        """

        """
        如何解释给小学生:

        要向小学生解释这段代码，可以把它比喻成一个玩具积木系统，每个“DeviceMesh”就像一个大盒子，
        里面装着很多小积木块。这些积木块代表了电脑的不同部分，比如不同的处理器或显卡。

        ### 第一部分：获取积木盒子的一部分
        - 代码的第一个函数 `get_tensor_parallel_mesh` 就像是在大盒子里找到特定的一部分积木，比如只要绿色积木。
          你需要先确认大盒子里有所有的积木(`assert self._GLOBAL_MESH`)，然后找到你想要的部分(`self._MESH_DIM_NAMES_MAPPING[0]`)，最后把那部分积木拿出来。

        ### 第二部分：从大盒子中切出一个小盒子
        - 第二个函数 `__getitem__` 则像是在大盒子里切出一个小盒子，比如你想要的部分是所有小车的积木。
          你要告诉程序你想要什么样的小盒子（比如小车的盒子叫“tp”），然后程序会帮你在大盒子里找到这一部分(`mesh_dim_name`)。

        举个例子，假设你有一个2层的积木盒子，每层里面有4块积木，这些积木代表了不同的部分。
        如果你说你想要小车（“tp”），程序就会帮你把每层的小车部分挑出来。
        如果你说你想要楼房（“dp”），它就会帮你找到不同楼层里的楼房积木。

        通过这个方法，你可以从大盒子中快速找到自己想要的积木，不用每次都从头开始找，节省时间。

        ### 关键点
        - **积木盒子** = `DeviceMesh`
        - **找积木的名字** = `mesh_dim_name`
        - **确认盒子里有积木** = `assert`
        - **挑选积木** = `return`

        这个比喻可以帮助小学生理解代码的基本功能和逻辑。
        """

        """
        在大模型分布式训练中, dp 在这个例子里面是如何数据并行的?
        ===============================================

        在这个例子中，"dp" 代表数据并行（Data Parallelism），是一种在大模型分布式训练中常用的方法。数据并行的核心思想是将相同的模型副本分布在不同的设备（如GPU）上，并将数据划分为多个批次，每个设备处理一部分数据，然后在训练过程中每个设备独立计算梯度，最后将所有设备计算得到的梯度聚合起来，更新模型参数。

        ### 具体解释这个例子中的数据并行:

        假设你有两个主机（host），每个主机上有4个GPU。这里的 `DeviceMesh` 是一个 2x4 的矩阵，其中每一行代表一个主机，每一列代表该主机上的一个GPU：

        ```
        [
        [0, 1, 2, 3],  # 第一台主机上的4个GPU
        [4, 5, 6, 7]   # 第二台主机上的4个GPU
        ]
        ```

        在 `mesh_dim_names=["dp", "tp"]` 中，"dp" 代表数据并行（Data Parallelism），"tp" 代表张量并行（Tensor Parallelism）。

        #### 当你调用 `mesh["dp"]` 时：
        - `mesh["dp"]` 会返回一个 1D 的子 `DeviceMesh`，这个子 `DeviceMesh` 包含在数据并行维度上处于同一个位置的设备。例如：
        - `mesh["dp"]` 在 rank 0 和 rank 4 上调用时，会返回 `[0, 4]`，这表示在数据并行的维度上，两个主机上的这两个GPU将共享相同的一份数据。
        - 同理，在 rank 1 和 rank 5 上调用时，会返回 `[1, 5]`，依此类推。

        #### 数据并行的实现方式：
        1. **数据分割**: 输入的数据会被分割成多个批次（mini-batches），每个批次分别发送到不同的设备上（比如 `[0, 4]` 共享一份数据）。
        2. **独立计算**: 每个设备独立地计算模型前向传播和反向传播。
        3. **梯度聚合**: 在每个设备计算完梯度后，这些梯度会在所有参与数据并行的设备之间进行通信和聚合（通常使用 `AllReduce` 操作），最终更新模型的参数。

        这样，所有设备都会参与模型的训练，每个设备处理不同的数据，但共享相同的模型参数，最终得到与单机训练类似的结果。

        ### 总结
        在这个例子中，"dp" 维度上的设备（如 `[0, 4]`，`[1, 5]` 等）代表参与数据并行的设备组。数据并行通过在这些设备上并行处理不同的数据子集，并在每次迭代后聚合梯度，从而加速大模型的训练过程。
        """

        """
        在上面这个例子里面如何解释 tp: tensor parallelism?
        ===============================================

        在这个例子中，"tp" 代表张量并行（Tensor Parallelism），这是在大模型分布式训练中常用的一种并行方法。与数据并行不同，张量并行的核心思想是将模型的某些计算或张量分割到多个设备上，并行处理，从而减少单个设备的内存负担和计算量。

        ### 具体解释这个例子中的张量并行:

        同样，我们有两个主机（host），每个主机上有4个GPU。`DeviceMesh` 是一个 2x4 的矩阵：

        ```
        [
        [0, 1, 2, 3],  # 第一台主机上的4个GPU
        [4, 5, 6, 7]   # 第二台主机上的4个GPU
        ]
        ```

        在 `mesh_dim_names=["dp", "tp"]` 中，"tp" 代表张量并行（Tensor Parallelism）。

        #### 当你调用 `mesh["tp"]` 时：
        - `mesh["tp"]` 会返回一个 1D 的子 `DeviceMesh`，这个子 `DeviceMesh` 包含在张量并行维度上处于同一个位置的设备。例如：
        - `mesh["tp"]` 在 rank 0, 1, 2, 3 上调用时，会返回 `[0, 1, 2, 3]`，这表示在张量并行的维度上，这些设备将一起分担模型的某些计算任务。
        - 同理，在 rank 4, 5, 6, 7 上调用时，会返回 `[4, 5, 6, 7]`，这组设备将一起分担另外一部分模型的计算。

        #### 张量并行的实现方式：
        1. **模型切分**: 在张量并行中，模型的某些大张量（如权重矩阵）会被切分成多个子张量，分别分配到不同的设备上。例如，一个大的矩阵可能会被按列或按行切分，每一部分被分配到不同的GPU上进行计算。

        2. **并行计算**: 各个设备并行地对其负责的张量部分进行计算。例如，如果一个矩阵乘法操作被分割到四个GPU上，每个GPU将只处理这个矩阵的一部分乘法。

        3. **结果聚合**: 在每个设备完成计算后，它们的结果会被聚合起来，形成完整的输出。这通常通过设备之间的通信操作来完成。

        ### 例子中的张量并行
        - 当你调用 `mesh["tp"]` 时，比如 `mesh["tp"]` 在 rank 0, 1, 2, 3 上返回 `[0, 1, 2, 3]`，这表示这四个GPU将共同处理一个模型的张量操作，比如某个权重矩阵的计算。
        - 类似地，`mesh["tp"]` 在 rank 4, 5, 6, 7 上返回 `[4, 5, 6, 7]`，这表示另外四个GPU也将共同处理模型的另一部分张量操作。

        ### 总结
        在这个例子中，"tp" 维度上的设备组（如 `[0, 1, 2, 3]` 和 `[4, 5, 6, 7]`）代表参与张量并行的设备组。张量并行通过将模型的某些计算任务分割到多个设备上并行执行，从而减少每个设备的计算和内存负担，提高大模型训练的效率。
        """

        """
        要向初中生解释 `create_child_mesh`，可以这样说：
        ========================================================================

        想象一下你有一张二维的座位表，表格中每个格子代表一个学生坐在某个位置。现在，你想根据某种规则，比如按排或按列，把这些学生分成更小的组。`create_child_mesh` 就是执行这个分组任务的函数。

        在这个座位表的例子中：

        1. **DeviceMesh** 可以理解为整个座位表，它包含所有的学生（即设备或处理单元）。
        2. **Mesh dimension（网格维度）** 是你要根据的规则，可以是行（`dp`）或列（`tp`）。
        3. **create_child_mesh** 就像是你选择了一条规则，比如按列（`tp`），然后它会帮你从原来的座位表中切出对应的列，把这些列中学生分成一个新的、更小的组。

        具体来说，当你调用 `create_child_mesh`，它会按照你指定的维度，把设备（或者说处理单元）重新组合成一个更小的网格，这样你就可以在这个更小的网格上执行特定的任务。

        这就好比你在学校的一个大班里挑出一些学生组成一个小组，这个小组可以专注于某个特别的项目或者活动，而不是让整个班级都参与进来。

        -----------------------------------------------------------------------

        向博士解释 `create_child_mesh` 可以涉及更深入的技术细节和应用场景，结合分布式深度学习和高性能计算的背景。
        ========================================================================

        `create_child_mesh` 是一个在分布式计算框架中，用于基于指定的维度，从父 `DeviceMesh` 中生成子 `DeviceMesh` 的方法。这个过程涉及在多维设备拓扑结构中，动态地划分计算资源，以优化任务分配和执行效率。

        ### 背景与动机
        在大规模深度学习训练中，模型通常需要跨多个计算设备（如 GPU 或 TPU）进行分布式训练。为了管理这些设备，`DeviceMesh` 被用来描述这些设备在多维空间中的分布。`DeviceMesh` 是一个 n 维网格，其中每个维度表示设备的某种并行策略，如数据并行（Data Parallelism, DP）、张量并行（Tensor Parallelism, TP）等。

        然而，在实际应用中，不同的任务对计算资源的需求可能不同。例如，一些任务可能更适合在数据并行的维度上进行划分，而另一些任务可能需要在张量并行的维度上进行划分。这就需要我们能够灵活地从原始 `DeviceMesh` 中生成子网格，以满足不同任务的需求。

        ### `create_child_mesh` 的工作原理
        `create_child_mesh` 通过以下步骤来生成子 `DeviceMesh`：

        1. **维度选择与交换：** 首先，根据指定的 `mesh_dim_name` 找到对应的维度索引 `mesh_dim`。然后，将这个维度与最后一个维度进行交换，使其成为次要维度。这样做的目的是将目标维度的设备组合在一起，便于后续的划分。

        2. **子网格生成：** 交换后，对设备网格进行重新形状调整，使得目标维度上的设备被平展到一起。然后，遍历这个新形状的网格，根据目标维度提取出一维的子网格（子设备集）。

        3. **当前设备检查与分配：** 对每个子网格，检查当前设备是否属于该子网格。如果属于，则将该子网格作为当前任务的子 `DeviceMesh`，并记录其父 `DeviceMesh` 以便追踪层次关系。

        4. **子父关系映射：** 在生成子 `DeviceMesh` 后，更新映射关系，将子网格与父网格关联起来，确保在需要时可以追溯设备的分布结构。

        ### 实际应用与优势
        在分布式深度学习中，`create_child_mesh` 提供了一种灵活的方式来动态调整设备的组织结构，从而适应不同的计算需求。例如，在 Transformer 模型训练中，你可能需要在某些层使用张量并行，而在其他层使用数据并行。通过 `create_child_mesh`，可以在运行时动态生成适合每一层的子网格，从而最大化计算效率并减少通信开销。

        此外，`create_child_mesh` 的设计考虑到了扩展性和灵活性。通过允许动态调整设备的组织结构，它能够适应不同的集群配置和任务需求。这在异构计算环境（如混合使用 GPU 和 TPU）中特别有用，因为不同的硬件可能对并行策略有不同的要求。

        ### 总结
        `create_child_mesh` 是一个关键的方法，支持在复杂分布式系统中动态生成和管理设备子网格。它通过灵活的设备划分策略，优化了分布式深度学习中的计算资源分配，提升了模型训练的效率和扩展性。在大规模模型训练和高性能计算的背景下，这种动态子网格生成的能力是实现高效并行计算的核心技术之一。

        """

        if self.mesh.ndim <= 1:
            raise RuntimeError(f"Cannot slice a DeviceMesh with {self.mesh.ndim} dimension.")
        if self.mesh_dim_names is None:
            raise KeyError(
                "No `mesh_dim_names` found.",
                "To slice the device mesh, please call `init_device_mesh` with `mesh_dim_names`.",
            )
        if mesh_dim_name not in self.mesh_dim_names:
            raise KeyError(
                f"Mesh dimension '{mesh_dim_name}' does not exist.",
                f"Available mesh dimensions are: {self.mesh_dim_names}",
            )
        mesh_dim = self.mesh_dim_names.index(mesh_dim_name)
        submesh = mesh_resources.create_child_mesh(self, mesh_dim, mesh_dim_name)

        return submesh

    def get_dim_groups(self, mesh_dim: Optional[int] = None) -> Union[ProcessGroup, List[ProcessGroup]]:
        if not hasattr(self, "_dim_group_infos"):
            raise RuntimeError("DeviceMesh process groups not initialized!")
        if mesh_dim is not None:
            return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(_find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim]))
            return dim_groups

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    @property
    def ndevice(self) -> int:
        return torch.numel(self.mesh)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.mesh.shape)

    def get_rank(self) -> int:
        return get_rank()

    def get_local_rank(self, mesh_dim: Optional[int] = None) -> int:
        """
        Returns the local rank of the given mesh_dim of the DeviceMesh.

        Args:
            mesh_dim (int, optional): it is the index of the mesh dimension. Default is None.

        Returns:
            An integer denotes the local rank.

        The following program runs on each process/rank in an SPMD manner. In this example, we have 2
        hosts with 4 GPUs each.
        Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 0, 1, 2, 3 would return 0.
        Calling mesh_2d.get_local_rank(mesh_dim=0) on rank 4, 5, 6, 7 would return 1.
        Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 0, 4 would return 0.
        Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 1, 5 would return 1.
        Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 2, 6 would return 2.
        Calling mesh_2d.get_local_rank(mesh_dim=1) on rank 3, 7 would return 3.
        """
        if self.ndim > 1 and mesh_dim is None:
            raise RuntimeError(
                f"Found the DeviceMesh have {self.mesh.ndim} dimensions",
                "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1.",
            )
        elif mesh_dim is None:
            mesh_dim = 0

        mesh_dim_group = self.get_dim_groups(mesh_dim)
        assert isinstance(mesh_dim_group, ProcessGroup), "We expect ProcessGroup before calling `get_rank`!"

        return get_rank(mesh_dim_group)

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None

    def enforce_cpu_mesh_tensor(self) -> None:
        """
        move `mesh` tensor to cpu for deterministic device;
        necessary for comparison and checkpoint loading.
        """
        with torch.no_grad():
            self.mesh = self.mesh.cpu()

    def get_submesh(self, mesh_dims: Union[List[int], List[str]]) -> "DeviceMesh":
        dims = []
        for dim in mesh_dims:
            if isinstance(dim, int):
                dims.append(dim)
            elif isinstance(dim, str):
                assert dim in self.mesh_dim_names, f"Mesh dimension '{dim}' does not exist."
                dims.append(self.mesh_dim_names.index(dim))
        return mesh_resources.create_submesh_along_multi_dims(self, dims)

    def get_all_submesh(self, dim: int or str) -> List["DeviceMesh"]:
        if isinstance(dim, str):
            assert dim in self.mesh_dim_names, f"Mesh dimension '{dim}' does not exist."
            mesh_dim = self.mesh_dim_names.index(dim)
        else:
            mesh_dim = dim
        return mesh_resources.create_submesh_group(self, mesh_dim)

    def get_mapping_rank(self, other: "DeviceMesh"):
        """
        for cross mesh resharding
        we assume that the mesh is 1,2,4,8
        the size will have gcd value
        """
        mesh_list = self.mesh.view(-1).tolist()
        index = mesh_list.index(self.get_rank())
        other_mesh_list = other.mesh.view(-1).tolist()
        gcd_value = math.gcd(len(mesh_list), len(other_mesh_list))
        if gcd_value == 1 and len(mesh_list) != 1 and len(other_mesh_list) != 1:
            raise RuntimeError(f"mesh resharding the wrong shape of device mesh {mesh_list} vs {other_mesh_list}")

        a = len(mesh_list)
        b = len(other_mesh_list)
        factor = max(a, b) // min(a, b)

        if a > b:  # group down
            data = {}
            for i in range((index // factor) * factor, factor):
                data.update({mesh_list[index]: other_mesh_list[index // factor]})
            return data
        elif a < b:  # group up
            return [other_mesh_list[i] for i in range(index * factor, (index + 1) * factor)]
        else:
            return other_mesh_list[index]


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
) -> DeviceMesh:
    """
    Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.
    This creates a DeviceMesh with a mesh layout of n-d dimensional array, n being the len(mesh_shape)
    and ith dimension being in size mesh_shape[i]. If mesh_dim_names is provided, each dimension is
    labeled as mesh_dim_names[i].


    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
        mesh_shape: Tuple[int]: A tuple describes the dimension of the multi-dimesnion array
        that describes the layout of devices.
    Kwargs:
        mesh_dim_names: Optional[Tuple[str]]: A tuple of mesh dim names to be assigned to each dimension
        of the multi-dimensional array that describes the layout of devices. Its length must match the length
        of `mesh_shape`. Each string in mesh_dim_names must be unique.

    Returns:
        A :class:`DeviceMesh` object

    .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups
    behind the scene, which are required for distributed communications.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.distributed._tensor.device_mesh import init_device_mesh
        >>>
        >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
        >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
    """
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError(
                "Each mesh_dim_name must be uqique.",
                f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}",
            )

        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                "mesh_shape and mesh_dim_names should have same length!",
                f"Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}.",
            )

    mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
