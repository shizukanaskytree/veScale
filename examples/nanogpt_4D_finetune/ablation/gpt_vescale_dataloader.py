import os
import numpy as np
import torch
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale import distribute_tensor
from vescale.dtensor.placement_types import Replicate

# 设备和数据集的基本设置
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 1024
data_dir = "data/shakespeare"
batch_size = 8
local_batch_size = batch_size
ddp_rank = 0
world_size = 4
dp_size = 2
tp_size = 2

# 初始化 VeScale 的设备网格
VESCALE_DEVICE_MESH.init_device_mesh(device, (dp_size, tp_size), mesh_dim_names=["DP", "TP"])
mesh = VESCALE_DEVICE_MESH.get()

def get_batch(split, bsz=batch_size, lbsz=local_batch_size):
    """
    函数用于加载数据并生成X, Y的批次，支持多GPU和分布式训练。
    split: 'train' 或 'val'，用以选择数据集。
    bsz: 批次大小
    lbsz: 本地批次大小（在多GPU的情况下每个设备的批次大小）

    ----------------------------------------------------------------------------

    调用 `x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的
    主要目的是将张量 `x` 按照给定的设备网格 `VESCALE_DEVICE_MESH["TP"]` 和分布策略 `[Replicate()]`
    在分布式设备上进行分布式存储和计算。下面是具体含义：

    ### 1. **`distribute_tensor` 函数**
    这个函数负责将全局的 `torch.Tensor` 根据 `device_mesh` 和 `placements`（分布策略）
    在多设备上分布。它允许将张量拆分成不同的部分或在不同设备上复制，以优化并行计算。

    ### 2. **参数解释**
    - **`x`**: 这是一个全局的 `torch.Tensor`，你希望将其分布到多设备（例如多个 GPU）上，
      以实现并行计算。
    - **`VESCALE_DEVICE_MESH["TP"]`**: 这是一个 `DeviceMesh` 对象，表示设备的拓扑结构。
      在 `TP`（Tensor Parallelism，张量并行）维度上进行分布。它定义了哪些设备（例如 GPU）将参与计算，并指定张量的分布方式。
    - **`[Replicate()]`**: 这是分布策略，表示将张量复制到指定的设备网格中。具体来说，
      `Replicate()` 表示将整个张量在多个设备上进行复制，每个设备上都会有一份完整的张量拷贝。

    ### 3. **`Replicate()` 的含义**
    在 `placements` 中使用 `[Replicate()]` 表示不对张量进行分片，而是直接在每个设备上保留完整的张量。
    这通常用于需要每个设备上都访问同样数据的场景，比如并行化模型的某些部分时，希望在每个计算节点上都有相同的权重。

    ### 4. **工作原理**
    当你调用 `distribute_tensor` 函数时，函数首先验证张量和设备网格的维度是否匹配，并检查 `placements` 的合法性。
    接下来，函数根据 `Replicate()` 策略，在设备网格的每个维度上将张量复制到各个设备。
    复制后的张量可以在多个设备上并行计算，这在深度学习中可以显著加速训练。

    ### 5. **应用场景**
    - 在大规模分布式训练中，尤其是在涉及多 GPU 或多节点的情况下，使用 `distribute_tensor`
      可以有效管理和分发数据，从而提升计算效率。
    - 使用 `Replicate()` 复制张量，常见于某些参数或数据必须在多个设备上共享的场景，例如并行模型的权重共享。

    ### 总结
    `x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的作用是
    将张量 `x` 复制到 `VESCALE_DEVICE_MESH["TP"]` 设备网格中的每个设备上。这意味着每个设备上
    都会有 `x` 的完整拷贝，使得每个计算节点都可以访问相同的数据或权重，适用于并行计算的场景。

    ----------------------------------------------------------------------------

    在这段代码中，`x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])`
    的作用是将张量 `x` 复制到 `VESCALE_DEVICE_MESH["TP"]` 设备网格中的所有设备上。
    通过这个调用，数据可以被分布到多个 GPU 上，供并行计算使用。为了更具体地解释调用时的含义，
    结合实际参数，以下是详细说明：

    ### 1. **设备网格 (`VESCALE_DEVICE_MESH["TP"]`)**
    在代码中，设备网格通过以下方式初始化：

    ```python
    VESCALE_DEVICE_MESH.init_device_mesh(device, (dp_size, tp_size), mesh_dim_names=["DP", "TP"])
    ```

    - `dp_size = 2` 和 `tp_size = 2` 表示你使用了 2 个数据并行（DP）和 2 个张量并行（TP）的设备，这意味着你有 4 个设备（2×2）。
    - `mesh_dim_names=["DP", "TP"]` 指定了这两个维度的名称，分别表示数据并行和张量并行的维度。

    这里 `VESCALE_DEVICE_MESH["TP"]` 对应的是张量并行（TP）维度的设备网格。

    ### 2. **`distribute_tensor` 的作用**
    `distribute_tensor` 函数的主要作用是将张量 `x` 根据设备网格 `VESCALE_DEVICE_MESH["TP"]`
    和指定的分布策略 `[Replicate()]` 在多个设备上分布。具体的步骤如下：

    - **张量 `x` 的形状**: `x` 是一个形状为 `[8, 1024]` 的张量，
      表示你在这个批次（batch size = 8）中有 8 个样本，每个样本的长度是 1024（与 `block_size` 对应）。
    - **设备网格 `VESCALE_DEVICE_MESH["TP"]`**: 该网格表示 2 个设备的并行方式（即 `tp_size = 2`）。
      这个维度决定了张量如何在这些设备之间分布。
    - **`[Replicate()]`**: `Replicate()` 表示将张量完整地复制到每个设备上。
      在这种情况下，`x` 不会被分片，而是在张量并行维度的每个设备上保留完整的副本。

    ### 3. **调用 `distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的实际含义**
    - **设备网格的结构**: 你的设备网格中有 2 个张量并行设备（`tp_size = 2`）。这些设备可能是 2 个 GPU 或者其他计算设备。
    - **复制张量**: `Replicate()` 意味着张量 `x` 将被完整复制到 `TP` 维度的每个设备上。
      由于 `tp_size = 2`，这意味着每个 GPU 都会获得一份完整的 `x` 副本。
    - **并行计算准备**: 通过在每个设备上复制 `x`，你可以在多个 GPU 上执行相同的数据处理或计算操作。
      每个设备都可以独立地处理这份数据，而无需从其他设备获取。

    ### 4. **多设备情况下的处理**
    在多 GPU 环境下（`world_size = 4`，其中有 2 个 `DP` 设备和 2 个 `TP` 设备），
    张量 `x` 会被复制到 `TP` 维度的所有设备上，这样确保每个 GPU 都能访问完整的数据，而无需在设备之间传输。

    具体到你的设备拓扑结构：
    - **设备总数**: `dp_size = 2` 和 `tp_size = 2`，因此你有 4 个设备在并行训练。
    - **数据并行**: `DP` 维度负责处理不同的数据批次（每个 `DP` 设备处理不同的批次）。
    - **张量并行**: `TP` 维度负责在每个张量并行设备上保持相同的张量副本。

    通过这种方式，整个系统可以在多个设备上高效地分布计算任务。

    ### 5. **总结**
    `x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的作用是
    将张量 `x` 在张量并行维度 `TP` 的设备上进行复制。由于 `tp_size = 2`，
    意味着你有 2 个 GPU（或其他设备）会得到 `x` 的完整副本。这是为了让每个设备能够在同一批次的数据上独立进行并行计算。

    这种操作特别适合在多 GPU 环境中进行张量并行的训练场景，使得每个 GPU 都可以访问同样的数据，从而加速训练过程。
    """
    # 加载数据集，使用np.memmap避免内存泄漏
    if split == "train":
        data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    else:
        data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    # 随机生成索引
    ix = torch.randint(len(data) - block_size, (bsz,)).to(device)

    # 多GPU时，广播索引并分割到本地
    if world_size > 1:
        torch.distributed.broadcast(ix, src=0, async_op=False)
    ix = torch.split(ix, lbsz)[ddp_rank]

    # 生成X和Y批次
    x = torch.stack([torch.from_numpy((data[i: i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1: i + 1 + block_size]).astype(np.int64)) for i in ix])

    # 将X和Y转移到设备并启用pin_memory
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    # 使用 VeScale API 分布式张量
    """
    在这段代码中，`x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的作用与模型的 `sharding plan` （分片计划）是相关的，尤其是与模型数据和权重如何在设备上进行分布的策略有关。

    ### 为什么设置为 `[Replicate()]`？

    在训练大型模型（如 GPT）时，数据和模型的参数可能需要在多设备上进行分布。`Replicate()` 是一种策略，表示数据或者张量在设备网格中不进行切分，而是在每个设备上保留一份完整的副本。以下是具体原因：

    1. **数据复制策略**：数据加载部分的 `x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 使用了 `Replicate()`，表示将输入数据 `x` 复制到张量并行设备网格中的每个设备上。这意味着每个张量并行设备都有完整的输入数据拷贝。这对于训练时输入的并行处理是必要的，因为每个设备都需要完整的数据来计算损失和梯度。

    2. **与 `sharding plan` 的关系**：在 `sharding plan` 中，例如：
    ```python
    fwd_plan = {
        "transformer.wte.input": [[Replicate()]],
        "transformer.wte.output": [[Replicate()]],
    }
    ```
    这里的 `Replicate()` 与数据加载部分的 `Replicate()` 是一致的。它表明模型的嵌入层（`wte`）输入和输出都不会在张量并行（TP）维度上切分，而是在每个并行设备上保留完整的副本。这样做的原因是，在模型的前向传播中，某些层（例如嵌入层或某些层的输出）需要所有并行设备共享同样的数据。

    3. **为什么不进行 `Shard()` 分片？**
    对于模型参数的某些部分，例如大型线性层的权重，可能会使用 `Shard()` 来在多个设备上切分权重以减少每个设备的内存占用。但对于输入数据（如 `x` 和 `y`），它们通常需要被每个设备完整地保留，因此 `Replicate()` 是合理的选择。将输入数据切分（shard）会导致每个设备只能看到部分数据，这不适用于计算损失和梯度的场景。

    4. **与设备网格的关系**：
    ```python
    VESCALE_DEVICE_MESH.init_device_mesh(device, (dp_size, tp_size), mesh_dim_names=["DP", "TP"])
    ```
    这里初始化了设备网格，`dp_size = 4` 和 `tp_size = 1` 表示 4 个数据并行设备和 1 个张量并行设备。在这种配置下，`Replicate()` 作用于张量并行维度（`TP` 维度）。由于 `tp_size = 1`，张量并行设备网格实际上只有 1 个设备，因此输入数据直接复制到唯一的张量并行设备上。

    ### 具体调用 `x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 的含义

    - **输入数据 `x`**：数据加载函数 `get_batch` 中生成了输入数据 `x` 和目标标签 `y`，它们的形状为 `[batch_size, block_size]`，在你的例子中是 `[8, 1024]`。

    - **`distribute_tensor` 的作用**：`distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])` 将输入数据 `x` 复制到张量并行设备网格 `TP` 的每个设备上。在当前配置中，由于 `tp_size = 1`，实际上只有一个张量并行设备，但如果你将 `tp_size` 增加到 2 或更多，则 `x` 会复制到每个张量并行设备。

    ### 结论

    - 设置 `[Replicate()]` 是因为输入数据需要在每个张量并行设备上都保持一致，确保每个设备能够看到完整的输入数据，进行前向传播和梯度计算。
    - 这与 `sharding plan` 中对某些模型参数（如嵌入层）的处理一致，这些参数需要在每个并行设备上被复制，而不是切分。
    """
    if world_size > 1:
        x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])
        y = distribute_tensor(y, VESCALE_DEVICE_MESH["TP"], [Replicate()])

    return x, y

# 测试函数
if __name__ == "__main__":
    X, Y = get_batch("train")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    # X shape: torch.Size([8, 1024])
    # Y shape: torch.Size([8, 1024])
