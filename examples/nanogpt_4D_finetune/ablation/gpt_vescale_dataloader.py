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
    if world_size > 1:
        x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])
        y = distribute_tensor(y, VESCALE_DEVICE_MESH["TP"], [Replicate()])

    return x, y

# 测试函数
if __name__ == "__main__":
    X, Y = get_batch("train")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
