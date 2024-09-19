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
### Non-appointed parameters and buffers will be `Replicate` (i.e., default plan). from vescale/dmodule/_dmodule.py
from vescale.dtensor.placement_types import Replicate, Shard

fwd_plan = {
    "transformer.wte.input": [[Replicate()]],
    "transformer.wte.output": [[Replicate()]],
    "transformer.wpe.input": [[Replicate()]],
    "transformer.wpe.output": [[Replicate()]],
    r"transformer.h.\d+.input": [[Shard(1)]],
    r"transformer.h.\d+.attn.input": [[Replicate()]],
    r"transformer.h.\d+.attn.c_proj.output": [[Replicate()]],
    r"transformer.h.\d+.attn.output": [[Shard(1)]],
    r"transformer.h.\d+.mlp.c_fc.input": [[Replicate()]],
    r"transformer.h.\d+.mlp.c_proj.output": [[Replicate()]],
    r"transformer.h.\d+.mlp.output": [[Shard(1)]],
    "transformer.ln_f.input": [[Shard(1)]],
    "lm_head.input": [[Shard(2)]],
    "lm_head.output": [[Replicate()]],
}

fwd_plan_dist_dropout = {
    "transformer.wte.input": [[Replicate()]],
    "transformer.wte.output": [[Replicate()]],
    "transformer.wpe.input": [[Replicate()]],
    "transformer.wpe.output": [[Replicate()]],
    r"transformer.h.\d+.input": [[Shard(1)]],
    r"transformer.h.\d+.attn.input": [[Replicate()]],
    r"transformer.h.\d+.attn.c_proj.output": [[Shard(1)]],
    r"transformer.h.\d+.attn.output": [[Shard(1)]],
    r"transformer.h.\d+.mlp.c_fc.input": [[Replicate()]],
    r"transformer.h.\d+.mlp.c_proj.output": [[Shard(1)]],
    r"transformer.h.\d+.mlp.output": [[Shard(1)]],
    "transformer.ln_f.input": [[Shard(1)]],
    "lm_head.input": [[Shard(2)]],
    "lm_head.output": [[Replicate()]],
}

params_plan = {
    "transformer.wte.weight": [Shard(1)],
    "transformer.wpe.weight": [Shard(1)],
    r"transformer.h.\d+.attn.q_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.q_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.k_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.k_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.v_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.v_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.c_proj.weight": [Shard(1)],
    r"transformer.h.\d+.attn.c_proj.bias": [Replicate()],
    r"transformer.h.\d+.mlp.c_fc.weight": [Shard(0)],
    r"transformer.h.\d+.mlp.c_fc.bias": [Shard(0)],
    r"transformer.h.\d+.mlp.c_proj.weight": [Shard(1)],
    r"transformer.h.\d+.mlp.c_proj.bias": [Replicate()],
    "lm_head.weight": [Shard(1)],
}

nanoGPT_plan = {"parameter": params_plan, "forward": fwd_plan}

nanoGPT_plan_dist_dropout = {"parameter": params_plan, "forward": fwd_plan_dist_dropout}





"""
The main difference between `fwd_plan` and `fwd_plan_dist_dropout`
=================================================================

The main difference between `fwd_plan` and `fwd_plan_dist_dropout` lies in how the outputs of certain layers, particularly the attention (`attn`) and MLP (`mlp`) components, are treated in terms of sharding and replication.

### Key differences:
1. **Attention Outputs (`attn.c_proj.output`)**:
   - In `fwd_plan`: `r"transformer.h.\d+.attn.c_proj.output": [[Replicate()]]` — This indicates that the output of the `attn.c_proj` layer is fully replicated across all devices.
   - In `fwd_plan_dist_dropout`: `r"transformer.h.\d+.attn.c_proj.output": [[Shard(1)]]` — This indicates that the output of the `attn.c_proj` layer is sharded along the 1st dimension.

2. **MLP Outputs (`mlp.c_proj.output`)**:
   - In `fwd_plan`: `r"transformer.h.\d+.mlp.c_proj.output": [[Replicate()]]` — This indicates that the output of the MLP's `c_proj` layer is replicated.
   - In `fwd_plan_dist_dropout`: `r"transformer.h.\d+.mlp.c_proj.output": [[Shard(1)]]` — This indicates that the output of the MLP's `c_proj` layer is sharded.

### General Concept:
- **`fwd_plan`** focuses on **replicating** key outputs like `attn.c_proj.output` and `mlp.c_proj.output`, ensuring these outputs are fully available across all devices. This can be useful for more synchronized operations, where multiple devices need full access to the same output data.

- **`fwd_plan_dist_dropout`** introduces **sharding** in those areas (`attn.c_proj.output` and `mlp.c_proj.output`) where the `fwd_plan` used replication. Sharding reduces the amount of data that needs to be transferred across devices and may be more suitable for distributed training with dropout layers, optimizing memory usage and communication.

This suggests that `fwd_plan_dist_dropout` is more tailored towards distributed settings where efficiency in memory and communication is key, likely in scenarios involving dropout, which may involve dropping parts of the network for certain devices.






What is shard(1), shard(2)?
===========================

In the context of distributed deep learning, `Shard(1)` and `Shard(2)` refer to how tensor data is split (or "sharded") across multiple devices along specific dimensions. This is important for distributing computations and model parameters efficiently across multiple devices (e.g., GPUs, TPUs) to parallelize training.

### Breakdown of `Shard(1)` and `Shard(2)`:
- **`Shard(1)`**: This means the tensor is sharded (split) along the **1st dimension** (the second axis in a 0-based index system). For example, if the tensor has shape `[batch_size, seq_len, hidden_dim]`, sharding along the 1st dimension would split the tensor across the `seq_len` dimension. Different devices would each get a portion of this dimension.

- **`Shard(2)`**: This means the tensor is sharded along the **2nd dimension** (the third axis in a 0-based index system). Using the same example shape `[batch_size, seq_len, hidden_dim]`, sharding along the 2nd dimension would split the tensor across the `hidden_dim` dimension, distributing parts of the hidden layer across different devices.

### Use Case:
Sharding is used in distributed systems to partition data or model parameters across multiple devices. By dividing the tensor across a specific dimension:
- **`Shard(1)`** might be used to distribute sequences across devices, which is useful for parallelizing computations over longer sequences.
- **`Shard(2)`** could be used to distribute the hidden dimensions of a model, spreading out the weight matrices and activations to reduce memory consumption per device.

### Example:
If you have a tensor of shape `[64, 128, 1024]` (batch of 64, sequence length of 128, hidden size of 1024):
- **`Shard(1)`**: The sequence length (128) would be divided among the devices. If there are 4 devices, each device would get a sub-tensor of shape `[64, 32, 1024]` (splitting the sequence length evenly).
- **`Shard(2)`**: The hidden dimension (1024) would be divided among the devices. If there are 4 devices, each device would get a sub-tensor of shape `[64, 128, 256]` (splitting the hidden dimension evenly).

Sharding helps scale models to larger sizes by distributing the workload and memory requirements across devices.
"""