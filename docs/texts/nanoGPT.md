# Experiment Results

log: https://gist.github.com/shizukanaskytree/ec72f12268dbd5c068283ffee9743836

### Analysis Report: Fine-tuning of GPT-2 on Shakespeare Dataset Using veScale Framework

#### Summary

This report analyzes the fine-tuning of a GPT-2 model on the Shakespeare dataset using the veScale framework. The training process utilized a 4D parallelism strategy with 2 data parallelism (DP) and 2 tensor parallelism (TP) sizes. The process involved multiple iterations, with evaluation at regular intervals to monitor the loss and performance. The analysis covers key points such as warnings, configurations, network communication, and performance metrics.

#### Key Observations

1. **Environment and Configuration Warnings**:
    - The environment variable `OMP_NUM_THREADS` was set to 1 by default to avoid system overload. This may require tuning for optimal performance.
    - Deprecation warnings were observed for `torch.utils._pytree._register_pytree_node`. These should be updated to the recommended alternatives to ensure compatibility with future versions.

2. **Model Configuration**:
    - The model was fine-tuned using a learning rate of 3e-5, with a batch size of 32, and gradient accumulation steps set to 1. The maximum number of iterations was 20.
    - Dropout was set to 0.1, and distributed dropout was enabled. The model was initialized from pre-trained OpenAI GPT-2 weights.

3. **Network Communication**:
    - The NCCL library was used for communication between GPUs. However, there were warnings about missing network plugins (`libnccl-net.so`), leading to the use of internal implementations.
    - The network communication utilized the Socket interface, and various environment variables like `NCCL_IB_DISABLE` and `NCCL_P2P_LEVEL` were set to customize the communication behavior.
    - The service threads reported the successful closing of connections and completion of communication setups after each training phase.

4. **Performance Metrics**:
    - The training started with a relatively high loss of 4.1266, which gradually decreased to 3.5719 by iteration 20. The validation loss followed a similar trend, starting at 4.0516 and decreasing to 3.4000.
    - The model utilization factor (MFU) was initially negative, indicating inefficiencies in the early stages of training. However, it improved as training progressed, reaching 2.22% by iteration 19.

5. **Training Dynamics**:
    - The initial training steps showed a steep decrease in both training and validation loss, indicating rapid learning. The training process appeared to stabilize after 10 iterations.
    - The MFU metric steadily improved, reflecting better utilization of resources over time. This could indicate that the model and environment adjustments, such as network optimizations, positively impacted performance.

6. **Overall Stability**:
    - The process concluded without any critical errors or crashes, indicating a stable training environment. The NCCL library successfully managed the multi-GPU communication, despite initial warnings about missing plugins.

#### Recommendations

1. **Update Deprecated Methods**: The deprecated `torch.utils._pytree._register_pytree_node` should be updated to the recommended `torch.utils._pytree.register_pytree_node` to maintain compatibility with future releases.

2. **Tune `OMP_NUM_THREADS`**: To optimize performance, the `OMP_NUM_THREADS` variable should be tuned according to the system's capabilities and the workload requirements.

3. **Investigate Network Plugins**: The absence of `libnccl-net.so` suggests potential areas for optimization in network communication. Installing the appropriate plugins might enhance performance, especially in distributed training scenarios.

4. **Monitor Resource Utilization**: The initial negative MFU suggests that resource utilization was inefficient during the early iterations. Monitoring and adjusting the parallelism strategy might improve efficiency further.

#### Conclusion

The fine-tuning of GPT-2 on the Shakespeare dataset using the veScale framework was successful, with significant improvements in loss and resource utilization observed over time. Despite some initial warnings and inefficiencies, the overall process was stable and showed a positive trend in model performance. Addressing the identified warnings and optimizing configurations could lead to further improvements in future training runs.


# nanoGPT fine-tuning Code on Shakespeare dataset



