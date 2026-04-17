# 错误码参考

## HIP 错误码

| 错误码 | 含义 | 常见原因 | 解决方案 |
|--------|------|---------|---------|
| `hipErrorOutOfMemory` | 显存不足 | 模型/批次过大 | 减小 batch size 或使用量化 |
| `hipErrorInvalidValue` | 无效参数 | 参数配置错误 | 检查 API 参数 |
| `hipErrorNoDevice` | 无可用设备 | 驱动问题 | 检查 rocm-smi |
| `hipErrorPeerAccessUnsupported` | 不支持 P2P | 硬件/拓扑限制 | 检查 DCU 拓扑 |

## PyTorch 错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `CUDA error: device-side assert triggered` | 张量索引越界 | 检查输入数据 |
| `RuntimeError: Expected all tensors to be on the same device` | 设备不一致 | 检查 `.to(device)` |
| `RuntimeError: HIP error: invalid argument` | HIP 参数错误 | 检查 kernel 参数 |
| `OOM when allocating tensor` | 显存不足 | 减小模型/批次 |

## vLLM 错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `ValueError: The model's max seq len is smaller` | max_model_len 超限 | 减小 --max-model-len |
| `torch.cuda.OutOfMemoryError` | 显存不足 | 降低 gpu-memory-utilization |
| `RuntimeError: Cannot add new token` | 超出并发限制 | 增大 --max-num-seqs |

## NCCL 错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `NCCL error: unhandled system error` | 网络问题 | 检查网络连接 |
| `NCCL WARN: Connect/recv failed` | 节点通信失败 | 检查防火墙和 SSH |
| `NCCL error: invalid usage` | NCCL 配置错误 | 检查环境变量 |
