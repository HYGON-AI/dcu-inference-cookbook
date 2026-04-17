# 算子优化

## 算子融合

算子融合（Operator Fusion）将多个小算子合并为一个 kernel，减少显存访问和 kernel launch 开销。

### 常见融合算子

| 融合算子 | 组成 | 加速效果 |
|---------|------|---------|
| Fused LayerNorm | LayerNorm + Residual | 10-20% |
| Fused Attention | QKV Proj + Attention + Output Proj | 15-30% |
| Fused MLP | Gate + Up + Down Proj | 10-15% |
| Fused RMSNorm | RMSNorm + Residual | 10-20% |

### Flash Attention

```bash
# vLLM 默认使用 Flash Attention
# 确保安装了兼容版本
pip install flash-attn --no-build-isolation
```

## Triton 自定义算子

DCU 支持 Triton 编写自定义算子：

```python
import triton
import triton.language as tl

@triton.jit
def fused_add_relu_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.maximum(x + y, 0)  # Fused Add + ReLU
    tl.store(output_ptr + offsets, output, mask=mask)
```

## ROCm 特定优化

### MIOpen 配置

```bash
# 预热 MIOpen kernel 缓存
export MIOPEN_FIND_ENFORCE=3
export MIOPEN_USER_DB_PATH=/path/to/cache

# 启用 MIOpen 性能模式
export MIOPEN_LOG_LEVEL=2
```

### HIP 优化

```bash
# 启用 HIP 可见设备
export HIP_VISIBLE_DEVICES=0,1,2,3

# 大页内存
export HSA_ENABLE_LARGE_PAGE=1
```

## 性能分析工具

```bash
# ROCm Profiler
rocprof --stats python inference.py

# PyTorch Profiler
python -c "
import torch
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    # 你的推理代码
    pass

print(prof.key_averages().table(sort_by='cuda_time_total'))
"
```
