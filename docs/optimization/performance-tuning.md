# 性能调优

## 调优优先级

```
影响程度从高到低：

1. 数据类型 (bf16 vs fp16 vs fp32)
2. 批处理策略 (continuous batching)
3. KV Cache 管理 (PagedAttention)
4. 张量并行 (Tensor Parallelism)
5. 算子融合 (Kernel Fusion)
6. CUDA Graph (计算图缓存)
7. 量化 (INT8 / INT4)
```

## 数据类型选择

| 类型 | 精度 | 速度 | 显存 | 推荐 |
|------|------|------|------|------|
| FP32 | 高 | 慢 | 高 | ❌ 不推荐 |
| FP16 | 中 | 快 | 中 | ⚠️ 注意溢出 |
| BF16 | 中 | 快 | 中 | ✅ 首选 |
| INT8 | 低 | 最快 | 低 | ✅ 量化场景 |
| INT4 | 低 | 最快 | 最低 | ✅ 极限显存 |

## 批处理优化

### Continuous Batching

vLLM 和 SGLang 默认启用，核心思想：

```
传统 Static Batching:
[Req1 ████████████░░░░░░░░░░]  ← 等待最慢的请求
[Req2 ██████░░░░░░░░░░░░░░░░]
[Req3 ██████████████████████]

Continuous Batching:
[Req1 ████████████░░░░░░░░░░]
[Req2 ██████░░░░░░░░░░░░░░░░]
[Req3 ██████████████████████]
[Req4 ░░░░░░░░░░░░░░░░░░░░░░]  ← 新请求立即插入
```

### 调优参数

```bash
# vLLM
--max-num-seqs 256          # 最大并发请求数
--max-num-batched-tokens 8192  # 单次调度最大 token 数
```

## KV Cache 优化

### PagedAttention

- 将 KV Cache 分页管理，减少显存碎片
- vLLM 默认启用

### Prefix Caching

- 缓存公共前缀（system prompt），避免重复计算
- SGLang 的 RadixAttention 实现更优

```bash
# SGLang 启用 prefix caching
python -m sglang.launch_server \
    --model-path <model> \
    --enable-prefix-caching
```

## CUDA Graph

- 缓存计算图，减少 kernel launch 开销
- 适用于固定输入形状的场景

```bash
# vLLM 默认启用，调试时可关闭
--enforce-eager  # 禁用 CUDA Graph
```

## 环境变量调优

```bash
# ROCm 相关
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_ENABLE_PRE_VEGA=0

# 内存相关
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# NCCL 通信
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
```
