# Qwen3.5 on SGLang

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1100-2x-sglang-0510) |
|                                                                       | BF16 | 0.5.10 | K100_AI | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-k100_ai-2x-sglang-0510) |
| [Qwen/Qwen3.5-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1100-2x-sglang-0510) |
| [hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-fp8-w8a8-ifb-bw1100-2x-sglang-0510) |

### Qwen3.5-397B-A17B 部署配置

Qwen3.5-397B-A17B 采用 MoE 架构（397B 总参数 / 17B 激活参数）。

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Qwen3.5-397B-A17B-Channel-FP8](https://www.modelscope.cn/models/hygon/Qwen3.5-397B-A17B-Channel-FP8) | FP8 W8A8 | 0.5.10 | BW1100 |  4 | IFB  | [**`>_`**](#qwen35-397b-a17b-channel-fp8-ifb-bw1100-4x-sglang-0510)   |
|                                                                                                            | FP8 W8A8 | 0.5.10 | BW1100 | 12 | 1P1D | [**`>_`**](#qwen35-397b-a17b-channel-fp8-1p1d-bw1100-12x-sglang-0510) |
| [hygon/Qwen3.5-397B-A17B-W8A8](https://www.modelscope.cn/models/hygon/Qwen3.5-397B-A17B-W8A8)              | INT8 W8A8 | 0.5.10 | BW1000 |  8 | IFB  | [**`>_`**](#qwen35-397b-a17b-w8a8-ifb-bw1000-8x-sglang-0510)          |

## 启动命令

### Qwen3.5-27B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path Qwen/Qwen3.5-27B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --reasoning-parser qwen3 \
    --chunked-prefill-size -1
```
### Qwen3.5-27B IFB K100_AI 2x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_ROCM_USE_AITER_MOE=False
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path Qwen/Qwen3.5-27B \
    --tp-size 2 \
    --attention-backend fa3 \
    --page-size 64 \
    --pp-size 1 \
    --mem-fraction-static 0.8 \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 256 \
    --max-prefill-tokens 45000 \
    --mamba-scheduler-strategy extra_buffer \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2
```

### Qwen3.5-35B-A3B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path Qwen/Qwen3.5-35B-A3B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --enable-piecewise-cuda-graph \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --chunked-prefill-size -1
```

> NMZ 卡使用 `fp8_e4m3`，非 NMZ 卡使用 `fp8_e5m2`，请按照使用硬件情况进行配置。

### Qwen3.5-35B-A3B-Channel-FP8-w8a8 IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --enable-piecewise-cuda-graph \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --chunked-prefill-size -1 \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --mamba-scheduler-strategy extra_buffer
```

### Qwen3.5-397B-A17B-Channel-FP8 IFB BW1100 4x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_KV_LAYOUT_DCU_FA=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_MODELSCOPE=1
export GPU_MAX_HW_QUEUES=4

sglang serve \
  --numa-node 0 0 0 0 1 1 1 1 \
  --model-path hygon/Qwen3.5-397B-A17B-Channel-FP8 \
  --trust-remote-code \
  --tp-size 4 \
  --pp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --watchdog-timeout 36000 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-piecewise-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --skip-server-warmup \
  --cuda-graph-max-bs 50
```

> 图文数据集场景需额外添加：环境变量 `export SGLANG_USE_CUDA_IPC_TRANSPORT=1`，启动命令添加 `--mm-attention-backend fa3`，`--keep-mm-feature-on-device`

### Qwen3.5-397B-A17B-Channel-FP8 1P1D BW1100 12x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_KV_LAYOUT_DCU_FA=1
export SGLANG_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export MC_TOPO_FILE_FORCE=mc_topo.config
export MC_IB_GID_INDEX=1
export GPU_MAX_HW_QUEUES=4

sglang serve \
  --numa-node 0 0 0 0 1 1 1 1 \
  --model-path hygon/Qwen3.5-397B-A17B-Channel-FP8 \
  --trust-remote-code \
  --tp-size 4 \
  --pp-size 1 \
  --dtype bfloat16 \
  --host "$(ip route get 1.1.1.1 | awk '/src/{print $7}')" \
  --port 30000 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.90 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --disable-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

> 图文数据集场景需额外添加：环境变量 `export SGLANG_USE_CUDA_IPC_TRANSPORT=1`，启动命令添加 `--mm-attention-backend fa3`。

#### D node 0

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_KV_LAYOUT_DCU_FA=1
export SGLANG_USE_MODELSCOPE=1
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export MC_TOPO_FILE_FORCE=mc_topo.config
export MC_IB_GID_INDEX=1
export GPU_MAX_HW_QUEUES=4

sglang serve \
  --numa-node 0 0 0 0 1 1 1 1 \
  --model-path hygon/Qwen3.5-397B-A17B-Channel-FP8 \
  --trust-remote-code \
  --tp-size 8 \
  --pp-size 1 \
  --dtype bfloat16 \
  --host "$(ip route get 1.1.1.1 | awk '/src/{print $7}')" \
  --port 30000 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.90 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --enable-piecewise-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

> 图文数据集场景需额外添加：环境变量 `export SGLANG_USE_CUDA_IPC_TRANSPORT=1`，启动命令添加 `--mm-attention-backend fa3`。

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<P_node0_ip>:30000 \
  --decode http://<D_node0_ip>:30000 \
  --policy cache_aware \
  --port 30001
```

### Qwen3.5-397B-A17B-W8A8 IFB BW1000 8x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_KV_LAYOUT_DCU_FA=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_MODELSCOPE=1
export GPU_MAX_HW_QUEUES=4

sglang serve \
  --numa-node 3 1 1 0 7 5 5 4 \
  --model-path hygon/Qwen3.5-397B-A17B-W8A8 \
  --trust-remote-code \
  --tp-size 8 \
  --pp-size 1 \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e5m2 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.90 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --enable-piecewise-cuda-graph \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

> 图文数据集场景使用 `--mm-attention-backend fa3`；纯文本场景可移除该参数。

## API 调用

### Qwen3.5-35B-A3B

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

### Qwen3.5-397B-A17B IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Qwen3.5-397B-A17B-Channel-FP8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/Qwen3.5-397B-A17B-Channel-FP8",
    "messages": [
      {"role": "system", "content": "你是一个专业的编程助手。"},
      {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
    ],
    "max_tokens": 128
  }'
```

### Qwen3.5-397B-A17B PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 默认端口为 `30000`，
若与 P node 0 部署在同一机器上需指定其他端口（示例中为 `30001`）。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Qwen3.5-397B-A17B-Channel-FP8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://<router_ip>:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/Qwen3.5-397B-A17B-Channel-FP8",
    "messages": [
      {"role": "system", "content": "你是一个专业的编程助手。"},
      {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
    ],
    "max_tokens": 128
  }'
```

