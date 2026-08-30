# Qwen3-VL on SGLang

## 模型简介

Qwen3-VL 是阿里云推出的新一代多模态视觉语言模型，支持文本、图像和视频等多模态输入。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-VL-235B-A22B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-VL-235B-A22B-Instruct) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 4x | IFB | [**`>_`**](#qwen3-vl-235b-a22b-instruct-ifb-bw1100-4x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 8x | IFB | [**`>_`**](#qwen3-vl-235b-a22b-instruct-ifb-bw1000-8x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 8x | IFB | [**`>_`**](#qwen3-vl-235b-a22b-instruct-ifb-k100_ai-8x-sglang-0512) |
| [Qwen/Qwen3-VL-30B-A3B-Thinking](https://www.modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Thinking) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 1x | IFB | [**`>_`**](#qwen3-vl-30b-a3b-thinking-ifb-bw1100-1x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 2x | IFB | [**`>_`**](#qwen3-vl-30b-a3b-thinking-ifb-bw1000-2x-sglang-0512) |

## 启动命令

### Qwen3-VL-30B-A3B-Thinking IFB BW1100 1x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-VL-30B-A3B-Thinking \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --mm-attention-backend fa3 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 1 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --reasoning-parser qwen3-thinking \
  --tool-call-parser qwen \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-VL-30B-A3B-Thinking IFB BW1000 2x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-VL-30B-A3B-Thinking \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --mm-attention-backend fa3 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 2 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --reasoning-parser qwen3-thinking \
  --tool-call-parser qwen \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-VL-235B-A22B-Instruct IFB BW1100 4x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --numa-node 0 0 1 1 2 2 3 3 \
  --mm-attention-backend fa3 \
  --dtype float16 \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --mem-fraction-static 0.85 \
  --attention-backend fa3
```

### Qwen3-VL-235B-A22B-Instruct IFB BW1000 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --mm-attention-backend fa3 \
  --dtype float16 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chunked-prefill-size 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 8 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --mem-fraction-static 0.925 \
  --attention-backend fa3
```

### Qwen3-VL-235B-A22B-Instruct IFB K100_AI 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1

sglang serve \
  --model-path Qwen/Qwen3-VL-235B-A22B-Instruct \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --mm-attention-backend fa3 \
  --dtype float16 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 8 \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.91 \
  --attention-backend fa3
```

## API 调用

### IFB

#### Qwen3-VL-30B-A3B-Thinking

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Thinking",
    messages=[...],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-VL-30B-A3B-Thinking", "messages": [...], "max_tokens": 128}'
```

#### Qwen3-VL-235B-A22B-Instruct

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-235B-A22B-Instruct",
    messages=[{"role": "user", "content": "请描述图片内容"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-VL-235B-A22B-Instruct", "messages": [{"role": "user", "content": "请描述图片内容"}], "max_tokens": 128}'
```
