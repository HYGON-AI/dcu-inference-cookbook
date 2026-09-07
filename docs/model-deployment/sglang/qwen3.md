# Qwen3 on SGLang

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式和工具调用。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 1 | IFB | [**\`>_\`**](#qwen3-06b-ifb-bw1100-1x-sglang-0512) |
|                                                                         | BF16 | [0.5.12](../docker_images.md) | BW1000 | 1 | IFB | [**\`>_\`**](#qwen3-06b-ifb-bw1000-1x-sglang-0512) |
|                                                                         | BF16 | [0.5.12](../docker_images.md) | K100_AI | 1 | IFB | [**\`>_\`**](#qwen3-06b-ifb-k100_ai-1x-sglang-0512) |
| [Qwen/Qwen3-4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 1 | IFB | [**\`>_\`**](#qwen3-4b-ifb-bw1100-1x-sglang-0512) |
|                                                                     | BF16 | [0.5.12](../docker_images.md) | BW1000 | 1 | IFB | [**\`>_\`**](#qwen3-4b-ifb-bw1000-1x-sglang-0512) |
|                                                                     | BF16 | [0.5.12](../docker_images.md) | K100_AI | 1 | IFB | [**\`>_\`**](#qwen3-4b-ifb-k100_ai-1x-sglang-0512) |
| [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 1 | IFB | [**\`>_\`**](#qwen3-8b-ifb-bw1100-1x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 1 | IFB | [**\`>_\`**](#qwen3-8b-ifb-bw1000-1x-sglang-0512) |
|  | BF16 | 0.5.10 | BW1000 | 1x | IFB | [**\`>_\`**](#qwen3-8b-ifb-bw1000-1x-sglang-0510) |
| [Qwen/Qwen3-30B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 1x | IFB | [**\`>_\`**](#qwen3-30b-a3b-ifb-bw1100-1x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 2x | IFB | [**\`>_\`**](#qwen3-30b-a3b-ifb-bw1000-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 2x | IFB | [**\`>_\`**](#qwen3-30b-a3b-ifb-k100_ai-2x-sglang-0512) |
| [hygon/Qwen3-30B-A3B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3-30B-A3B-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 1x | IFB | [**\`>_\`**](#qwen3-30b-a3b-channel-int8-w8a8-ifb-bw1100-1x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 1x | IFB | [**\`>_\`**](#qwen3-30b-a3b-channel-int8-w8a8-ifb-bw1000-1x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | K100_AI | 1x | IFB | [**\`>_\`**](#qwen3-30b-a3b-channel-int8-w8a8-ifb-k100_ai-1x-sglang-0512) |
| [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B) | BF16 | 0.5.10 | BW1100 | 2x | IFB | [**\`>_\`**](#qwen3-32b-ifb-bw1100-2x-sglang-0510) |
| [Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) | BF16 | 0.5.10 | BW1100 | 4x | IFB | [**\`>_\`**](#qwen3-235b-a22b-ifb-bw1100-4x-sglang-0510) |

## 启动命令

### Qwen3-0.6B IFB BW1100 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-0.6B IFB BW1000 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-0.6B IFB K100_AI 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-0.6B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.85 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-4B IFB BW1100 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-4B IFB BW1000 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-4B IFB K100_AI 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-4B \
  --dtype bfloat16 \
  --tp-size 1 \
  --mem-fraction-static 0.9 \
  --trust-remote-code \
  --page-size 64 \
  --disable-cuda-graph \
  --skip-server-warmup \
  --disable-overlap-schedule \
  --attention-backend fa3 \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```

### Qwen3-8B IFB BW1100 1x SGLang 0.5.12

```bash
sglang serve \
  --model-path Qwen/Qwen3-8B \
  --tp-size 1 \
  --trust-remote-code \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen
```

### Qwen3-8B IFB BW1000 1x SGLang 0.5.12

```bash
sglang serve \
  --model-path Qwen/Qwen3-8B \
  --tp-size 1 \
  --trust-remote-code \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen
```

### Qwen3-8B IFB BW1000 1x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.85
```

### Qwen3-30B-A3B IFB BW1100 1x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype float16 \
  --kv-cache-dtype fp8_e4m3 \
  --tensor-parallel-size 1 \
  --page-size 64 \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-30B-A3B IFB BW1000 2x SGLang 0.5.12

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
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype float16 \
  --kv-cache-dtype fp8_e5m2 \
  --tensor-parallel-size 2 \
  --page-size 64 \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-30B-A3B IFB K100_AI 2x SGLang 0.5.12

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
export SGLANG_ROCM_USE_AITER_MOE=0


sglang serve \
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype float16 \
  --moe-runner-backend triton \
  --tensor-parallel-size 2 \
  --page-size 64 \
  --tool-call-parser qwen \
  --disable-custom-all-reduce \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-30B-A3B-Channel-INT8-w8a8 IFB BW1100 1x SGLang 0.5.12

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
  --model-path hygon/Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --moe-runner-backend lightop \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-30B-A3B-Channel-INT8-w8a8 IFB BW1000 1x SGLang 0.5.12

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

sglang serve \
  --model-path hygon/Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --moe-runner-backend lightop \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 1 \
  --kv-cache-dtype fp8_e5m2 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-30B-A3B-Channel-INT8-w8a8 IFB K100_AI 1x SGLang 0.5.12

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
export SGLANG_ROCM_USE_AITER_MOE=0

sglang serve \
  --model-path hygon/Qwen3-30B-A3B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --numa-node 0 0 0 0 1 1 1 1 \
  --dtype bfloat16 \
  --quantization w8a8_int8 \
  --moe-runner-backend triton \
  --tool-call-parser qwen \
  --reasoning-parser qwen3 \
  --tensor-parallel-size 1 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3
```

### Qwen3-32B IFB BW1100 2x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size 2 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.85
```

### Qwen3-235B-A22B IFB BW1100 4x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.90
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```
