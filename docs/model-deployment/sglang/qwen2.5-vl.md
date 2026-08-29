# Qwen2.5-VL on SGLang

## 模型简介

Qwen2.5-VL 是通义千问视觉语言模型系列，覆盖多种参数规模和指令模型。本页提供 Qwen2.5-VL-32B-Instruct 在不同 HCU 平台上的 SGLang 部署方案。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2.5-VL-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1100-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1000-4x-sglang-0512) |

## 启动命令

### Qwen2.5-VL-32B-Instruct IFB BW1100 2x SGLang 0.5.12

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
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --mm-attention-backend fa3
```

### Qwen2.5-VL-32B-Instruct IFB BW1000 4x SGLang 0.5.12

#### BMZ

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
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --tool-call-parser qwen25
```

#### KME

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
  --model-path Qwen/Qwen2.5-VL-32B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --page-size 64 \
  --dtype bfloat16 \
  --disable-custom-all-reduce \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --mm-attention-backend fa3 \
  --tool-call-parser qwen25
```

## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-VL-32B-Instruct", "messages": [...], "max_tokens": 128}'
```
