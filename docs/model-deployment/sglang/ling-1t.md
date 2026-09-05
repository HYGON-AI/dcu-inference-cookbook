# Ling-1T on SGLang

## 模型简介

Ling-1T-FP8 是 inclusionAI 发布的 Ling 系列模型。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [inclusionAI/Ling-1T-FP8](https://www.modelscope.cn/models/inclusionAI/Ling-1T-FP8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#ling-1t-fp8-ifb-bw1100-8x-sglang-0512) |

## 启动命令

### Ling-1T-FP8 IFB BW1100 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_TORCH_PROFILER_DIR=/home/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=0
export SGLANG_USE_FP8_W8A8_MOE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

sglang serve \
  --numa-node 0 0 0 0 1 1 1 1 \
  --trust-remote-code \
  --model-path inclusionAI/Ling-1T-FP8 \
  --attention-backend fa3 \
  --page-size 64 \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --pp-size 1 \
  --mem-fraction-static 0.9
```

## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ling-1T-FP8",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 128
  }'
```
