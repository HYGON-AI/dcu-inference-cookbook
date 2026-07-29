# Hunyuan V3 on SGLang

## 模型简介

Hunyuan V3（Hy3）是腾讯混元团队开发的混合专家（MoE）大语言模型。该模型提供 FP8 权重，可在 HCU 平台上通过 SGLang 进行高效推理部署，并兼容 OpenAI API。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Hy3-Channel-FP8-w8a8](https://modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.12 | BW1100 | 8 | IFB | [**`>_`**](#hy3-fp8-ifb-bw1100-8x-sglang-0510) |

## 启动命令

### Hy3-FP8 IFB BW1100 8x SGLang 0.5.12

```bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_MAX_NUM_CONTEXTS=32

sglang serve \
  --model-path hygon/Hy3-Channel-FP8-w8a8 \
  --dp-size 8 \
  --tp-size 8 \
  --deepep-mode auto \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size=1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --context-length 8192 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --watchdog-timeout 36000 \
  --cuda-graph-max-bs 16 \
  --enforce-shared-experts-fusion
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache。"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/Hy3-Channel-FP8-w8a8",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 128
  }'
```
