# Kimi-K2.5 on SGLang

## 模型简介

Kimi K2.5 是一个开源的原生多模态智能体模型，在 Kimi-K2-Base 基础上通过约 15 万亿混合视觉与文本 token 的持续预训练构建而成。它无缝融合了视觉与语言理解能力，并具备先进的智能体功能，包括即时模式与思考模式，以及对话式与智能体式两种范式。

## 模型列表

| 模型         | 参数量  | 上下文  | 推荐硬件                             |
| ---------- | ---- | ---- | -------------------------------- |
| Kimi-K2.5   | 1T   | 256K | 8x DCU 144GB TP                    |

## 启动命令

### Kimi-K2.5-W4A16

#### IFB【tp8】

```
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1

python3 -m sglang.launch_server \
  --model-path /module/Kimi-K2.5 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr $(hostname -I | awk '{print $1}'):5001 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --attention-backend dcu_mla \
  --enable-torch-compile \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 65536
```

#### PD分离

##### P节点【tp8】：

```
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=60
export SGLANG_ROCM_USE_AITER_MOE=1

python3 -m sglang.launch_server \
  --model-path /module/Kimi-K2.5 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr $(hostname -I | awk '{print $1}'):5001 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --attention-backend dcu_mla \
  --numa-node 0 0 1 1 2 2 3 3 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 81920 \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill
```

##### D节点【tp8】

```
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=60
export SGLANG_ROCM_USE_AITER_MOE=1

python3 -m sglang.launch_server \
  --model-path /module/Kimi-K2.5 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr $(hostname -I | awk '{print $1}'):5001 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --attention-backend dcu_mla \
  --numa-node 0 0 1 1 2 2 3 3 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 81920 \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9 \
  --disaggregation-mode decode
```

##### SGLang-Router

```
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://10.16.6.62:30000 --decode http://10.16.6.69:30000 --policy round_robin --port 30020
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.16.6.62:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Kimi-k2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

## curl 调用
```
curl http://10.16.6.62:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Kimi-k2.5",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```