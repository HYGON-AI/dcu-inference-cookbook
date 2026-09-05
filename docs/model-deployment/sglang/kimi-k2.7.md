# Kimi-K2.7 on SGLang

## 模型简介

Kimi-K2.7-Code 的 SGLang 部署说明。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.7-Code](https://www.modelscope.cn/models/moonshotai/Kimi-K2.7-Code) | INT4 W4A16 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#kimi-k27-code-ifb-bw1100-8x-sglang-0512) |
|  | INT4 W4A16 | [0.5.12](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#kimi-k27-code-ifb-bw1000-16x-sglang-0512) |

## 启动命令

### Kimi-K2.7-Code IFB BW1100 8x SGLang 0.5.12

```bash
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_ROCM_USE_AITER_MOE=1
export GPU_MAX_HW_QUEUES=4
export SGLANG_AITER_W4A16_PREFILL_MOEC=1

sglang serve \
  --model-path moonshotai/Kimi-K2.7-Code \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --max-running-requests 64 \
  --context-length 81920 \
  --chunked-prefill-size 8192 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla
```

### Kimi-K2.7-Code IFB BW1000 16x SGLang 0.5.12

#### node 0

```bash
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_SOCKET_IFNAME=ens47f0np0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_ROCM_USE_AITER_MOE=1
export GPU_MAX_HW_QUEUES=4
export SGLANG_AITER_W4A16_PREFILL_MOEC=1

sglang serve \
  --model-path moonshotai/Kimi-K2.7-Code \
  --kv-cache-dtype fp8_e5m2 \
  --host <node0_ip> \
  --port <port0> \
  --trust-remote-code \
  --page-size 64 \
  --max-running-requests 64 \
  --context-length 81920 \
  --chunked-prefill-size 8192 \
  --dist-init-addr <node0_ip>:5123 \
  --nnodes 2 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 2 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla
```

#### node 1

```bash
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_SOCKET_IFNAME=ens47f0np0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=600
export SGLANG_ROCM_USE_AITER_MOE=1
export GPU_MAX_HW_QUEUES=4
export SGLANG_AITER_W4A16_PREFILL_MOEC=1

sglang serve \
  --model-path moonshotai/Kimi-K2.7-Code \
  --kv-cache-dtype fp8_e5m2 \
  --host <node1_ip> \
  --trust-remote-code \
  --page-size 64 \
  --max-running-requests 64 \
  --context-length 81920 \
  --chunked-prefill-size 8192 \
  --dist-init-addr <node0_ip>:5123 \
  --nnodes 2 \
  --node-rank 1 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 2 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.7-Code",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a Python hello-world program."},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K2.7-Code","messages":[{"role":"user","content":"Write a Python hello-world program."}],"max_tokens":128}'
```
