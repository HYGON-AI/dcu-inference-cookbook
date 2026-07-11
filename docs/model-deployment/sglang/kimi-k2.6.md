# Kimi-K2.6 on SGLang

## 模型简介

Kimi K2.6 是一个开源的原生多模态智能体模型，在长周期编码、以编码驱动的设计、主动自主执行以及基于智能体集群的任务编排等实际能力方面取得了显著进展。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.6](https://www.modelscope.cn/models/moonshotai/Kimi-K2.6) | INT4 W4A16 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k26-ifb-bw1100-8x-sglang-0510) |
|  | INT4 W4A16 | 0.5.10 | BW1100 | 16 | 1P1D | [**`>_`**](#kimi-k26-1p1d-bw1100-16x-sglang-0510) |

## 启动命令

### Kimi-K2.6 IFB BW1100 8x SGLang 0.5.10

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
export ALLREDUCE_STREAM_WITH_COMPUTE=1

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
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

### Kimi-K2.6 1P1D BW1100 16x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

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
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=60
export SGLANG_ROCM_USE_AITER_MOE=1

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
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

#### D node 0

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
export SGLANG_USE_MARLIN_W4A16_MOE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_IB_GID_INDEX=0
export SGLANG_HEALTH_CHECK_TIMEOUT=60
export SGLANG_ROCM_USE_AITER_MOE=1

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
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

#### Router

```
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://<P_node_ip>:30000 --decode http://<D_node_ip>:30000 --policy round_robin --port 30020
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2.6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.6",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```