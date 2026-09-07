# Kimi-K2.6 on SGLang

## 模型简介

Kimi K2.6 是一个开源的原生多模态智能体模型，在长周期编码、以编码驱动的设计、主动自主执行以及基于智能体集群的任务编排等实际能力方面取得了显著进展。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.6](https://www.modelscope.cn/models/moonshotai/Kimi-K2.6) | INT4 W4A16 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#kimi-k26-ifb-bw1100-8x-sglang-0512) |
|  | INT4 W4A16 | [0.5.12](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#kimi-k26-ifb-bw1000-16x-sglang-0512) |
|  | INT4 W4A16 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k26-ifb-bw1100-8x-sglang-0510) |
|  | INT4 W4A16 | 0.5.10 | BW1100 | 16 | 1P1D | [**`>_`**](#kimi-k26-1p1d-bw1100-16x-sglang-0510) |
|  | INT4 W4A16 | 0.5.12 | ScaleX40 | 24 | 1P1D | [**`>_`**](#kimi-k26-1p1d-bw1100-超节点ep16-24x-sglang-0512) |

## 启动命令

### Kimi-K2.6 IFB BW1100 8x SGLang 0.5.12

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_HCU_CUSTOM_ALLREDUCE=1
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

sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --nnodes 1 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla \
  --enable-torch-compile \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 65536
```

### Kimi-K2.6 IFB BW1000 16x SGLang 0.5.12

#### node 0

```bash
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_SOCKET_IFNAME=ens47f0np0
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

sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8_e5m2 \
  --host <node0_ip> \
  --port <port0> \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr <node0_ip>:<port1> \
  --nnodes 2 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 2 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.9 \
  --enable-torch-compile \
  --attention-backend hcu_mla \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 65536
```

#### node 1

```bash
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_SOCKET_IFNAME=ens47f0np0
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

sglang serve \
  --model-path moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8_e5m2 \
  --host <node1_ip> \
  --port <port0> \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr <node0_ip>:<port1> \
  --nnodes 2 \
  --node-rank 1 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 2 \
  --reasoning-parser kimi_k2 \
  --tool-call-parser kimi_k2 \
  --mem-fraction-static 0.9 \
  --enable-torch-compile \
  --attention-backend hcu_mla \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 65536
```

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

### Kimi-K2.6 1P1D BW1100 超节点ep16 24x SGLang 0.5.12

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

离线EPLB参考：[EPLB](../../optimization/static-eplb-sglang.md)。

#### P node

```
NODE_RANK=${1:-0}
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
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
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1
export AITER_ENABLE_SUPERNODE_AR=1 AITER_REBUILD=1  # 构建 aiter
export AITER_AR_TRANSPORT=fabric AITER_AR_ENABLE_REG_CAPTURE=0
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr <P_node0_ip>:5001 \
  --nnodes 2 \
  --node-rank ${NODE_RANK} \
  --dtype bfloat16 \
  --tp-size 4 \
  --pp-size 2 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla \
  --numa-node 0 5 3 2 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 81920 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --custom-all-reduce-backend aiter \
```

#### D node

```
NODE_RANK=${1:-0}
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
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
export SGLANG_USE_MARLIN_W4A16_MOE_OPT=1
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=6442450944
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256

export ROCSHMEM_IPC_MNVL=1
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1

python3 -m sglang.launch_server \
  --model-path moonshotai/Kimi-K2.6 \
  --kv-cache-dtype fp8_e4m3 \
  --host $(hostname -I | awk '{print $1}') \
  --port 30000 \
  --trust-remote-code \
  --page-size 64 \
  --dist-init-addr <P_node0_ip>:5003 \
  --nnodes 4 \
  --node-rank ${NODE_RANK} \
  --dtype bfloat16 \
  --tp-size 16 \
  --pp-size 1 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla \
  --numa-node 0 5 3 2 \
  --chunked-prefill-size -1 \
  --max-running-requests 512 \
  --context-length 81920 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode decode \
  --ep-size 16 \
  --dp-size 16 \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path lightseekorg/kimi-k2.6-eagle3-mla \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --init-expert-location <path_to_eplb_trace_pt> \
  --ep-num-redundant-experts 16
```

## Router

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