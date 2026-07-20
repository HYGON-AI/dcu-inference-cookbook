# MiMo-V2-Flash on SGLang

## 模型简介

MiMo-V2-Flash 是小米推出的大规模 MoE（混合专家）语言模型，总参数量309B，采用混合注意力和 256 experts + top-8 路由策略，支持 262K 超长上下文。模型原生支持 EAGLE（MTP）投机解码，可显著提升 decode 吞吐。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [XiaomiMiMo/MiMo-V2-Flash](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2-Flash) | BF16 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#mimo-v2-flash-ifb-bw1100-8x-sglang-0510) |
| [hygon/MiMo-V2-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.10](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510) |
| [hygon/MiMo-V2-Flash-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.10](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-int8-w8a8-ifb-bw1000-8x-sglang-0510) |
| [hygon/MiMo-V2-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.12 | BW1100 | 20x | 2P0.5D | [**`>_`**](#mimo-v2-flash-channel-fp8-w8a8-2p05d-bw1100-20x-sglang-0512) |

## 启动命令

### MiMo-V2-Flash IFB BW1100 8x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 4 \
    --page-size 64 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --disable-radix-cache \
    --context-length 262144 \
    --attention-backend fa3 \
    --chunked-prefill-size -1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

### MiMo-V2-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 4 \
    --page-size 64 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --disable-radix-cache \
    --context-length 262144 \
    --attention-backend fa3 \
    --chunked-prefill-size -1 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

### MiMo-V2-Flash-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.10

```bash
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-INT8-w8a8 \
    --pp-size 1 \
    --dp-size 1 \
    --tp-size 8 \
    --page-size 64 \
    --trust-remote-code \
    --mem-fraction-static 0.75 \
    --quantization slimquant_marlin \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --port 11010 \
    --attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

### MiMo-V2-Flash-Channel-FP8-w8a8 2P0.5D BW1100 20x SGLang 0.5.12

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 1

```bash
sysctl -w kernel.numa_balancing=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NUMA_BIND_V2=1
export SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 4 \
    --page-size 128 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.88 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
    --numa-node 0 0 0 0 1 1 1 1
```

#### P node 2

```bash
sysctl -w kernel.numa_balancing=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NUMA_BIND_V2=1
export SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 4 \
    --page-size 128 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.88 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disaggregation-mode prefill \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
    --disaggregation-bootstrap-port 8999 \
    --numa-node 0 0 0 0 1 1 1 1
```

#### D node

```bash
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_NUMA_BIND_V2=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 1 \
    --tp-size 4 \
    --page-size 128 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.88 \
    --max-running-requests 64 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disaggregation-mode decode \
    --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
    --numa-node 0 0 0 0 1 1 1 1
```

#### Router

Router 部署在与 P node 0 不同的机器上，使用 round_robin 策略在两个 P 节点之间轮询，并将请求转发到 D 节点。端口 `8998`、`8999` 为各 P 节点的 bootstrap 端口。

```bash
python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://<P_node0_ip>:30000 8998 \
    --prefill http://<P_node1_ip>:30000 8999 \
    --decode http://<D_node_ip>:30000 \
    --policy round_robin \
    --port $port
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2-Flash",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"},
    ],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2-Flash",
    "messages": [
      {"role": "system", "content": "你是一个专业的 AI 助手。"},
      {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```
