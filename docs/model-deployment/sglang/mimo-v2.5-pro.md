# MiMo-V2.5-Pro on SGLang

## 模型简介

MiMo-V2.5-Pro 是小米推出的大规模 MoE（混合专家）语言模型，总参数量1.02T，激活参数量为 42B，支持 1M 超长上下文，采用混合注意力架构和三层多词元预测，可显著提升 decode 吞吐。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 16x | IFB | [**`>_`**](#mimo-v25-pro-channel-fp8-w8a8-ifb-bw1100-16x-sglang-0512) |

## 启动命令

### MiMo-V2.5-Pro-Channel-FP8-w8a8 IFB BW1100 16x SGLang 0.5.12

双机部署（2 nodes），每机 8 卡，ep=16 跨双机，dp=2。

#### Node 0

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_SOCKET_IFNAME=ibs73
export GLOO_SOCKET_IFNAME=ibs73
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_AITER_ASM_MOE_USE_SHUFFLE=1
export SGLANG_NUMA_BIND_V2=1
export NCCL_TOPO_MAPPING_FILE=topo_mapping.xml
export TOPO_DUMP_FILE=topo_dump.xml
export NCCL_GRAPH_DUMP_FILE=graph_dump.xml
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export SGLANG_USE_AITER_AR=1
export SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_MAX_NUM_CONTEXTS=72
export ROCSHMEM_HEAP_SIZE=3737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export ROCSHMEM_TOPO_FILE_FORCE=/home/sglang-ws/620/tests_mpi/topo.config

sglang serve \
    --model-path hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 16 \
    --ep-size 16 \
    --attn-cp-size 1 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --decode-log-interval 10 \
    --page-size 128 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.84 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 1048576 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dist-init-addr <node0_ip>:12015 \
    --nnodes 2 \
    --node-rank 0 \
    --tokenizer-worker-num 64 \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --numa-node 0 0 0 0 1 1 1 1
```

#### Node 1

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export NCCL_SOCKET_IFNAME=ibs73
export GLOO_SOCKET_IFNAME=ibs73
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_AITER_ASM_MOE_USE_SHUFFLE=1
export SGLANG_NUMA_BIND_V2=1
export NCCL_TOPO_MAPPING_FILE=topo_mapping.xml
export TOPO_DUMP_FILE=topo_dump.xml
export NCCL_GRAPH_DUMP_FILE=graph_dump.xml
export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export SGLANG_USE_AITER_AR=1
export SGLANG_ENABLE_ATTN_TP_USE_AITER_CUSTOM_COMM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_MAX_NUM_CONTEXTS=72
export ROCSHMEM_HEAP_SIZE=3737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export ROCSHMEM_TOPO_FILE_FORCE=/home/sglang-ws/620/tests_mpi/topo.config

sglang serve \
    --model-path hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 16 \
    --ep-size 16 \
    --attn-cp-size 1 \
    --enable-dp-attention \
    --moe-dense-tp-size 1 \
    --enable-dp-lm-head \
    --page-size 128 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.84 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 1048576 \
    --attention-backend fa3 \
    --kv-cache-dtype fp8_e4m3 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --dist-init-addr <node0_ip>:12015 \
    --nnodes 2 \
    --node-rank 1 \
    --tokenizer-worker-num 64 \
    --model-loader-extra-config '{"enable_multithread_load": "true","num_threads": 64}' \
    --numa-node 0 0 0 0 1 1 1 1
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8",
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
curl -X 'POST' \
    'http://localhost:30000/v1/chat/completions' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "hygon/MiMo-V2.5-Pro-Channel-FP8-w8a8",
    "messages": [
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"}
    ],
    "max_tokens": 128,
    "temperature": 0
    }'
```