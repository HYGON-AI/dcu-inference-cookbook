# Qwen3-235B-A22B-Instruct-2507 on vLLM

## 模型简介

Qwen3-235B-A22B-Instruct-2507 是通义千问 Qwen3 系列的 MoE 指令模型，适用于复杂问答、代码、工具调用和长上下文推理等场景。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen3-235B-A22B-Instruct-2507]| FP16 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1100-8x-vllm-018) |
| [Qwen3-235B-A22B-Instruct-2507] | FP16 | 0.15 | BW1000 | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1000-8x-vllm-018) |
| [Qwen3-235B-A22B-Instruct-2507]  | FP16 | 0.15 | K100_AI | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-k100_ai-8x-vllm-018) |

## 启动命令

### Qwen3-235B-A22B-Instruct-2507 IFB BW1100 8x vLLM 0.15

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1  
export VLLM_RANK0_NUMA=0   ##按照实际的
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

model_path=/Qwen/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.9

vllm serve ${model_path} \
    --dtype ${data_type} \
    --host 0.0.0.0 \
    --port ${port} \
    --trust-remote-code \
    -tp ${tp} \
    --distributed-executor-backend mp \
    --gpu-memory-utilization ${gpu_memory} \
    --disable-cascade-attn \
    --max-model-len 61440 \
    --max-num-batched-tokens 204800 \
    --kv-cache-dtype fp8_e4m3
```

### Qwen3-235B-A22B-Instruct-2507 IFB BW1000 8x vLLM 0.15

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_RANK0_NUMA=0  #按照实际
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=3
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

model_path=/Qwen/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.95

vllm serve ${model_path} \
    --dtype ${data_type} \
    --host 0.0.0.0 \
    --port ${port} \
    --trust-remote-code \
    -tp ${tp} \
    --distributed-executor-backend mp \
    --gpu-memory-utilization ${gpu_memory} \
    --max-model-len 61440 \
    --kv-cache-dtype fp8_e5m2 \
    --disable-cascade-attn
```

### Qwen3-235B-A22B-Instruct-2507 IFB K100_AI 8x vLLM 0.15

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_RANK0_NUMA=0  #按照实际
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1

model_path=/Qwen/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.95

vllm serve ${model_path} \
    --dtype ${data_type} \
    --host 0.0.0.0 \
    --port ${port} \
    --trust-remote-code \
    -tp ${tp} \
    --distributed-executor-backend mp \
    --gpu-memory-utilization ${gpu_memory} \
    --disable-cascade-attn \
    --max-model-len 40960
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请介绍一下 Qwen3 模型的特点。"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "请介绍一下 Qwen3 模型的特点。"}
    ],
    "max_tokens": 128
  }'
```
