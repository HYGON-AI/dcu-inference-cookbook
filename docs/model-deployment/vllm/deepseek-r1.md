# DeepSeek-R1 on vLLM

## 模型简介

DeepSeek-R1 是 DeepSeek 推出的推理强化模型，面向复杂推理、数学与代码场景，通过强化学习提升模型的思维链推理能力。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-R1-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-R1-0528-W4A8-V2) | W4A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-R1-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-R1-0528-W4A8-V2) | W4A8 | 0.15 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-ifb-bw1000-8x-vllm-015) |

## 启动命令

### DeepSeek-R1-W4A8 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_CUSTOM_CACHE=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_FLASH_MLA=1
export VLLM_USE_CAT_MLA=1

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  -q slimquant_w4a8_marlin \
  -tp 8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e5m2 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_w4a8_marlin"}'
```

### DeepSeek-R1-W4A8 IFB BW1000 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_CUSTOM_CACHE=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_FLASH_MLA=1

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  -q slimquant_w4a8_marlin \
  -tp 8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e5m2 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_w4a8_marlin"}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-R1-0528-W4A8-V2",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```bash
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/DeepSeek-R1-0528-W4A8-V2",
  "messages": [
    {"role": "user", "content": "你好，请用Python写一个贪吃蛇的游戏脚本"}
  ],
  "max_tokens": 1500,
  "temperature": 0.0
  }'
```
