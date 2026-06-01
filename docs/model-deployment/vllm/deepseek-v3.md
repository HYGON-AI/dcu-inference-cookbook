# DeepSeek-V3 on vLLM

## 模型简介

DeepSeek-V3 是由深度求索推出的基于MoE架构的高性能开源大语言模型，以低成本训练实现接近甚至媲美顶级闭源模型的推理与编程能力。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3-Channel-FP8-w8a8) | FP8 W8A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-channel-fp8-w8a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-V3-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0528-W4A8-V2) | W4A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-w4a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-V3-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0528-W4A8-V2) | W4A8 | 0.15 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-v3-w4a8-ifb-bw1000-8x-vllm-015) |
| [hygon/DeepSeek-V3-0324-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0324-Channel-INT8-w8a8) | INT8 W8A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1100-8x-vllm-015) |

## 启动命令

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export VLLM_USE_MODELSCOPE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_USE_CAT_MLA=1
export VLLM_SPEC_DECODE_EAGER=1 
export VLLM_USE_GLOBAL_CACHE13=1 
export VLLM_FUSED_MOE_CHUNK_SIZE=16384  
export VLLM_USE_LIGHTOP=1
export VLLM_USE_FLASH_ATTN_FP8=1
vllm serve hygon/DeepSeek-V3-0324-Channel-INT8-w8a8  \
  --trust-remote-code   \
  -q slimquant_marlin \
  --dtype bfloat16  \
  -tp 8   \
  --max-model-len 65536  \
  --disable-log-requests  \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --kv-cache-dtype fp8 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}'
```

### DeepSeek-V3-W4A8 IFB BW1100 8x vLLM 0.15

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

vllm serve hygon/DeepSeek-V3-0528-W4A8-V2 \
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

### DeepSeek-V3-W4A8 IFB BW1000 8x vLLM 0.15

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

vllm serve hygon/DeepSeek-V3-0528-W4A8-V2 \
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
    model="hygon/DeepSeek-V3-0324-Channel-INT8-w8a8",
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
curl http://0.0.0.0:8000/v1/completions -H "Content-Type: application/json" -d '{
"model": "hygon/DeepSeek-V3-0324-Channel-INT8-w8a8",
"prompt":"你好，请用Python写一个贪吃蛇的游戏脚本",
"temperature":0.0,
"max_tokens": 1500
}'

```
