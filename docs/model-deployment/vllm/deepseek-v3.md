# GLM-5 on vLLM

## 模型简介

DeepSeek-V3 是由深度求索推出的基于MoE架构的高性能开源大语言模型，以低成本训练实现接近甚至媲美顶级闭源模型的推理与编程能力。

## 模型列表

| 模型                           | 参数量  | 上下文  | 推荐硬件             |
|------------------------------|------|------|------------------|
| DeepSeek-V3-0324 | 671B | 128K | 8x BW1101 144GB  |

## 启动命令

### DeepSeek-V3-0324-Channel-FP8 TP8+MTP3

```
rm -rf ~/.cache
rm -rf ~/.triton
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
vllm serve /path/to/DeepSeek-V3-0324-Channel-fp8  \
        --trust-remote-code   \
        -q slimquant_marlin \
        -tp 8   \
        --dtype bfloat16  \
        --max-model-len 65536  \
        --disable-log-requests  \
        --gpu-memory-utilization 0.90 \
        --max-num-batched-tokens 16384 \
        --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
        --kv-cache-dtype fp8
        --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="/path/to/DeepSeek-V3-0324-Channel-fp8",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

## curl 示例

```bash
curl http://0.0.0.0:8000/v1/completions -H "Content-Type: application/json" -d '{
"model": "/path/to/DeepSeek-V3-0324-Channel-fp8",
"prompt":"你好，请用Python写一个贪吃蛇的游戏脚本",
"temperature":0.0,
"max_tokens": 1500
}'

```
