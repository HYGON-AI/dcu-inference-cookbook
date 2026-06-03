# GLM-4.7 on vLLM

## 模型简介

GLM-4.7 是智谱 AI 推出的大语言模型，支持长上下文和高效推理。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [GLM-4.7-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-4.7-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100 | 8 | IFB | [**``>_``**](#glm-47-channel-int8-w8a8-ifb-bw1100-8x-vllm-018) |
|                                                                               | INT8 W8A8 | 0.18 | BW1000 | 8 | IFB | [**``>_``**](#glm-47-channel-int8-w8a8-ifb-bw1000-8x-vllm-018) |

## 启动命令

### GLM-4.7-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve hygon/GLM-4.7-Channel-INT8-w8a8 \
  -tp 8 \
  -q slimquant_marlin \
  --dtype bfloat16 \
  --disable-cascade-attn \
  --max-model-len 74000 \
  --max-num-batched-tokens 8192 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization slimquant_marlin \
  --compilation-config.cudagraph_mode PIECEWISE
```

### GLM-4.7-Channel-INT8-w8a8 IFB BW1000 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve hygon/GLM-4.7-Channel-INT8-w8a8 \
  -tp 8 \
  -q slimquant_marlin \
  --dtype bfloat16 \
  --disable-cascade-attn \
  --max-model-len 74000 \
  --max-num-batched-tokens 8192 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization slimquant_marlin \
  --compilation-config.cudagraph_mode PIECEWISE
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-4.7-Channel-INT8-w8a8",
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
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/GLM-4.7-Channel-INT8-w8a8",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
