# Qwen3 on vLLM

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持多种参数规模，原生支持思考模式（thinking mode）和工具调用。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)               | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-bw1000-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-k100_ai-1x-vllm-018) |
| [Qwen/Qwen3-14B](https://www.modelscope.cn/models/Qwen/Qwen3-14B)             | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-bw1000-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-k100_ai-1x-vllm-018) |
| [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)             | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-32b-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen3-32b-ifb-bw1000-2x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen3-32b-ifb-k100_ai-2x-vllm-018) |
| [Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) | BF16 | 0.18 | BW1100  | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-bw1100-4x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000  | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-bw1000-8x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-k100_ai-8x-vllm-018) |

## 启动命令

### Qwen3-8B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-8B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-8B IFB K100_AI 1x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-8B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-14B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-14B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-14B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-14B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-14B IFB K100_AI 1x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-14B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-32B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-32B \
  -tp 1 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-32B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3-32B \
  -tp 2 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-32B IFB K100_AI 2x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-32B \
  -tp 2 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-235B-A22B IFB BW1100 4x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1

vllm serve Qwen/Qwen3-235B-A22B \
  -tp 4 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-235B-A22B IFB BW1000 8x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1

vllm serve Qwen/Qwen3-235B-A22B \
  -tp 8 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

### Qwen3-235B-A22B IFB K100_AI 8x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-235B-A22B \
  -tp 8 \
  --trust-remote-code \
  --disable-cascade-attn \
  --max-num-batched-tokens 10240
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",  # 替换为实际使用的模型名
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
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
