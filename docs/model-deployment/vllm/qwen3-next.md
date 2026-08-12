# Qwen3-Next on vLLM

## 模型简介

Qwen3-Next 是 Qwen 系列的下一代模型架构。本页提供 Qwen3-Next-80B-A3B-Instruct 在 HCU 平台上的 vLLM 部署最佳实践。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-Next-80B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct) | BF16 | [0.18](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-bw1100-4x-vllm-018) |
|                                                                                                                   | BF16 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-bw1000-4x-vllm-018) |
|                                                                                                                   | BF16 | [0.18](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-k100ai-4x-vllm-018) |

## 启动命令

### Qwen3-Next-80B-A3B-Instruct IFB BW1100 4x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1024 \
  --trust-remote-code \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### Qwen3-Next-80B-A3B-Instruct IFB BW1000 4x vLLM 0.18

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1024 \
  --trust-remote-code \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```
### Qwen3-Next-80B-A3B-Instruct IFB K100_AI 4x vLLM 0.18

```bash

vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 1024 \
  --trust-remote-code \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
