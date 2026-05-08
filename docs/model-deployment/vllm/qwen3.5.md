# Qwen3.5 on vLLM

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型 | 参数量 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|--------|---------|---------|
| Qwen3.5-7B | 7B | 128K | BF16 | 1x DCU 64GB |
| Qwen3.5-14B | 14B | 128K | BF16 | 1x DCU 64GB |
| Qwen3.5-32B | 32B | 128K | BF16 | 1x DCU 128GB / 2x DCU TP |
| Qwen3.5-72B | 72B | 128K | BF16 | 2x DCU 128GB TP / 4x DCU 64GB TP |

## 启动命令

### Qwen3.5-7B（单卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-7B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

### Qwen3.5-72B（四卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-72B \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-7B",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- 与 Qwen3 共享相同的架构，DCU 兼容性一致
- 72B 模型建议至少 4x DCU 64GB 或 2x DCU 128GB
- 支持思考模式和工具调用，用法同 Qwen3
