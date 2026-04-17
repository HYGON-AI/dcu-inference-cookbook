# Kimi-K2 on vLLM

## 模型简介

Kimi-K2 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| Kimi-K2-1.5B | 1.5B | 128K | 1x DCU 64GB |
| Kimi-K2-7B | 7B | 128K | 1x DCU 64GB |
| Kimi-K2-13B | 13B | 128K | 1x DCU 64GB |
| Kimi-K2-72B | 72B | 128K | 2x DCU 128GB TP / 4x DCU 64GB TP |

## 启动命令

### Kimi-K2-7B（单卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-K2-7B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

### Kimi-K2-72B（四卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model moonshotai/Kimi-K2-72B \
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
    model="moonshotai/Kimi-K2-7B",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- Kimi-K2 原生支持 bf16
- 超长上下文（>32K）场景 KV Cache 占用大，建议适当降低 `--max-model-len`
- 72B 模型需要 4x DCU 64GB 或 2x DCU 128GB
