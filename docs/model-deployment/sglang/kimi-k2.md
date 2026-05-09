# Kimi-K2 on SGLang

## 模型简介

Kimi-K2 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型 | 参数量 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|--------|---------|---------|
| Kimi-K2-1.5B | 1.5B | 128K | BF16 | 1x BW1000 64GB |
| Kimi-K2-7B | 7B | 128K | BF16 | 1x BW1000 64GB |
| Kimi-K2-13B | 13B | 128K | BF16 | 1x BW1000 64GB |
| Kimi-K2-72B | 72B | 128K | BF16 | 2x BW1100 144GB TP / 4x BW1000 64GB TP |

## 启动命令

### Kimi-K2-7B（单卡）

```bash
python -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-7B \
    --tp-size 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### Kimi-K2-72B（四卡）

```bash
python -m sglang.launch_server \
    --model-path moonshotai/Kimi-K2-72B \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

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
- 超长上下文场景 KV Cache 占用大，建议适当降低上下文长度
- SGLang 的 RadixAttention 对长文档处理场景优化明显
