# GLM-5 on SGLang

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| GLM-5-9B | 9B | 128K | 1x DCU 64GB |
| GLM-5-25B | 25B | 128K | 1x DCU 128GB / 2x DCU TP |
| GLM-5-72B | 72B | 128K | 2x DCU 128GB TP / 4x DCU 64GB TP |
| GLM-5-130B | 130B | 128K | 4x DCU 128GB TP |

## 启动命令

### GLM-5-9B（单卡）

```bash
python -m sglang.launch_server \
    --model-path THUDM/GLM-5-9B \
    --tp-size 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### GLM-5-72B（四卡）

```bash
python -m sglang.launch_server \
    --model-path THUDM/GLM-5-72B \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="THUDM/GLM-5-9B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- GLM-5 原生支持 bf16，DCU 兼容性良好
- 需要 `--trust-remote-code`
- 中文场景下 RadixAttention 对重复 system prompt 优化明显
