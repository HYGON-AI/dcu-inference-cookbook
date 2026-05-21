# Kimi-K2 on SGLang

## 模型简介

Kimi-K2 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2-Instruct](https://www.modelscope.cn/models/moonshotai/Kimi-K2-Instruct) | FP8 W8A8 | 0.5.10 | BW1100 | 16x | IFB | [**\`>_\`**](#kimi-k2-instruct-ifb-bw1100-16x-sglang-0510) |

## 启动命令

### Kimi-K2-Instruct IFB BW1100 16x SGLang 0.5.10

```bash
sglang serve moonshotai/Kimi-K2-Instruct \
    --tp-size 16 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```
