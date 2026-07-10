# Qwen3 on SGLang

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式和工具调用。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B) | BF16 | 0.5.10 | BW1000 | 1x | IFB | [**\`>_\`**](#qwen3-8b-ifb-bw1000-1x-sglang-0510) |
| [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B) | BF16 | 0.5.10 | BW1100 | 2x | IFB | [**\`>_\`**](#qwen3-32b-ifb-bw1100-2x-sglang-0510) |
| [Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) | BF16 | 0.5.10 | BW1100 | 4x | IFB | [**\`>_\`**](#qwen3-235b-a22b-ifb-bw1100-4x-sglang-0510) |

## 启动命令

### Qwen3-8B IFB BW1000 1x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.85
```

### Qwen3-32B IFB BW1100 2x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size 2 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.85
```

### Qwen3-235B-A22B IFB BW1100 4x SGLang 0.5.10

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp-size 4 \
    --trust-remote-code \
    --attention-backend fa3 \
    --page-size 64 \
    --mem-fraction-static 0.90
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```
