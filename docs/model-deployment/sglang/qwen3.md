# Qwen3 on SGLang

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式和工具调用。

## 模型列表

| 模型 | 参数量 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|--------|---------|---------|
| Qwen3-0.6B | 0.6B | 128K | BF16 | 1x BW1000 64GB |
| Qwen3-1.7B | 1.7B | 128K | BF16 | 1x BW1000 64GB |
| Qwen3-4B | 4B | 128K | BF16 | 1x BW1000 64GB |
| Qwen3-8B | 8B | 128K | BF16 | 1x BW1000 64GB |
| Qwen3-14B | 14B | 128K | BF16 | 1x BW1000 64GB |
| Qwen3-32B | 32B | 128K | BF16 | 1x BW1100 144GB / 2x DCU TP |
| Qwen3-235B-A22B | 235B (MoE) | 128K | BF16 | 4x BW1100 144GB TP |

## 启动命令

### Qwen3-8B（单卡）

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-8B \
    --tp-size 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### Qwen3-32B（双卡）

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-32B \
    --tp-size 2 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### Qwen3-235B-A22B（MoE，四卡）

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-235B-A22B \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.90
```

## API 调用

SGLang 兼容 OpenAI API：

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

### 结构化生成

```python
import json

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "列出 Python 的主要特性"}],
    response_format={"type": "json_object"},
)
result = json.loads(response.choices[0].message.content)
```

## DCU 适配注意

- Qwen3 原生支持 bf16，在 DCU 上运行稳定
- MoE 模型激活参数 22B，实际显存需求低于同等 dense 模型
- SGLang 的 RadixAttention 对多轮对话场景优化明显
- 思考模式会生成更多 token，注意上下文长度设置
