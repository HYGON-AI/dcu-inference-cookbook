# MiniMax-2.x on vLLM

## 模型简介

MiniMax-2.x 是 MiniMax 推出的大规模 MoE（混合专家）语言模型系列，总参数量 456B，激活参数约 45B，在长文本理解和生成方面表现突出。

## 模型列表

| 模型 | 总参数 | 激活参数 | 上下文 | 推荐硬件 |
|------|--------|---------|--------|---------|
| MiniMax-Text-01 | 456B | ~45B | 1M | 4x DCU 128GB TP |

## 启动命令

### MiniMax-Text-01（四卡 128GB）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model MiniMaxAI/MiniMax-Text-01 \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype bfloat16
```

### 显存不足时

```bash
# 降低上下文长度
--max-model-len 8192

# 启用 KV Cache 量化
--kv-cache-dtype int8

# 提高显存利用率
--gpu-memory-utilization 0.98
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="MiniMaxAI/MiniMax-Text-01",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "请详细分析大模型在金融领域的应用前景"},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- MoE 架构：总参数 456B，但每次推理只激活约 45B，实际显存需求低于 dense 模型
- 需要 `--trust-remote-code`
- 建议使用 4x DCU 128GB（512GB 总显存）
- 长上下文场景 KV Cache 占用大，MoE 模型尤为明显
- 如果遇到 OOM，优先降低 `--max-model-len` 或启用 `--kv-cache-dtype int8`
