# GLM-5 on vLLM

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
python -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-5-9B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

### GLM-5-72B（四卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model THUDM/GLM-5-72B \
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
    model="THUDM/GLM-5-9B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

### 工具调用

```python
tools = [{
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": "搜索知识库",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
            },
            "required": ["query"],
        },
    },
}]

response = client.chat.completions.create(
    model="THUDM/GLM-5-9B",
    messages=[{"role": "user", "content": "帮我查一下 DCU 的性能参数"}],
    tools=tools,
    tool_choice="auto",
)
```

## DCU 适配注意

- GLM-5 原生支持 bf16，DCU 兼容性良好
- 智谱模型通常需要 `--trust-remote-code`
- 中文场景下建议开启前缀缓存 `--enable-prefix-caching`
- 长上下文场景注意 KV Cache 显存
