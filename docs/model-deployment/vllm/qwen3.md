# Qwen3 on vLLM

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式（thinking mode）和工具调用。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| Qwen3-0.6B | 0.6B | 128K | 1x DCU 64GB |
| Qwen3-1.7B | 1.7B | 128K | 1x DCU 64GB |
| Qwen3-4B | 4B | 128K | 1x DCU 64GB |
| Qwen3-8B | 8B | 128K | 1x DCU 64GB |
| Qwen3-14B | 14B | 128K | 1x DCU 64GB |
| Qwen3-32B | 32B | 128K | 1x DCU 128GB / 2x DCU TP |
| Qwen3-235B-A22B | 235B (MoE) | 128K | 4x DCU 128GB TP |

## 启动命令

### Qwen3-8B（单卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

### Qwen3-32B（双卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-32B \
    --tensor-parallel-size 2 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --dtype bfloat16
```

### Qwen3-235B-A22B（MoE，四卡）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 4 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --dtype bfloat16
```

## API 调用

### 普通对话

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

### 思考模式（Thinking Mode）

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[
        {"role": "user", "content": "解方程: 3x^2 - 7x + 2 = 0"},
    ],
    max_tokens=4096,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

# 思考过程在 reasoning_content 中
message = response.choices[0].message
if hasattr(message, "reasoning_content"):
    print("=== 思考过程 ===")
    print(message.reasoning_content)
print("=== 最终回答 ===")
print(message.content)
```

### 工具调用（Function Calling）

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
            },
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

## DCU 适配注意

- Qwen3 原生支持 bf16，在 DCU 上运行稳定
- MoE 模型（235B）激活参数 22B，实际显存需求低于同等 dense 模型
- 思考模式会生成更多 token，注意 `--max-model-len` 设置
- 长上下文（>32K）会显著增加 KV Cache 显存占用
