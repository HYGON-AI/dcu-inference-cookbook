# Qwen3 on vLLM

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式（thinking mode）和工具调用。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)           | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-06b-ifb-bw1100-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-06b-ifb-bw1000-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-06b-ifb-k100_ai-1x-vllm-018)          |
| [Qwen/Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B)           | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-17b-ifb-bw1100-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-17b-ifb-bw1000-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-17b-ifb-k100_ai-1x-vllm-018)          |
| [Qwen/Qwen3-4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B)               | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-4b-ifb-bw1100-1x-vllm-018)            |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-4b-ifb-bw1000-1x-vllm-018)            |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-4b-ifb-k100_ai-1x-vllm-018)           |
| [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)               | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-bw1100-1x-vllm-018)            |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-bw1000-1x-vllm-018)            |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-8b-ifb-k100_ai-1x-vllm-018)           |
| [Qwen/Qwen3-14B](https://www.modelscope.cn/models/Qwen/Qwen3-14B)             | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-bw1100-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-bw1000-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen3-14b-ifb-k100_ai-1x-vllm-018)          |
| [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)             | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-32b-ifb-bw1100-1x-vllm-018)           |
|                                                                               | BF16 | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen3-32b-ifb-bw1000-2x-vllm-018)           |
|                                                                               | BF16 | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen3-32b-ifb-k100_ai-2x-vllm-018)          |
| [Qwen/Qwen3-30B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)     | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen3-30b-a3b-ifb-bw1100-1x-vllm-018)       |
|                                                                               | BF16 | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen3-30b-a3b-ifb-bw1000-2x-vllm-018)       |
|                                                                               | BF16 | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen3-30b-a3b-ifb-k100_ai-2x-vllm-018)      |
| [Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) | BF16 | 0.18 | BW1100  | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-bw1100-4x-vllm-018)     |
|                                                                               | BF16 | 0.18 | BW1000  | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-bw1000-8x-vllm-018)     |
|                                                                               | BF16 | 0.18 | K100_AI | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-ifb-k100_ai-8x-vllm-018)    |
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507) | BF16 | 0.18 | BW1100 | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1100-4x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000 | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1000-8x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-k100_ai-8x-vllm-018) |
| [Qwen/Qwen3-235B-A22B-FP8-Channel](https://www.modelscope.cn/models/hygon/Qwen3-235B-A22B-W8A8) | FP8 | 0.18 | BW1100 | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-fp8-channelwise-ifb-bw1100-4x-vllm-018) |
|                                                                               | FP8 | 0.18 | BW1000 | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-fp8-channelwise-ifb-bw1000-4x-vllm-018) |
|                                                                               | FP8 | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-fp8-channelwise-ifb-k100_ai-4x-vllm-018) |

## 启动命令

### Qwen3-0.6B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_NUMA_BIND=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve Qwen/Qwen3-0.6B \
    -tp 1 \
    --trust-remote-code \

    --dtype bfloat16 \

    -cc '{"pass_config": {"fuse_act_quant": false}, "custom_ops": ["all"]}'
```

### Qwen3-0.6B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-0.6B IFB K100_AI 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-1.7B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-1.7B IFB BW1000 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-1.7B IFB K100_AI 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-4B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_NUMA_BIND=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve Qwen/Qwen3-0.6B \
    -tp 1 \
    --trust-remote-code \

    --dtype bfloat16 \

    --max-num-batched-tokens 10240 \
    --kv-cache-dtype fp8_e5m2 \
    -cc '{"pass_config": {"fuse_act_quant": false}, "custom_ops": ["all"]}'
```

### Qwen3-4B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-4B IFB K100_AI 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-8B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-8B IFB BW1000 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-8B IFB K100_AI 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-14B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-14B IFB BW1000 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-14B IFB K100_AI 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-32B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-32B IFB BW1000 2x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-32B IFB K100_AI 2x vLLM 0.18

<!-- TODO: 启动命令待补充 -->


### Qwen3-30B-A3B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_NUMA_BIND=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve Qwen3/Qwen3-30B-A3B-Instruct-2507 \
    -tp 2 \
    --trust-remote-code \
    --max-num-batched-tokens 10240 \
```

### Qwen3-30B-A3B IFB BW1100 1x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-30B-A3B IFB K100_AI 2x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B IFB BW1100 4x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B IFB BW1000 8x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B IFB K100_AI 8x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B-Instruct-2507 IFB BW1100 4x vLLM 0.18
```bash
export VLLM_NUMA_BIND=1
export VLLM_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve Qwen3/Qwen3-235B-A22B-Instruct-2507 \
    -tp 2 \
    --trust-remote-code \
    --max-num-batched-tokens 10240 \
```

### Qwen3-235B-A22B-Instruct-2507 IFB BW1000 8x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B-Instruct-2507 IFB K100_AI 8x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B-FP8-Channelwise IFB BW1100 4x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B-FP8-Channelwise IFB BW1000 4x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

### Qwen3-235B-A22B-FP8-Channelwise IFB K100_AI 4x vLLM 0.18

<!-- TODO: 启动命令待补充 -->

## 环境变量

| 环境变量 | 值 | 对总吞吐 | 对 TPOT | 说明 |
|---------|---|---------|---------|------|
| `VLLM_USE_PD_SPLIT` | `1`（默认） | 友好 | 不友好 | PD 分离调度，prefill 独占 step，decode 被饿死 |
| `VLLM_USE_PD_SPLIT` | `0` | 不友好 | 友好 | 混合调度，decode 不被 prefill 阻塞 |
| `VLLM_USE_PIECEWISE` | `1` | 友好 | 不友好 | 分段 CUDA Graph，适配动态 batch，吞吐稳定 |
| `VLLM_USE_PIECEWISE` | `0`（默认） | 不友好 | 友好 | 整块 CUDA Graph，执行开销小，单步延迟低 |

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

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "Qwen/Qwen3-8B",
  "messages": [
    {"role": "user", "content": "解释量子计算的基本原理"}
  ],
  "max_tokens": 1024
  }'
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
