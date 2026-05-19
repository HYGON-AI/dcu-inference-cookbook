# Qwen3 on vLLM

## 模型简介

Qwen3 是阿里通义千问第三代大语言模型，支持 0.6B ~ 235B 多种参数规模，原生支持思考模式（thinking mode）和工具调用。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-0.6B](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)           | BF16 | 0.15 | BW1000  | 1 | IFB | [**`>_`**](#qwen3-06b-ifb-bw1000-1x-vllm-015)           |
| [Qwen/Qwen3-1.7B](https://www.modelscope.cn/models/Qwen/Qwen3-1.7B)           | BF16 | 0.15 | BW1000  | 1 | IFB | - |
| [Qwen/Qwen3-4B](https://www.modelscope.cn/models/Qwen/Qwen3-4B)               | BF16 | 0.15 | BW1000  | 1 | IFB | - |
| [Qwen/Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)               | BF16 | 0.15 | BW1000  | 1 | IFB | - |
| [Qwen/Qwen3-14B](https://www.modelscope.cn/models/Qwen/Qwen3-14B)             | BF16 | 0.15 | BW1000  | 1 | IFB | - |
| [Qwen/Qwen3-32B](https://www.modelscope.cn/models/Qwen/Qwen3-32B)             | BF16 | 0.15 | BW1100  | 1 | IFB | - |
| [Qwen/Qwen3-30B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3-30B-A3B)     | BF16 | 0.15 | BW1000  | 2 | IFB | [**`>_`**](#qwen3-30b-a3b-ifb-bw1000-2x-vllm-015)       |
|                                                                               | BF16 | 0.15 | BW1100  | 1 | IFB | - |
| [Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B) | BF16 | 0.15 | BW1100  | 4 | IFB | - |
|                                                                               | BF16 | 0.15 | BW1000  | 8 | IFB | - |
| [Qwen/Qwen3-235B-A22B-Instruct-2507](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B-Instruct-2507) | BF16 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1100-8x-vllm-015) |
|                                                                               | BF16 | 0.15 | BW1000 | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-bw1000-8x-vllm-015) |
|                                                                               | BF16 | 0.15 | K100_AI | 8 | IFB | [**`>_`**](#qwen3-235b-a22b-instruct-2507-ifb-k100_ai-8x-vllm-015) |
| [Qwen/Qwen3-235B-A22B-FP8-Channel](https://www.modelscope.cn/models/hygon/Qwen3-235B-A22B-W8A8) | FP8 | 0.15 | BW1100 | 4 | IFB | [**`>_`**](#qwen3-235b-a22b-fp8-channelwise-ifb-bw1100-4x-vllm-015) |

## 启动命令

### Qwen3-0.6B IFB BW1000 1x vLLM 0.15

```bash
export VLLM_USE_PIECEWISE=1

vllm serve Qwen/Qwen3-0.6B \
    -tp 1 \
    --trust-remote-code \
    --disable-log-requests \
    --dtype bfloat16 \
    --disable-cascade-attn \
    -cc '{"pass_config": {"fuse_act_quant": false}, "custom_ops": ["all"]}'
```

### Qwen3-30B-A3B IFB BW1000 2x vLLM 0.15

```bash
export VLLM_USE_PD_SPLIT=0

vllm serve Qwen/Qwen3-30B-A3B \
    -tp 2 \
    --trust-remote-code \
    --dtype float16 \
    --disable-cascade-attn \
    -cc '{"pass_config": {"fuse_act_quant": false}, "custom_ops": ["all"]}'
```

### Qwen3-235B-A22B-Instruct-2507 IFB BW1100 8x vLLM 0.15

```bash 
export VLLM_RANK0_NUMA=0   ##按照实际的
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

vllm serve /Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dtype float16 \
    --trust-remote-code \
    -tp 8 \
    --disable-cascade-attn 
```

### Qwen3-235B-A22B-Instruct-2507 IFB BW1000 8x vLLM 0.15

```bash
export VLLM_RANK0_NUMA=0  #按照实际的
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=3
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

vllm serve /Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dtype float16 \
    --trust-remote-code \
    -tp 8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 40960 \
    --disable-cascade-attn
```

### Qwen3-235B-A22B-Instruct-2507 IFB K100_AI 8x vLLM 0.15

```bash
export VLLM_RANK0_NUMA=0  #按照实际的
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1

vllm serve /Qwen/Qwen3-235B-A22B-Instruct-2507 \
    --dtype float16 \
    --trust-remote-code \
    -tp 8 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 40960 \
    --disable-cascade-attn
```

### Qwen3-235B-A22B-FP8-Channelwise IFB BW1100 4x vLLM 0.15

```bash
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_OPT_RESHAPE_AND_CACHE=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1

vllm serve Qwen/Qwen3-235B-A22B-FP8-Channel \
    -tp 4 \
    --trust-remote-code \
    --dtype bfloat16 \
    -q slimquant_marlin \
    --disable-cascade-attn \
    -cc '{"pass_config": {"fuse_act_quant": false}, "custom_ops": ["all"]}'
```
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
