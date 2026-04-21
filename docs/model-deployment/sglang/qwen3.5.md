# Qwen3.5 on SGLang

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| Qwen3.5-7B | 7B | 128K | 1x DCU 64GB |
| Qwen3.5-14B | 14B | 128K | 1x DCU 64GB |
| Qwen3.5-32B | 32B | 128K | 1x DCU 128GB / 2x DCU TP |
| Qwen3.5-72B | 72B | 128K | 2x DCU 128GB TP / 4x DCU 64GB TP |
| Qwen3.5-397B| 397B| 128K | 4x DCU 128GB TP / 8x DCU 128GB TP |

## 镜像

```txt
10.16.6.35:5000/jenkins/model_test_env/sglang:daily-20260416-1454
```

## 启动命令

### Qwen3.5-7B（单卡）

```bash
export SGLANG_ENABLE_SPEC_V2=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1          # FP8模型开启
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_KV_LAYOUT_DCU_FA=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-7B \
    --tp-size 1 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port $port \
    --attention-backend fa3 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --chunked-prefill-size -1
```

### Qwen3.5-72B（四卡）

```bash
export SGLANG_ENABLE_SPEC_V2=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1          # FP8模型开启
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_KV_LAYOUT_DCU_FA=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-72B \
    --tp-size 4 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port $port \
    --attention-backend fa3 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --chunked-prefill-size -1
```

### Qwen3.5-397B（四卡）

```bash
export SGLANG_ENABLE_SPEC_V2=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1          # FP8模型开启
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_KV_LAYOUT_DCU_FA=1

python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-397B \
    --tp-size 4 \
    --trust-remote-code \
    --dtype bfloat16 \
    --port $port \
    --attention-backend fa3 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --chunked-prefill-size -1
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-7B",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- 与 Qwen3 共享相同架构，DCU 兼容性一致
- 72B 模型建议至少 4x DCU 64GB 或 2x DCU 128GB
- SGLang 结构化生成对代码生成场景特别有用
