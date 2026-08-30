# Qwen3-Next on SGLang

## 模型简介

Qwen3-Next 是 Qwen 系列的大语言模型。本页提供 Qwen3-Next-80B-A3B-Instruct 在 HCU 平台上的 SGLang 部署方案。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3-Next-80B-A3B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen3-Next-80B-A3B-Instruct) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-bw1100-4x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-bw1000-4x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#qwen3-next-80b-a3b-instruct-ifb-k100_ai-4x-sglang-0512) |

## 启动命令

### Qwen3-Next-80B-A3B-Instruct IFB BW1100 4x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Qwen3-Next-80B-A3B-Instruct IFB BW1000 4x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Qwen3-Next-80B-A3B-Instruct IFB K100_AI 4x SGLang 0.5.12

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_ROCM_USE_AITER_MOE=0

sglang serve \
  --model-path Qwen/Qwen3-Next-80B-A3B-Instruct \
  --tp-size 4 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --disable-custom-all-reduce
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3-Next-80B-A3B-Instruct",
    messages=[...],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Next-80B-A3B-Instruct", "messages": [...], "max_tokens": 128}'
```
