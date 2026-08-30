# QwQ on SGLang

## 模型简介

QwQ 是 Qwen 系列的推理模型。本页提供 QwQ-32B 在 HCU 平台上的 SGLang 部署方案。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/QwQ-32B](https://www.modelscope.cn/models/Qwen/QwQ-32B) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#qwq-32b-ifb-bw1100-2x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwq-32b-ifb-bw1000-4x-sglang-0512) |
|  | BF16 | [0.5.12](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#qwq-32b-ifb-k100_ai-4x-sglang-0512) |

## 启动命令

### QwQ-32B IFB BW1100 2x SGLang 0.5.12

```bash
export SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING=1

sglang serve \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --model-path Qwen/QwQ-32B \
  --tp-size 2 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --reasoning-parser deepseek-r1 \
  --kv-cache-dtype fp8_e4m3
```

### QwQ-32B IFB BW1000 4x SGLang 0.5.12

```bash
export SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING=1

sglang serve \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --model-path Qwen/QwQ-32B \
  --tp-size 2 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --reasoning-parser deepseek-r1
```

### QwQ-32B IFB K100_AI 4x SGLang 0.5.12

```bash
export SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING=1

sglang serve \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 256 \
  --model-path Qwen/QwQ-32B \
  --tp-size 4 \
  --pp-size 1 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --reasoning-parser deepseek-r1 \
  --disable-custom-all-reduce
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[...],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/QwQ-32B", "messages": [...], "max_tokens": 128}'
```
