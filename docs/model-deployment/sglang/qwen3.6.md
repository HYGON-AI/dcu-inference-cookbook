# Qwen3.6 on SGLang

## 模型简介

Qwen3.6 模型相较于 Qwen3.5 模型，**在智能体编程能力、推理速度、架构效率及多模态表现上实现了全面升级。**

## 模型列表


| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.6-27B](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen36-27b-ifb-bw1100-2x-sglang-0510) |
| [Qwen/Qwen3.6-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-bw1100-2x-sglang-0510) |


## 启动命令

### Qwen3.6-27B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1

sglang serve --model-path Qwen/Qwen3.6-27B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser qwen3 \
    --trust-remote-code
```

### Qwen3.6-35B-A3B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_MARLIN_W16A16_MOE=1

sglang serve --model-path Qwen/Qwen3.6-35B-A3B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --enable-piecewise-cuda-graph \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64 \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --chunked-prefill-size -1
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.6-35B-A3B",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```

