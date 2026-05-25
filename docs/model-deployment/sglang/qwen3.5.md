# Qwen3.5 on SGLang

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1100-2x-sglang-0510) |
| [Qwen/Qwen3.5-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B) | BF16 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1100-2x-sglang-0510) |
| [hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-fp8-w8a8-ifb-bw1100-2x-sglang-0510) |
| [Qwen/Qwen3.5-397B-A17B](https://www.modelscope.cn/models/Qwen/Qwen3.5-397B-A17B) | BF16 | 0.5.10 | BW1100 | 4 | IFB | [**`>_`**](#qwen35-397b-a17b-ifb-bw1100-4x-sglang-0510) |

## 启动命令

### Qwen3.5-27B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path Qwen/Qwen3.5-27B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --chunked-prefill-size -1
```

### Qwen3.5-35B-A3B IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path Qwen/Qwen3.5-35B-A3B \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --enable-piecewise-cuda-graph \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --mamba-scheduler-strategy extra_buffer \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --chunked-prefill-size -1
```

> NMZ 卡使用 `fp8_e4m3`，非 NMZ 卡使用 `fp8_e5m2`，请按照使用硬件情况进行配置。

### Qwen3.5-35B-A3B-Channel-FP8-w8a8 IFB BW1100 2x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
export SGLANG_USE_AITER_LINEAR_ATTN=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
    --attention-backend fa3 \
    --mm-attention-backend fa3 \
    --speculative-algorithm EAGLE \
    --enable-piecewise-cuda-graph \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --tp-size 2 --pp-size 1 \
    --page-size 64  \
    --chunked-prefill-size -1 \
    --kv-cache-dtype fp8_e4m3  \
    --trust-remote-code \
    --mamba-scheduler-strategy extra_buffer
```

### Qwen3.5-397B-A17B IFB BW1100 4x SGLang 0.5.10

```bash
export SGLANG_ENABLE_SPEC_V2=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_TOPK_SOFTMAX=1
export SGLANG_USE_CAUSAL_CONV1D=1
export SGLANG_KV_LAYOUT_DCU_FA=1
export SGLANG_USE_MODELSCOPE=1

sglang serve --model-path Qwen/Qwen3.5-397B-A17B \
    --tp-size 4 \
    --trust-remote-code \
    --dtype bfloat16 \
    --attention-backend fa3 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.90 \
    --disable-radix-cache \
    --chunked-prefill-size -1
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-35B-A3B",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-35B-A3B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```

