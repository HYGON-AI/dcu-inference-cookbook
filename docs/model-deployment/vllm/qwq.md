# QwQ on vLLM

## 模型简介

QwQ 是 Qwen 系列的推理模型。本页提供 QwQ-32B 在 HCU 平台上的 vLLM 部署最佳实践。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/QwQ-32B](https://www.modelscope.cn/models/Qwen/QwQ-32B) | BF16 | [0.18](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#qwq-32b-ifb-bw1100-4x-vllm-018) |
|  | BF16 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwq-32b-ifb-bw1000-4x-vllm-018) |
|  | BF16 | [0.18](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#qwq-32b-ifb-k100_ai-4x-vllm-018) |

## 启动命令

### QwQ-32B IFB BW1100 4x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_MODELSCOPE=1

vllm serve Qwen/QwQ-32B \
  --host 0.0.0.0 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### QwQ-32B IFB BW1000 4x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_MODELSCOPE=1

vllm serve Qwen/QwQ-32B \
  --host 0.0.0.0 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### QwQ-32B IFB K100_AI 4x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_USE_MODELSCOPE=1

vllm serve Qwen/QwQ-32B \
  --host 0.0.0.0 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/QwQ-32B",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/QwQ-32B",
    "messages": [
      {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    "max_tokens": 128
  }'
```
