# Qwen3.8 on vLLM

## 模型简介

Qwen3.8 系列模型面向长上下文推理与工具调用场景，支持 vLLM 部署。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) | BF16 | 0.21 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1100-1x-vllm-021) |
| [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) | BF16 | 0.21 | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-vllm-021) |
| [Qwen/Qwen3.8-27B](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B) | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-ifb-bw1000-2x-vllm-018-hotfix) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | W8A8 | 0.21 | BW1100 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1100-1x-vllm-021) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | W8A8 | 0.21 | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-021) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | W8A8 | 0.21 | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-vllm-021) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | W8A8 | 0.18-hotfix | BW1000 | 1 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018-hotfix) |
| [hygon/Qwen3.8-27B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.8-27B-Channel-INT8-w8a8) | W8A8 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen38-27b-channel-int8-w8a8-ifb-bw1000-2x-vllm-018-hotfix) |

## 启动命令

### Qwen3.8-27B IFB BW1100 1x vLLM 0.21

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' 
```

### Qwen3.8-27B IFB BW1000 2x vLLM 0.21

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}'
```

### Qwen3.8-27B IFB BW1000 2x vLLM 0.18-hotfix

```bash
vllm serve Qwen/Qwen3.8-27B \
  -tp 2 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x vLLM 0.21

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 2 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  -q slimquant_marlin \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18-hotfix

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.8-27B-Channel-INT8-w8a8 IFB BW1000 2x vLLM 0.18-hotfix

```bash
vllm serve hygon/Qwen3.8-27B-Channel-INT8-w8a8 \
  -tp 2 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_CUSTOM \
  --max-num-batched-tokens 10240 \
  --compilation-config '{"cudagraph_mode":"FULL","max_cudagraph_capture_size":2048}' \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.8-27B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "中国的首都是哪里？"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.8-27B",
    "max_tokens": 2048,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": [
        {"type": "text", "text": "中国的首都是哪里？"}
      ]}
    ]
  }'
```