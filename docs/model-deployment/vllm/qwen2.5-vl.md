# Qwen2.5-VL on vLLM

## 模型简介

Qwen2.5-VL 是阿里通义千问视觉语言模型系列，支持图像、视频与文本等多模态输入。本页提供 Qwen2.5-VL-32B-Instruct、Qwen2.5-VL-72B-Instruct 和 Qwen2.5-VL-72B-Instruct-quantized.w8a8 在 HCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2.5-VL-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) | BF16 | 0.21 | BW1100 | 1 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1100-1x-vllm-021) |
|  | BF16 | 0.21 | BW1000 | 2 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1000-2x-vllm-021) |
|  | BF16 | 0.21 | K100_AI | 4 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-k100_ai-4x-vllm-021) |
|  | BF16 | 0.18 | BW1100 | 1 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1100-1x-vllm-018) |
|  | BF16 | 0.18 | BW1000 | 2 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1000-2x-vllm-018) |
|  | BF16 | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-k100_ai-4x-vllm-018) |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1100-1x-vllm-018-hotfix) |
|  | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1000-2x-vllm-018-hotfix) |
|  | BF16 | 0.18-hotfix | K100_AI | 4 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-k100_ai-4x-vllm-018-hotfix) |
| [Qwen/Qwen2.5-VL-72B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct) | BF16 | 0.21 | BW1100 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-bw1100-4x-vllm-021) |
|  | BF16 | 0.21 | BW1000 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-bw1000-4x-vllm-021) |
|  | BF16 | 0.21 | K100_AI | 8 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-k100_ai-8x-vllm-021) |
|  | BF16 | [0.18](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-bw1100-4x-vllm-018) |
|  | BF16 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-bw1000-4x-vllm-018) |
|  | BF16 | [0.18](../docker_images.md) | K100_AI | 8 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-ifb-k100_ai-8x-vllm-018) |
| hygon/Qwen2.5-VL-72B-Instruct-quantized.w8a8| INT8 W8A8 | 0.21 | BW1100 | 1 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-quantizedw8a8-ifb-bw1100-1x-vllm-021) |
|  | INT8 W8A8 | 0.21 | BW1000 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-quantizedw8a8-ifb-bw1000-4x-vllm-021) |
|  | INT8 W8A8 | [0.18](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-quantizedw8a8-ifb-bw1100-4x-vllm-018) |
|  | INT8 W8A8 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#qwen25-vl-72b-instruct-quantizedw8a8-ifb-bw1000-4x-vllm-018) |

## 启动命令

### Qwen2.5-VL-32B-Instruct IFB BW1100 1x vLLM 0.21

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 1 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --allowed-local-media-path "${VL_DATA}" \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen2.5-VL-32B-Instruct IFB BW1000 2x vLLM 0.21

```bash
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 2 \
    --trust-remote-code \
    --max-model-len 32768 \
    --allowed-local-media-path "${VL_DATA}" \
    --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen2.5-VL-32B-Instruct IFB K100_AI 4x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_ROCM_USE_AITER=0
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 4 \
    --trust-remote-code \
    --max-model-len 32768 \
    --allowed-local-media-path "${VL_DATA}" \
    --attention-backend TRITON_ATTN
```
### Qwen2.5-VL-32B-Instruct IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 1 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --allowed-local-media-path /path-to/VL_data/ \
```

### Qwen2.5-VL-32B-Instruct IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 2 \
    --trust-remote-code \
    --allowed-local-media-path /path-to/VL_data/ \
```

### Qwen2.5-VL-32B-Instruct IFB K100_AI 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 4 \
    --trust-remote-code \
    --allowed-local-media-path /path-to/VL_data/ \
```
### Qwen2.5-VL-32B-Instruct IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 1 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --allowed-local-media-path /path-to/VL_data/ \
    --kv-cache-dtype fp8_e4m3 \
    --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen2.5-VL-32B-Instruct IFB BW1000 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 2 \
    --trust-remote-code \
    --max-model-len 32768 \
    --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen2.5-VL-32B-Instruct IFB K100_AI 4x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 4 \
    --trust-remote-code \
    --allowed-local-media-path /path-to/VL_data/ \
```

### Qwen2.5-VL-72B-Instruct IFB BW1100 4x vLLM 0.21

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  -tp 4 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen2.5-VL-72B-Instruct IFB BW1000 4x vLLM 0.21

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  -tp 4 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen2.5-VL-72B-Instruct IFB K100_AI 8x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_ROCM_USE_AITER=0
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  -tp 8 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 32768 \
  --attention-backend TRITON_ATTN
```

### Qwen2.5-VL-72B-Instruct IFB BW1100 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  -tp 4 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768
```

### Qwen2.5-VL-72B-Instruct IFB BW1000 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  -tp 4 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 32768
```

### Qwen2.5-VL-72B-Instruct IFB K100_AI 8x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --host 0.0.0.0 \
  -tp 8 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 32768
```

### Qwen2.5-VL-72B-Instruct-quantized.w8a8 IFB BW1100 1x vLLM 0.21

```bash
vllm serve hygon/Qwen2.5-VL-72B-Instruct-quantized.w8a8 \
  -tp 1 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 40960 \
  -q slimquant_marlin \
  --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen2.5-VL-72B-Instruct-quantized.w8a8 IFB BW1000 4x vLLM 0.21

```bash
vllm serve hygon/Qwen2.5-VL-72B-Instruct-quantized.w8a8 \
  -tp 4 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 40960 \
  -q slimquant_marlin \
  --attention-backend FLASH_ATTN_CUSTOM \
  --allowed-local-media-path "${VL_DATA}"
```

### Qwen2.5-VL-72B-Instruct-quantized.w8a8 IFB BW1100 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve hygon/Qwen2.5-VL-72B-Instruct-quantized.w8a8 \
  -tp 4 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 40960 \
  -q slimquant_marlin \
  --allowed-local-media-path
```

### Qwen2.5-VL-72B-Instruct-quantized.w8a8 IFB BW1000 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve hygon/Qwen2.5-VL-72B-Instruct-quantized.w8a8 \
  -tp 4 \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-model-len 40960 \
  -q slimquant_marlin \
  --allowed-local-media-path
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "描述这张图片。"},
                {"type": "image_url", "image_url": {"url": "file:///path-to/VL_data/example.jpg"}},
            ],
        }
    ],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "描述这张图片。"},
          {"type": "image_url", "image_url": {"url": "file:///path-to/VL_data/example.jpg"}}
        ]
      }
    ],
    "max_tokens": 128
  }'
```

### PD 分离

暂无。

