# Qwen2 on vLLM

## 模型简介

Qwen2 是阿里通义千问开源大语言模型系列，支持多种参数规模。本页提供 Qwen2-0.5B-Instruct、Qwen2-1.5B-Instruct、Qwen2-7B-Instruct、Qwen2-57B-A14B-Instruct、Qwen2-72B-Instruct 在 DCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2-0.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-0.5B-Instruct) | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen2-05b-instruct-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen2-05b-instruct-ifb-bw1000-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen2-05b-instruct-ifb-k100_ai-1x-vllm-018) |
| [Qwen/Qwen2-1.5B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-1.5B-Instruct) | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen2-15b-instruct-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen2-15b-instruct-ifb-bw1000-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen2-15b-instruct-ifb-k100_ai-1x-vllm-018) |
| [Qwen/Qwen2-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-7B-Instruct) | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen2-7b-instruct-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen2-7b-instruct-ifb-bw1000-1x-vllm-018) |
|                                                                               | BF16      | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen2-7b-instruct-ifb-k100_ai-1x-vllm-018) |
| [Qwen/Qwen2-57B-A14B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-57B-A14B-Instruct) | BF16 | 0.18 | BW1100  | 2 | IFB | [**`>_`**](#qwen2-57b-a14b-instruct-ifb-bw1100-2x-vllm-018) |
|                                                                               | BF16      | 0.18 | BW1000  | 4 | IFB | [**`>_`**](#qwen2-57b-a14b-instruct-ifb-bw1000-4x-vllm-018) |
|                                                                               | BF16      | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen2-57b-a14b-instruct-ifb-k100_ai-4x-vllm-018) |
| [Qwen/Qwen2-72B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2-72B-Instruct) | BF16 | 0.18 | BW1100  | 2 | IFB | [**`>_`**](#qwen2-72b-instruct-ifb-bw1100-2x-vllm-018) |
|                                                                               | BF16      | 0.18 | BW1000  | 4 | IFB | [**`>_`**](#qwen2-72b-instruct-ifb-bw1000-4x-vllm-018) |
|                                                                               | BF16      | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen2-72b-instruct-ifb-k100_ai-4x-vllm-018) |

## 启动命令

### Qwen2-0.5B-Instruct IFB BW1100 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-0.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-0.5B-Instruct IFB BW1000 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-0.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-0.5B-Instruct IFB K100_AI 1x vLLM 0.18
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2-0.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-1.5B-Instruct IFB BW1100 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-1.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-1.5B-Instruct IFB BW1000 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-1.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-1.5B-Instruct IFB K100_AI 1x vLLM 0.18
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2-1.5B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-7B-Instruct IFB BW1100 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-7B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-7B-Instruct IFB BW1000 1x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-7B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-7B-Instruct IFB K100_AI 1x vLLM 0.18
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2-7B-Instruct \
  -tp 1 \
  --trust-remote-code
```

### Qwen2-57B-A14B-Instruct IFB BW1100 2x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-57B-A14B-Instruct \
  -tp 2 \
  --trust-remote-code
```

### Qwen2-57B-A14B-Instruct IFB BW1000 4x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-57B-A14B-Instruct \
  -tp 4 \
  --trust-remote-code
```

### Qwen2-57B-A14B-Instruct IFB K100_AI 4x vLLM 0.18
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2-57B-A14B-Instruct \
  -tp 4 \
  --trust-remote-code
```

### Qwen2-72B-Instruct IFB BW1100 2x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-72B-Instruct \
  -tp 2 \
  --trust-remote-code
```

### Qwen2-72B-Instruct IFB BW1000 4x vLLM 0.18
```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve Qwen/Qwen2-72B-Instruct \
  -tp 4 \
  --trust-remote-code
```

### Qwen2-72B-Instruct IFB K100_AI 4x vLLM 0.18
```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen2-72B-Instruct \
  -tp 4 \
  --trust-remote-code
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2-7B-Instruct",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-7B-Instruct",
    "messages": [
      {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    "max_tokens": 128
  }'
```
