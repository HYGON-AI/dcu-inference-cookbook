# Qwen3.6 on vLLM

## 模型简介

Qwen3.6 模型相较于 Qwen3.5 模型，**在智能体编程能力、推理速度、架构效率及多模态表现上实现了全面升级。**

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.6-27B](https://www.modelscope.cn/models/Qwen/Qwen3.6-27B) | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen36-27b-ifb-bw1100-1x-vllm-018) |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen36-27b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                  | BF16 | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen36-27b-ifb-bw1000-2x-vllm-018) |
|  | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen36-27b-ifb-bw1000-2x-vllm-018-hotfix) |
|                                                                  | BF16 | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen36-27b-ifb-k100_ai-2x-vllm-018) |
|  | BF16 | 0.18-hotfix | K100_AI | 2 | IFB | [**`>_`**](#qwen36-27b-ifb-k100_ai-2x-vllm-018-hotfix) |
| [Qwen/Qwen3.6-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B) | BF16 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-bw1100-1x-vllm-018) |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                         | BF16 | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-bw1000-2x-vllm-018) |
|  | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-bw1000-2x-vllm-018-hotfix) |
|                                                                         | BF16 | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-k100_ai-2x-vllm-018) |
|  | BF16 | 0.18-hotfix | K100_AI | 2 | IFB | [**`>_`**](#qwen36-35b-a3b-ifb-k100_ai-2x-vllm-018-hotfix) |

## 启动命令

### Qwen3.6-27B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.6-27B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-27B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.6-27B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.6-27B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.6-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-27B IFB BW1000 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.6-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.6-27B IFB K100_AI 2x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.6-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-27B IFB K100_AI 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.6-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.6-35B-A3B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-35B-A3B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.6-35B-A3B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-35B-A3B IFB BW1000 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.6-35B-A3B IFB K100_AI 2x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.6-35B-A3B IFB K100_AI 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.6-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.6-35B-A3B",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "Qwen/Qwen3.6-35B-A3B",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
