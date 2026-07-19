# Qwen3.5 on vLLM

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen3.5-4B](https://www.modelscope.cn/models/Qwen/Qwen3.5-4B)           | BF16      | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-bw1100-1x-vllm-018)          |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-bw1000-1x-vllm-018)           |
|                                                                               | BF16      | 0.18-hotfix | BW1000  | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-bw1000-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-k100_ai-1x-vllm-018)           |
|  | BF16 | 0.18-hotfix | K100_AI | 1 | IFB | [**`>_`**](#qwen35-4b-ifb-k100_ai-1x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-9B](https://www.modelscope.cn/models/Qwen/Qwen3.5-9B)           | BF16      | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-bw1100-1x-vllm-018)          |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-bw1000-1x-vllm-018)           |
|  | BF16 | 0.18-hotfix | BW1000 | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-bw1000-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-k100_ai-1x-vllm-018)           |
|  | BF16 | 0.18-hotfix | K100_AI | 1 | IFB | [**`>_`**](#qwen35-9b-ifb-k100_ai-1x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-27B](https://www.modelscope.cn/models/Qwen/Qwen3.5-27B)         | BF16      | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1100-1x-vllm-018)          |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1000-2x-vllm-018)          |
|  | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-bw1000-2x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-k100_ai-2x-vllm-018)          |
|  | BF16 | 0.18-hotfix | K100_AI | 2 | IFB | [**`>_`**](#qwen35-27b-ifb-k100_ai-2x-vllm-018-hotfix) |
| [hygon/Qwen3.5-27B-Channel-INT8-w8a8]() | INT8 W8A8 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-bw1100-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                                              | INT8 W8A8 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1000 | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018-hotfix) |
|                                                                                              | INT8 W8A8 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-k100_ai-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | K100_AI | 1 | IFB | [**`>_`**](#qwen35-27b-channel-int8-w8a8-ifb-k100_ai-1x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-27B-W8A8-FP8](https://modelscope.cn/models/hygon/Qwen3.5-27B-Channel-FP8-w8a8) | FP8 W8A8  | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-27b-w8a8-fp8-ifb-bw1100-1x-vllm-018) |
| [Qwen/Qwen3.5-35B-A3B](https://www.modelscope.cn/models/Qwen/Qwen3.5-35B-A3B) | BF16      | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1100-1x-vllm-018)     |
|  | BF16 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | BW1000  | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1000-2x-vllm-018)      |
|  | BF16 | 0.18-hotfix | BW1000 | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-bw1000-2x-vllm-018-hotfix) |
|                                                                               | BF16      | 0.18 | K100_AI | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-k100_ai-2x-vllm-018)      |
|  | BF16 | 0.18-hotfix | K100_AI | 2 | IFB | [**`>_`**](#qwen35-35b-a3b-ifb-k100_ai-2x-vllm-018-hotfix) |
| [hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-bw1100-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-bw1100-1x-vllm-018-hotfix) |
|                                                                                              | INT8 W8A8 | 0.18 | BW1000  | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1000 | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-bw1000-1x-vllm-018-hotfix) |
|                                                                                              | INT8 W8A8 | 0.18 | K100_AI | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-k100_ai-1x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | K100_AI | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-int8-w8a8-ifb-k100_ai-1x-vllm-018-hotfix) |
| [hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8)       | FP8 W8A8  | 0.18 | BW1100  | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-fp8-w8a8-ifb-bw1100-1x-vllm-018) |
|  | FP8 W8A8 | 0.18-hotfix | BW1100 | 1 | IFB | [**`>_`**](#qwen35-35b-a3b-channel-fp8-w8a8-ifb-bw1100-1x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-122B-A10B](https://www.modelscope.cn/models/Qwen/Qwen3.5-122B-A10B) | BF16   | 0.18 | BW1100  | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-bw1100-4x-vllm-018)  |
|  | BF16 | 0.18-hotfix | BW1100 | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-bw1100-4x-vllm-018-hotfix) |
|                                                                                  | BF16   | 0.18 | BW1000  | 8 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-bw1000-8x-vllm-018)   |
|  | BF16 | 0.18-hotfix | BW1000 | 8 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-bw1000-8x-vllm-018-hotfix) |
|                                                                                  | BF16   | 0.18 | K100_AI | 8 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-k100_ai-8x-vllm-018)   |
|  | BF16 | 0.18-hotfix | K100_AI | 8 | IFB | [**`>_`**](#qwen35-122b-a10b-ifb-k100_ai-8x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-122B-A10B-AWQ](https://www.modelscope.cn/models/tclf90/Qwen3.5-122B-A10B-AWQ) | AWQ | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-awq-ifb-k100_ai-4x-vllm-018) |
| [Qwen/Qwen3.5-122B-A10B-W8A8-INT8](https://www.modelscope.cn/models/Qwen/Qwen3.5-122B-A10B-W8A8-INT8) | INT8 W8A8 | 0.18 | BW1100  | 2 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-bw1100-2x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1100 | 2 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-bw1100-2x-vllm-018-hotfix) |
|                                                                                               | INT8 W8A8 | 0.18 | BW1000  | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-bw1000-4x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1000 | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-bw1000-4x-vllm-018-hotfix) |
|                                                                                               | INT8 W8A8 | 0.18 | K100_AI | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-k100_ai-4x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | K100_AI | 4 | IFB | [**`>_`**](#qwen35-122b-a10b-w8a8-int8-ifb-k100_ai-4x-vllm-018-hotfix) |
| [hygon/Qwen3.5-122B-A10B-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Qwen3.5-122B-A10B-Channel-FP8-w8a8)    | FP8 W8A8  | 0.18 | BW1100  | 2 | IFB | [**`>_`**](#qwen35-122b-a10b-channel-fp8-w8a8-ifb-bw1100-2x-vllm-018) |
|  | FP8 W8A8 | 0.18-hotfix | BW1100 | 2 | IFB | [**`>_`**](#qwen35-122b-a10b-channel-fp8-w8a8-ifb-bw1100-2x-vllm-018-hotfix) |
| [Qwen/Qwen3.5-397B-A17B-W8A8-INT8](https://modelscope.cn/models/hygon/Qwen3.5-397B-A17B-Channel-INT8-w8a8) | INT8 W8A8 | [0.18](../docker_images.md) | BW1100  | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-bw1100-8x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1100 | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-bw1100-8x-vllm-018-hotfix) |
|                                                                                                | INT8 W8A8 | 0.18 | BW1000  | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-bw1000-8x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | BW1000 | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-bw1000-8x-vllm-018-hotfix) |
|                                                                                                | INT8 W8A8 | 0.18 | K100_AI | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-k100_ai-8x-vllm-018) |
|  | INT8 W8A8 | 0.18-hotfix | K100_AI | 8 | IFB | [**`>_`**](#qwen35-397b-a17b-w8a8-int8-ifb-k100_ai-8x-vllm-018-hotfix) |
| [hygon/Qwen3.5-397B-A17B-Channel-FP8](https://www.modelscope.cn/models/hygon/Qwen3.5-397B-A17B-Channel-FP8) | FP8 W8A8 | [0.18](../docker_images.md) | BW1100  | 4 | IFB | [**`>_`**](#qwen35-397b-a17b-channel-fp8-ifb-bw1100-4x-vllm-018) |
|  | FP8 W8A8 | 0.18-hotfix | BW1100 | 4 | IFB | [**`>_`**](#qwen35-397b-a17b-channel-fp8-ifb-bw1100-4x-vllm-018-hotfix) |

## 启动命令

### Qwen3.5-4B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-4B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-4B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-4B IFB BW1000 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### Qwen3.5-4B IFB K100_AI 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-4B IFB K100_AI 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-4B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-9B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-9B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-9B IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-9B IFB BW1000 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-9B IFB K100_AI 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-9B IFB K100_AI 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-9B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-27B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-27B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-27B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-27B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-27B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-27B IFB BW1000 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-27B IFB K100_AI 2x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-27B IFB K100_AI 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-27B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-27B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-27B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-27B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-27B-Channel-INT8-w8a8 IFB K100_AI 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```

### Qwen3.5-27B-Channel-INT8-w8a8 IFB K100_AI 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.5-27B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```
### Qwen3.5-35B-A3B IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-35B-A3B IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.5-35B-A3B IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-35B-A3B IFB BW1000 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.5-35B-A3B IFB K100_AI 2x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-35B-A3B IFB K100_AI 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-35B-A3B \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB BW1000 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB K100_AI 1x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-35B-A3B-Channel-INT8-w8a8 IFB K100_AI 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/Qwen3.5-35B-A3B-Channel-INT8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-35B-A3B-Channel-FP8-w8a8 IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-35B-A3B-Channel-FP8-w8a8 IFB BW1100 1x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-35B-A3B-Channel-FP8-w8a8 \
  -tp 1 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-122B-A10B IFB BW1100 4x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-122B-A10B IFB BW1100 4x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.5-122B-A10B IFB BW1000 8x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-122B-A10B IFB BW1000 8x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1
vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM \
  --moe-backend aiter
```
### Qwen3.5-122B-A10B IFB K100_AI 8x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```

### Qwen3.5-122B-A10B IFB K100_AI 8x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-122B-A10B \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3
```
### Qwen3.5-122B-A10B-W8A8-INT8 IFB BW1100 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-122B-A10B-W8A8-INT8 IFB BW1100 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 2 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-122B-A10B-W8A8-INT8 IFB BW1000 4x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-122B-A10B-W8A8-INT8 IFB BW1000 4x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-122B-A10B-W8A8-INT8 IFB K100_AI 4x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```

### Qwen3.5-122B-A10B-W8A8-INT8 IFB K100_AI 4x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```
### Qwen3.5-122B-A10B-Channel-FP8-w8a8 IFB BW1100 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-122B-A10B-Channel-FP8-w8a8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-122B-A10B-Channel-FP8-w8a8 IFB BW1100 2x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-122B-A10B-Channel-FP8-w8a8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-397B-A17B-W8A8-INT8 IFB BW1100 8x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-397B-A17B-W8A8-INT8 IFB BW1100 8x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-397B-A17B-W8A8-INT8 IFB BW1000 8x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-397B-A17B-W8A8-INT8 IFB BW1000 8x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e5m2 \
  --attention-backend FLASH_ATTN_CUSTOM
```
### Qwen3.5-397B-A17B-W8A8-INT8 IFB K100_AI 8x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```

### Qwen3.5-397B-A17B-W8A8-INT8 IFB K100_AI 8x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
  -tp 8 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 
```
### Qwen3.5-397B-A17B-Channel-FP8 IFB BW1100 4x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve hygon/Qwen3.5-397B-A17B-Channel-FP8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin"
```

### Qwen3.5-397B-A17B-Channel-FP8 IFB BW1100 4x vLLM 0.18-hotfix

```bash
export VLLM_HCU_USE_PD_SPLIT=1

vllm serve hygon/Qwen3.5-397B-A17B-Channel-FP8 \
  -tp 4 \
  --trust-remote-code \
  --max-num-batched-tokens 10240 \
  -q slimquant_marlin \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --speculative-config.quantization "slimquant_marlin" \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_CUSTOM
```
## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-4B",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "Qwen/Qwen3.5-4B",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
