# Qwen3.5 on vLLM

## 模型简介

Qwen3.5 是 Qwen3 系列的增强版本，在推理能力、代码生成、多语言理解等方面进一步提升。

## 模型列表

| 模型 | 参数量 | 上下文 |量化方式|  推荐硬件 |
|------|--------|--------|--------|---------|
| Qwen3.5-4B | 4B | 262K |未量化(BF16) |1x BW1000 64GB |
| Qwen3.5-9B | 9B | 262K |未量化(BF16) |1x BW1000 64GB |
| Qwen3.5-27B | 27B | 262K |未量化(BF16) |2x BW1000 64GB |
| Qwen3.5-35B-A3B | 35B | 262K|未量化(BF16) | 2x BW1000 64GB |
| Qwen3.5-122B-A10B | 122B | 262K|未量化(BF16) | 8x BW1000 64GB |
| Qwen3.5-122B-A10B-W8A8-INT8 | 122B | 262K|W8A8 INT8 | 4x BW1000 64GB |
| Qwen3.5-397B-A17B-W8A8-INT8 | 397B | 262K |W8A8 INT8 | 8x BW1000 64GB|

## 启动命令

----
### Qwen3.5-4B

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3.5-4B \
    -tp 1 \
    --max-num-batched-tokens 10240 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --trust-remote-code \
    --disable-cascade-attn \
```
----

### Qwen3.5-9B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve  Qwen/Qwen3.5-9B \
    -tp 1 \
    --max-num-batched-tokens 10240 \
    --trust-remote-code \
    --disable-cascade-attn \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 
```
----

### Qwen3.5-27B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3.5-27B \
    -tp 2 \
    --trust-remote-code \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 
```
----

### Qwen3.5-27B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3.5-27B \
    -tp 2 \
    --trust-remote-code \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 
```
----
### Qwen3.5-35B-A3B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3.5-35B-A3B \
    -tp 2 \
    --trust-remote-code \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 
```
----
### Qwen3.5-122B-A10B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve  Qwen/Qwen3.5-122B-A10B \
    -tp 8 \
    --trust-remote-code \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 
```
----
### Qwen3.5-122B-A10B-W8A8-INT8
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve  Qwen/Qwen3.5-122B-A10B-W8A8-INT8 \
    -tp 4 \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --trust-remote-code \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --speculative-config.quantization "slimquant_marlin"
```
----
### Qwen3.5-397B-A17B-W8A8-INT8
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve  Qwen/Qwen3.5-397B-A17B-W8A8-INT8 \
    -tp 8 \
    --disable-cascade-attn \
    --max-num-batched-tokens 10240 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --speculative-config.method mtp \
    --speculative-config.num_speculative_tokens 3 \
    --speculative-config.quantization "slimquant_marlin"
```
----
## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-7B",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

