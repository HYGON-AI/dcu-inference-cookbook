# Qwen2 on vLLM

## 模型简介

Qwen2 是阿里通义千问开源大语言模型系列，支持多种参数规模。本页提供 Qwen2-72B 在 DCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [qwen/Qwen2-72B](https://www.modelscope.cn/models/qwen/Qwen2-72B) | BF16 | 0.18 | BW1000 | 4 | IFB | [**`>_`**](#qwen2-72b-ifb-bw1000-4x-vllm-018) |

## 启动命令

### Qwen2-72B IFB BW1000 4x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3 # 指定使用的 DCU 卡号
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1
export VLLM_HCU_USE_KVCACHE_E5M2=1

vllm serve qwen/Qwen2-72B \
    -tp 4 \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype fp8_e5m2 \
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen/Qwen2-72B",
    messages=[{"role": "user", "content": "解释量子计算的基本原理"}],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/Qwen2-72B",
    "messages": [
      {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    "max_tokens": 128
  }'
```

### PD 分离

暂无。
