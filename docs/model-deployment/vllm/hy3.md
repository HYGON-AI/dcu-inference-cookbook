# Hy3 on vLLM

## 模型简介

Hy3 是由腾讯混元团队开发的一款拥有 2950 亿参数的混合专家（MoE）模型，其中激活参数为 210 亿，MTP 层参数为 38 亿。基于 Channel FP8 量化优化，在 DCU 硬件上提供高效推理性能。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#hy3-channel-fp8-w8a8-ifb-bw1100-8x-vllm-021) |

## 启动命令

### Hy3-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.21

以下示例为单节点部署。

```bash
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_FUSE_MOE_GATE=1
export LMSLIM_USE_LIGHTOP=1
export GPU_MAX_HW_QUEUES=4
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --kv_cache_dtype fp8_e4m3 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --served-model-name hy3-fp8
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hy3-fp8",
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
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hy3-fp8",
  "messages": [
    {"role": "user", "content": "你好，请用Python写一个贪吃蛇的游戏脚本"}
  ],
  "max_tokens": 1500,
  "temperature": 0.0
  }'
```
