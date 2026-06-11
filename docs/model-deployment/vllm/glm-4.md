# GLM-4-32B-0414 on vLLM

## 模型简介

GLM-4-32B-0414 是智谱 AI 推出的 32B 参数稠密大语言模型，支持长上下文、工具调用、代码生成等能力。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [ZhipuAI/GLM-4-32B-0414](https://www.modelscope.cn/models/ZhipuAI/GLM-4-32B-0414) | BF16 | 0.18 | BW1100 | 1x | IFB | [**`>_`**](#glm-4-32b-0414-ifb-bw1100-1x-vllm-018) |
|                                                                               | BF16 | 0.18 | BW1000 | 2x | IFB | [**`>_`**](#glm-4-32b-0414-ifb-bw1000-2x-vllm-018) |
|                                                                               | BF16 | 0.18 | K100_AI | 2x | IFB | [**`>_`**](#glm-4-32b-0414-ifb-k100_ai-2x-vllm-018) |

## 启动命令

### GLM-4-32B-0414 IFB BW1100 1x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve ZhipuAI/GLM-4-32B-0414 \
  --tensor-parallel-size 1 \
  --trust-remote-code
```

### GLM-4-32B-0414 IFB BW1000 2x vLLM 0.18

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1

vllm serve ZhipuAI/GLM-4-32B-0414 \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

### GLM-4-32B-0414 IFB K100_AI 2x vLLM 0.18

```bash
vllm serve ZhipuAI/GLM-4-32B-0414 \
  --tensor-parallel-size 2 \
  --trust-remote-code
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="ZhipuAI/GLM-4-32B-0414",
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
  "model": "ZhipuAI/GLM-4-32B-0414",
  "messages": [
    {"role": "system", "content": "你是一个专业的编程助手。"},
    {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"}
  ],
  "max_tokens": 128
  }'
```
