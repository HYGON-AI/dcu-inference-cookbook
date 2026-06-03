# Qwen2.5-VL on vLLM

## 模型简介

Qwen2.5-VL 是阿里通义千问视觉语言模型系列，支持图像、视频与文本等多模态输入。本页提供 Qwen2.5-VL-32B-Instruct 在 DCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2.5-VL-32B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-32B-Instruct) | BF16 | 0.18 | BW1000 | 2 | IFB | [**`>_`**](#qwen25-vl-32b-instruct-ifb-bw1000-2x-vllm-018) |

## 启动命令

### Qwen2.5-VL-32B-Instruct IFB BW1000 2x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=1,2
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 2 \
    --trust-remote-code \
    --enable-chunked-prefill \
    --max-model-len 32768 \
    --allowed-local-media-path /path-to/VL_data/ \
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
