# Qwen2-VL on vLLM

## 模型简介

Qwen2-VL 是阿里通义千问视觉语言模型系列，支持图像、视频与文本等多模态输入。本页提供 Qwen2-VL-2B 在 DCU 上基于 vLLM 的推理部署方案。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Qwen/Qwen2-VL-2B](https://www.modelscope.cn/models/Qwen/Qwen2-VL-2B) | BF16 | 0.18 | BW1000 | 1 | IFB | [**`>_`**](#qwen2-vl-2b-ifb-bw1000-1x-vllm-018) |

## 启动命令

### Qwen2-VL-2B IFB BW1000 1x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0 # 指定使用的 DCU 卡号
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
export VLLM_HCU_USE_CUSTOM_OPS=1

vllm serve Qwen/Qwen2-VL-2B \
    -tp 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.95 \
    --chat-template-content-format openai \
    --chat-template qwen2_vl_openai_chat_template.jinja \
    --allowed-local-media-path /path-to/VL_data/
```

## Chat Template

将以下内容保存为 `qwen2_vl_openai_chat_template.jinja`，并通过 `--chat-template` 参数指定该模板。

```jinja
{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if loop.first and message['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}<|im_start|>{{ message['role'] }}
{% if message['content'] is string %}{{ message['content'] }}<|im_end|>
{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-2B",
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
    "model": "Qwen/Qwen2-VL-2B",
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
