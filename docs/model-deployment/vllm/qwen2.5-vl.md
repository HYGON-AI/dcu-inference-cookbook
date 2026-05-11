# Qwen2.5-VL on vLLM

## 模型简介

Qwen2.5-VL 是 Qwen 系列的多模态视觉语言模型，包含3B 7B、32B、72B 等多个参数规模。模型原生支持图像、文本、视频等多种输入，在视觉理解、文档解析、视频问答、目标定位等任务上表现优异，结合 vLLM 可实现高吞吐、低延迟的在线推理服务。


## 模型列表

| 模型 | 参数量 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|--------|--------|---------|
| Qwen2.5-VL-3B-Instruct | 3B | 128K | 未量化(BF16) | 1x BW1000 64GB |
| Qwen2.5-VL-7B-Instruct | 7B | 128K | 未量化(BF16)  | 1x BW1000 64GB |
| Qwen2.5-VL-32B-Instruct | 32B | 128K | 未量化(BF16) | 4x BW1000 64GB |
| Qwen2.5-VL-72B-Instruct | 72B | 128K | 未量化(BF16) | 8x BW1000 64GB |


## 启动命令

### Qwen2.5-VL-3B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve --model Qwen/Qwen2.5-VL-3B-Instruct \
    -tp 1 \
    --trust-remote-code \
```
### Qwen2.5-VL-7B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve --model Qwen/Qwen2.5-VL-7B-Instruct \
    -tp 1 \
    --trust-remote-code \
```
### Qwen2.5-VL-32B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve --model Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 4 \
    --trust-remote-code \
```
### Qwen2.5-VL-72B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve --model Qwen/Qwen2.5-VL-72B-Instruct \
    -tp 8 \
    --trust-remote-code \
```
### API调用
#### 单张图片
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片的内容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/sample.jpg"},
                },
            ],
        }
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)

```

#### 多张图片
```python
response = client.chat.completions.create(
    model="Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这两张图片有什么不同？"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image1.jpg"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image2.jpg"},
                },
            ],
        }
    ],
    max_tokens=2048,
    temperature=0.7,
)
```
#### 本地图片（Base64）
```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64_img = encode_image("local_photo.png")

response = client.chat.completions.create(
    model="Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么物体？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}"
                    },
                },
            ],
        }
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)

```

# 视频理解示例

```python
response = client.chat.completions.create(
    model="Qwen2.5-VL-32B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "总结这个视频的主要内容"
                },
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "https://example.com/demo.mp4"
                    }
                }
            ]
        }
    ],
    max_tokens=4096,
    temperature=0.7
)
```