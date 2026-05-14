# Qwen3-VL on vLLM

## 模型简介

Qwen3-VL 是阿里云推出的新一代多模态视觉语言模型（Vision-Language Model, VLM），支持文本、图像、视频等多模态输入，并具备更强的视觉理解、推理、Agent 操作与长上下文能力。
相比 Qwen2.5-VL，Qwen3-VL 在以下方面进一步增强：

- 更强的视觉推理能力（Chart / OCR / GUI / DocQA）
- 更高效的视频理解与长视频处理
- 更优秀的多图关联分析能力
- 支持更长上下文与更高分辨率输入
- 更适合 Agent 场景（屏幕理解、工具调用、GUI 操作）
- 更优的推理吞吐与推理稳定性

结合 vLLM 可实现高吞吐、低延迟的多模态在线推理服务。

---

## 模型列表

| 模型 | 参数量 | 上下文长度 | 量化方式 | 推荐硬件 |
|------|--------|------------|---------|---------|
| Qwen3-VL-2B-Instruct | 2B | 256K |未量化(BF16)| 1x BW1000 64GB |
| Qwen3-VL-2B-Thinking | 2B | 256K |未量化(BF16)| 1x BW1000 64GB |
| Qwen3-VL-4B-Instruct | 4B | 256K |未量化(BF16)|1x BW1000 64GB |
| Qwen3-VL-4B-Thinking | 4B | 256K |未量化(BF16)|1x BW1000 64GB |
| Qwen3-VL-8B-Instruct | 8B | 256K |未量化(BF16)|1x BW1000 64GB |
| Qwen3-VL-8B-Thinking | 8B | 256K |未量化(BF16)|1x BW1000 64GB |
| Qwen3-VL-30B-A3B-Instruct | 30B | 256K |未量化(BF16)| 4x BW1000 64GB |
| Qwen3-VL-30B-A3B-Thinking | 30B | 256K |未量化(BF16)| 4x BW1000 64GB |
| Qwen3-VL-235B-A22B-Instruct | 235B | 256K |未量化(BF16)| 16x BW1000 64GB |
| Qwen3-VL-235B-A22B-Thinking | 235B | 256K |未量化(BF16)| 16x BW1000 64GB |

---

## 启动命令

### Qwen3-VL-2B-Instruct

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-2B-Instruct \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```

### Qwen3-VL-2B-Thinking

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-2B-Thinking \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```

### Qwen3-VL-4B-Instruct

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-4B-Instruct \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```

### Qwen3-VL-4B-Thinking

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-4B-Thinking \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```

### Qwen3-VL-8B-Instruct

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```

### Qwen3-VL-8B-Thinking

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-8B-Thinking \
    --tensor-parallel-size 1 \
    --trust-remote-code \
```



### Qwen3-VL-30B-A3B-Instruct

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --tensor-parallel-size 4 \
    --trust-remote-code \
```

### Qwen3-VL-30B-A3B-Thinking

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-30B-A3B-Thinking \
    --tensor-parallel-size 4 \
    --trust-remote-code \
```



### Qwen3-VL-235B-A22B-Instruct

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct \
    --tensor-parallel-size 16 \
    --trust-remote-code \
```

### Qwen3-VL-235B-A22B-Thinking

```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen3-VL-235B-A22B-Thinking \
    --tensor-parallel-size 16 \
    --trust-remote-code \
```


## OpenAI API 调用示例
### 单张图片推理

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张图片的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/demo.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=2048,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

### 多张图片推理

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "分析这两张图片的区别"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image1.jpg"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image2.jpg"
                    }
                }
            ]
        }
    ],
    max_tokens=2048,
    temperature=0.7
)
```

---

### 本地图片（Base64）

```python
import base64

def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64_img = encode_image("demo.png")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "识别图片中的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_img}"
                    }
                }
            ]
        }
    ],
    max_tokens=2048,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

### 视频理解示例

```python
response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-30B-A3B-Instruct",
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

---

