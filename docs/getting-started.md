# 快速开始

> 5 分钟在 DCU 上跑起第一个 AI 模型推理。

## 前置条件

- 已安装 DCU 驱动和 DTK（参考 [environment-setup.md](environment-setup.md)）
- Python 3.10+ 环境
- 至少一张 DCU（64GB 显存推荐）

## 大语言模型 (LLM)

### 使用 Transformers 快速推理

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "你好，请介绍一下你自己"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 使用 vLLM 启动推理服务

```bash
pip install vllm

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --trust-remote-code

curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }'
```

## 多模态模型 (VLM)

### 视觉语言模型 — 图像理解

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_name)

messages = [
    {"role": "user", "content": [
        {"type": "image", "image": "https://example.com/photo.jpg"},
        {"type": "text", "content": "描述这张图片的内容"},
    ]}
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[...], return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

### 图像生成 — Stable Diffusion

```python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("a cat sitting on a table, high quality").images[0]
image.save("output.png")
```

### 语音识别 — Whisper

```python
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.float16, device_map="auto"
)

import torchaudio
audio, sr = torchaudio.load("audio.wav")
input_features = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features.to(model.device)
predicted_ids = model.generate(input_features)
print(processor.decode(predicted_ids[0]))
```

## 下一步

- [模型部署方案概览](model-deployment/overview.md)
- [多模态方案概览](multimodal/overview.md)
- [性能调优指南](optimization/performance-tuning.md)
- [框架适配详情](frameworks/vllm-dcu.md)
