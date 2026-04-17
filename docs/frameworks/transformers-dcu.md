# Transformers on DCU

## 安装

```bash
pip install transformers accelerate
```

## LLM 推理

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

messages = [{"role": "user", "content": "你好"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## VLM 推理（视觉语言模型）

### Qwen2.5-VL

```python
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": "https://example.com/photo.jpg"},
        {"type": "text", "content": "描述这张图片"},
    ],
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[...], return_tensors="pt", padding=True).to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=1024)
```

## 图像生成（Diffusers）

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe("a beautiful landscape", num_inference_steps=30).images[0]
image.save("output.png")
```

## 语音模型

### Whisper (ASR)

```python
import torch, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-large-v3", torch_dtype=torch.float16, device_map="auto"
)

audio, sr = torchaudio.load("audio.wav")
input_features = processor(
    audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
).input_features.to(model.device, dtype=torch.float16)
predicted_ids = model.generate(input_features)
print(processor.decode(predicted_ids[0], skip_special_tokens=True))
```

## device_map 策略

```python
# 自动分配（推荐）
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 手动指定
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"model.embed_tokens": 0, "model.layers": "auto", "lm_head": 0}
)

# 均匀分布到多卡
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced")
```

## 加速推理

### BetterTransformer

```python
model = model.to_bettertransformer()
```

### torch.compile

```python
model = torch.compile(model, mode="reduce-overhead")
```

### Flash Attention 2

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

## 适用场景

- 模型开发与调试
- 小规模推理
- VLM / 语音 / 图像生成等多模态场景
- 不适合高并发在线服务（推荐使用 vLLM / SGLang）
