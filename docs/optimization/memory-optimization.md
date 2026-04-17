# 显存优化

## 显存组成

### LLM

```
总显存 = 模型权重 + KV Cache + 激活值 + 临时缓冲区
```

| 组件 | 7B (BF16) | 7B (INT4) | 72B (BF16) |
|------|-----------|-----------|------------|
| 模型权重 | ~14GB | ~4GB | ~144GB |
| KV Cache (4K ctx) | ~2GB | ~2GB | ~16GB |
| 激活值 | ~4GB | ~4GB | ~32GB |
| **总计** | **~20GB** | **~10GB** | **~192GB** |

### VLM（额外开销）

```
总显存 = LLM 显存 + 视觉编码器(ViT) + 图像特征 + 投影层
```

| 组件 | 7B VLM | 72B VLM |
|------|--------|---------|
| LLM 部分 | ~14GB | ~144GB |
| 视觉编码器 (ViT-L) | ~2GB | ~2GB |
| 图像特征 (per image) | ~0.5GB | ~0.5GB |
| 投影层 | ~0.5GB | ~0.5GB |
| **总计 (单图)** | **~17GB** | **~147GB** |

### 图像生成模型

```
总显存 = UNet/DiT + VAE + Text Encoder + 潜空间特征
```

| 模型 | 最低显存 | 推荐 |
|------|---------|------|
| SDXL | 12GB | 16GB+ |
| SD3-Medium | 16GB | 24GB+ |
| FLUX.1-dev | 24GB | 32GB+ |
| CogVideoX-5B | 32GB | 48GB+ |

## 优化策略

### 1. 量化

#### GPTQ 量化

```python
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(bits=4, group_size=128, desc_act=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=gptq_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### AWQ 量化

```python
from transformers import AutoModelForCausalLM, AwqConfig

awq_config = AwqConfig(bits=4, group_size=128, zero_point=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct-AWQ",
    quantization_config=awq_config,
    device_map="auto",
)
```

### 2. KV Cache 量化

```bash
# vLLM 支持 KV Cache INT8 量化
python -m vllm.entrypoints.openai.api_server \
    --model <model> \
    --kv-cache-dtype int8
```

### 3. CPU Offload

```python
# 极端显存不足时，将部分层卸载到 CPU
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model,
    checkpoint_path,
    device_map="auto",
    offload_folder="offload",
    dtype=torch.bfloat16,
)
```

### 4. Diffusers 显存优化

```python
# VAE tiling（大分辨率图像生成）
pipe.vae.enable_tiling()

# VAE slicing（逐块处理，降低峰值显存）
pipe.vae.enable_slicing()

# 模型组件 CPU offload
pipe.enable_model_cpu_offload()

# 序列并行（视频生成）
pipe.enable_sequential_cpu_offload()
```

### 5. VLM 特定优化

```bash
# 限制单次请求图片数量
--limit-mm-per-prompt image=5

# 降低图片处理分辨率
# 在 Transformers 中指定 max_pixels
```

## 显存监控

```bash
# 实时监控
watch -n 1 rocm-smi --showmeminfo

# Python 中获取显存信息
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```
