# SD3 / FLUX / SDXL on DCU

## 模型简介

| 模型 | 类型 | 分辨率 | 特点 |
|------|------|--------|------|
| Stable Diffusion 3 | 文生图 | 1024x1024 | 多模态理解，三文本编码器 |
| FLUX.1-dev | 文生图 | 1024x1024 | 高质量，流匹配架构 |
| SDXL | 文生图 | 1024x1024 | 生态丰富，社区模型多 |
| ControlNet | 条件生成 | 可变 | 精确控制（边缘、深度、姿态等） |

## Diffusers 部署

### Stable Diffusion 3

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "A serene mountain landscape at sunset, golden light, photorealistic",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
image.save("sd3_output.png")
```

### FLUX.1-dev

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "A futuristic city with flying cars, cyberpunk style",
    num_inference_steps=50,
    guidance_scale=3.5,
).images[0]
image.save("flux_output.png")
```

### SDXL

```python
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
).to("cuda")

image = pipe(
    "A beautiful garden with cherry blossoms",
    num_inference_steps=30,
    guidance_scale=7.5,
).images[0]
image.save("sdxl_output.png")
```

### ControlNet

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16,
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("photo.png")
canny_image = pipe(
    "a beautiful landscape",
    image=image,
    controlnet_conditioning_scale=0.8,
    num_inference_steps=30,
).images[0]
canny_image.save("controlnet_output.png")
```

## ComfyUI 部署

ComfyUI 原生支持 SD3、FLUX、SDXL、ControlNet，通过节点组合实现复杂工作流：

```bash
# 模型文件放置
ComfyUI/models/checkpoints/    # SDXL, FLUX, SD3 主模型
ComfyUI/models/controlnet/     # ControlNet 模型
ComfyUI/models/lora/           # LoRA 权重
```

详见 → [comfyui-dcu.md](comfyui-dcu.md)

## 性能优化

```python
# VAE tiling（大分辨率）
pipe.vae.enable_tiling()

# VAE slicing（节省显存）
pipe.vae.enable_slicing()

# torch.compile
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")

# 降低推理步数
image = pipe(prompt, num_inference_steps=20)  # 默认 50
```

## 显存参考

| 模型 | 分辨率 | 最低显存 | 推荐 |
|------|--------|---------|------|
| SDXL | 1024x1024 | 12GB | 16GB+ |
| SD3-Medium | 1024x1024 | 16GB | 24GB+ |
| FLUX.1-dev | 1024x1024 | 24GB | 32GB+ |
| SDXL + ControlNet | 1024x1024 | 20GB | 32GB+ |

## DCU 适配注意

- Diffusers 底层使用 PyTorch，DCU 上通过 ROCm 兼容运行
- 部分自定义 CUDA kernel 可能需要 HIP 适配
- 建议使用 `torch.float16` 而非 `torch.bfloat16`（部分模型在 DCU 上 fp16 更稳定）
- 如果遇到 `NotImplementedError`，检查是否有 CUDA 专属算子未适配
