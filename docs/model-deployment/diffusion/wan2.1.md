# Wan2.1 on DCU

## 模型简介

Wan2.1 是阿里推出的开源视频生成模型，支持文生视频和图生视频，提供 1.3B 和 14B 两个版本。

## 模型列表

| 模型 | 参数量 | 分辨率 | 帧数 | 帧率 | 推荐硬件 |
|------|--------|--------|------|------|---------|
| Wan2.1-T2V-1.3B | 1.3B | 832x480 | 81 | 16fps | 1x DCU 64GB |
| Wan2.1-T2V-14B | 14B | 832x480 | 81 | 16fps | 1x DCU 64GB |
| Wan2.1-I2V-14B-480P | 14B | 832x480 | 81 | 16fps | 1x DCU 64GB |

## Diffusers 部署

### 文生视频

```python
import torch
from diffusers import WanPipeline
from diffusers.utils import export_to_video

pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B",
    torch_dtype=torch.float16,
).to("cuda")

video = pipe(
    prompt="A cat wearing sunglasses, walking down a city street, cinematic lighting",
    num_frames=81,
    height=480,
    width=832,
    num_inference_steps=40,
    guidance_scale=5.0,
).frames[0]

export_to_video(video, "wan_output.mp4", fps=16)
```

### 图生视频

```python
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = WanImageToVideoPipeline.from_pretrained(
    "Wan-AI/Wan2.1-I2V-14B-480P",
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("input.png").resize((832, 480))
video = pipe(
    image=image,
    prompt="The flowers gently sway in the wind",
    num_frames=81,
    height=480,
    width=832,
    num_inference_steps=40,
    guidance_scale=5.0,
).frames[0]

export_to_video(video, "wan_i2v_output.mp4", fps=16)
```

### 轻量版 1.3B

```python
pipe = WanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B",
    torch_dtype=torch.float16,
).to("cuda")

video = pipe(
    prompt="A dog running in the park",
    num_frames=81,
    height=480,
    width=832,
    num_inference_steps=30,
).frames[0]
```

## ComfyUI 部署

ComfyUI 通过自定义节点支持 Wan2.1：

```bash
# 安装 Wan 视频节点
cd ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
```

在 ComfyUI 中加载 Wan 模型，通过节点组合实现文生视频/图生视频工作流。

详见 → [comfyui-dcu.md](comfyui-dcu.md)

## 性能优化

```python
# VAE tiling / slicing
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# CPU offload
pipe.enable_model_cpu_offload()

# torch.compile
pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
```

## DCU 适配注意

- Wan2.1 基于 DiT 架构，算子适配相对简单
- 14B 版本建议 64GB+ DCU
- 1.3B 版本适合显存有限的场景
- 视频编码/解码使用 CPU（ffmpeg），不影响 DCU 兼容性
- 如果遇到 OOM，优先启用 `enable_model_cpu_offload()`
