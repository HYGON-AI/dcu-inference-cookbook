# CogVideoX on DCU

## 模型简介

CogVideoX 是智谱 AI 开源的视频生成模型，支持文生视频和图生视频。

## 模型列表

| 模型 | 参数量 | 分辨率 | 帧数 | 帧率 | 推荐硬件 |
|------|--------|--------|------|------|---------|
| CogVideoX-2B | 2B | 720x480 | 49 | 8fps | 1x BW1000 64GB |
| CogVideoX-5B | 5B | 720x480 | 49 | 8fps | 1x BW1000 64GB |

## Diffusers 部署

### 文生视频

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.float16,
).to("cuda")

video = pipe(
    prompt="A panda eating bamboo on a wooden table, high quality",
    num_videos_per_prompt=1,
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]

export_to_video(video, "cogvideox_output.mp4", fps=8)
```

### 图生视频

```python
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import load_image, export_to_video

pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.float16,
).to("cuda")

image = load_image("input.png").resize((720, 480))
video = pipe(
    image=image,
    prompt="The camera slowly zooms in on the subject",
    num_inference_steps=50,
    guidance_scale=6.0,
).frames[0]

export_to_video(video, "cogvideox_i2v_output.mp4", fps=8)
```

## 性能优化

```python
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
pipe.enable_model_cpu_offload()
```

## DCU 适配注意

- 3D Attention 算子可能需要 HIP 适配
- 建议先用 2B 版本验证 DCU 兼容性
- 如果遇到 OOM，优先启用 `enable_model_cpu_offload()`
