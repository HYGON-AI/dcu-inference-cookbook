# ComfyUI on DCU

ComfyUI 是一个基于节点的可视化 AI 图像/视频生成工作流平台，支持 Stable Diffusion、FLUX、Wan 等主流模型。

## 安装

```bash
# 克隆 ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# 安装依赖
pip install -r requirements.txt
```

## 启动

```bash
# 基础启动
python main.py

# 指定监听地址（远程访问）
python main.py --listen 0.0.0.0 --port 8188

# 指定 DCU 设备
python main.py --listen 0.0.0.0 --port 8188
```

启动后访问 `http://<server-ip>:8188` 打开 Web 界面。

## DCU 适配

ComfyUI 底层使用 PyTorch，DCU 上通过 ROCm 兼容运行：

```bash
# 1. 确认 PyTorch ROCm 版本
python -c "import torch; print(torch.version.hip)"

# 2. 如果 ComfyUI 未自动识别 DCU，手动设置
export HIP_VISIBLE_DEVICES=0  # 指定使用的 DCU 卡号

# 3. 启动
python main.py --listen 0.0.0.0 --port 8188
```

## 模型文件

将模型文件放到对应目录：

```
ComfyUI/
├── models/
│   ├── checkpoints/          # 主模型（SDXL, FLUX, Wan 等）
│   ├── lora/                 # LoRA 权重
│   ├── controlnet/           # ControlNet 模型
│   ├── vae/                  # VAE 模型
│   ├── clip/                 # CLIP/Text Encoder
│   ├── unet/                 # UNet/DiT 模型
│   └── diffusion_models/     # 扩散模型
├── input/                    # 输入图片
└── output/                   # 输出结果
```

### 下载模型示例

```bash
# FLUX.1-dev
cd models/checkpoints/
wget https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors

# SDXL
wget https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# Wan2.1-14B（视频生成）
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/*.safetensors
```

## 工作流示例

### 文生图（FLUX）

在 ComfyUI Web 界面中拖拽构建以下节点链：

```
[Load Checkpoint] → [CLIP Text Encode (Positive)]
                 → [CLIP Text Encode (Negative)]
                 → [Empty Latent Image]
                 → [KSampler]
                 → [VAE Decode]
                 → [Save Image]
```

或直接导入 JSON 工作流：

```json
{
  "3": {
    "class_type": "KSampler",
    "inputs": {
      "seed": 42,
      "steps": 30,
      "cfg": 3.5,
      "sampler_name": "euler",
      "scheduler": "normal",
      "denoise": 1.0,
      "model": ["4", 0],
      "positive": ["6", 0],
      "negative": ["7", 0],
      "latent_image": ["5", 0]
    }
  },
  "4": {
    "class_type": "CheckpointLoaderSimple",
    "inputs": {
      "ckpt_name": "flux1-dev.safetensors"
    }
  },
  "5": {
    "class_type": "EmptyLatentImage",
    "inputs": {
      "width": 1024,
      "height": 1024,
      "batch_size": 1
    }
  },
  "6": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "a beautiful sunset over mountains, photorealistic",
      "clip": ["4", 1]
    }
  },
  "7": {
    "class_type": "CLIPTextEncode",
    "inputs": {
      "text": "",
      "clip": ["4", 1]
    }
  },
  "8": {
    "class_type": "VAEDecode",
    "inputs": {
      "samples": ["3", 0],
      "vae": ["4", 2]
    }
  },
  "9": {
    "class_type": "SaveImage",
    "inputs": {
      "filename_prefix": "flux_output",
      "images": ["8", 0]
    }
  }
}
```

### ControlNet 条件生成

```
[Load Checkpoint] → [CLIP Text Encode]
                 → [Load ControlNet Model]
                 → [Apply ControlNet]
                 → [KSampler]
                 → [VAE Decode]
                 → [Save Image]

[Load Image] → [Preprocessor (Canny/Depth)] → [Apply ControlNet]
```

### Wan 视频生成

ComfyUI 支持通过自定义节点加载 Wan 模型：

```bash
# 安装 ComfyUI-VideoHelperSuite（视频相关节点）
cd ComfyUI/custom_nodes/
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
pip install -r ComfyUI-VideoHelperSuite/requirements.txt
```

## 常用自定义节点

```bash
cd ComfyUI/custom_nodes/

# ComfyUI Manager — 节点管理器（必装）
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# ControlNet 预处理器
git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git

# IP-Adapter
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# 视频生成辅助
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Wan 视频模型支持
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
```

## API 调用

ComfyUI 提供 REST API，可用于自动化部署：

```python
import json
import urllib.request

# 加载工作流 JSON
workflow = json.load(open("workflow.json"))

# 设置 prompt
workflow["prompt"]["6"]["inputs"]["text"] = "a cat sitting on a table"

# 提交任务
data = json.dumps({"prompt": workflow}).encode()
req = urllib.request.Request(
    "http://localhost:8188/prompt",
    data=data,
    headers={"Content-Type": "application/json"},
)
response = json.loads(urllib.request.urlopen(req).read())
prompt_id = response["prompt_id"]

# 查询结果
import time
time.sleep(10)  # 等待生成
history = json.loads(
    urllib.request.urlopen("http://localhost:8188/history/" + prompt_id).read()
)
output_images = history[prompt_id]["outputs"]["9"]["images"]
print(f"Generated: {output_images}")
```

## 性能优化

```bash
# 启用 FP16（默认已启用）
# --fp16-vae

# 指定 VAE
# --vae-path models/vae/sdxl_vae.safetensors

# 预加载模型到显存
# --preload-models

# 自定义临时目录
# --temp-directory /tmp/comfyui
```

## DCU 适配注意

- ComfyUI 依赖 PyTorch，确保安装 ROCm 版本的 PyTorch
- 部分自定义节点中的 CUDA kernel 可能需要 HIP 适配
- 如果遇到 `CUDA out of memory`，在节点中降低分辨率或 batch size
- 视频生成节点（Wan）显存需求大，建议 64GB+ DCU
- 使用 ComfyUI Manager 管理节点，方便排查兼容性问题
- 建议先用简单文生图工作流验证 DCU 兼容性，再逐步添加复杂节点

## 参考链接

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI 官方文档](https://docs.comfyui.com/)
- [ComfyUI 工作流分享](https://comfyworkflows.com/)
