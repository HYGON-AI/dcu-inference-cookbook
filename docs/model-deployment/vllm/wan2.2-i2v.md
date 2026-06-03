# Wan2.2-I2V on vLLM-Omni

## 模型简介

Wan2.2-I2V-A14B 是阿里通义实验室推出的图生视频（Image-to-Video）模型，基于 DiT 架构，支持从单张图片生成高质量视频。  
Wan2.2-I2V-A14B-4steps-Diffusers-bf16 是基于Wan2.2-I2V-A14B-Diffusers 架构，融合了 lightx2v/Wan2.2-Distill-Loras 的4步生成（4-Step）LoRA权重后，重新打包为完整 BF16 精度的 Diffusers 模型文件。

## 模型列表

| 模型权重 | 量化方式 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Wan2.2-I2V-A14B-4steps-Diffusers-bf16](https://www.modelscope.cn/models/hygon/Wan2.2-I2V-A14B-4steps-Diffusers-bf16) | BF16 | 0.18 | BW1000 | 8 | IFB | [**`>_`**](#wan22-i2v-a14b-4steps-diffusers-bf16-ifb-bw1000-8x-vllm-018)

## 启动命令
### Wan2.2-I2V-A14B-4steps-Diffusers-bf16 IFB BW1000 8x vLLM 0.18
### 离线推理(Offline Inference)
```bash
export ENABLE_SPARSE_ATTN=1

cd vllm-omni/examples/offline_inference/image_to_video

python image_to_video.py   --model hygon/Wan2.2-I2V-A14B-4steps-Diffusers-bf16  \
    --image input.png \
    --prompt "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  \
    --height 720 \
    --width 1280 \
    --num-frames 121 \
    --fps 24 \
    --guidance-scale 1.0 \
    --num-inference-steps 4 \
    --boundary-ratio 0.875 \
    --tensor-parallel-size 2 \
    --ulysses-degree 4  \
    --output i2v_output.mp4
```



### 在线推理(Online Inference)

#### 启动vllm server
```bash
export ENABLE_SPARSE_ATTN=1

vllm serve hygon/Wan2.2-I2V-A14B-4steps-Diffusers-bf16 \
    --omni \
    --port 8000 \
    --tensor-parallel-size 2 \
    --usp 4
```
#### 启动comfyui
```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```
启动后访问 `http://<server-ip>:8188` 打开 Web 界面

在 ComfyUI Web 界面中拖拽构建以下节点链：

```
[Load Image] → [vLLM-Omni Diffusion Sampling Params]
             → [vLLM-Omni Generate Video]
             → [Save Video]
```

或直接导入 JSON 工作流：
```json
{
  "97": {
    "inputs": {
      "image": "input.png"
    },
    "class_type": "LoadImage",
    "_meta": {
      "title": "加载图像"
    }
  },
  "108": {
    "inputs": {
      "filename_prefix": "video/Wan2.2_i2v",
      "format": "auto",
      "codec": "auto",
      "video": [
        "132",
        0
      ]
    },
    "class_type": "SaveVideo",
    "_meta": {
      "title": "保存视频"
    }
  },
  "132": {
    "inputs": {
      "url": "http://vllm-server-ip:8000/v1",
      "model": "Wan2.2-I2V-A14B-4steps-Diffusers-bf16",
      "prompt": "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character.",
      "negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
      "width": 1280,
      "height": 720,
      "fps": 24,
      "num_frames": 121,
      "image": [
        "97",
        0
      ],
      "sampling_params": [
        "133",
        0
      ]
    },
    "class_type": "VLLMOmniGenerateVideo",
    "_meta": {
      "title": "Generate Video"
    }
  },
  "133": {
    "inputs": {
      "n": 1,
      "num_inference_steps": 4,
      "guidance_scale": 1,
      "true_cfg_scale": 5,
      "vae_use_slicing": false,
      "vae_use_tiling": false,
      "seed": 284
    },
    "class_type": "VLLMOmniDiffusionSampling",
    "_meta": {
      "title": "Diffusion Sampling Params"
    }
  }
}
```