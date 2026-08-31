# Wan2.2-T2V on vLLM-Omni

## 模型简介

Wan2.2-T2V-A14B 是阿里通义实验室推出的文生视频（Text-to-Video）模型，基于 DiT 架构，支持根据文本提示词生成高质量视频。
Wan2.2-T2V-A14B-Diffusers 为其 Diffusers 格式的权重，可直接在 vLLM-Omni 上进行离线推理与在线服务部署。

## 模型列表

| 模型权重 | 量化方式 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://www.modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | BF16 | 0.18 | BW1000 | 8 | IFB | [**`>_`**](#wan22-t2v-a14b-diffusers-ifb-bw1000-8x-vllm-018) |
|                                                                                                       | BF16 | 0.18 | BW1100 | 8 | IFB | [**`>_`**](#wan22-t2v-a14b-diffusers-ifb-bw1100-8x-vllm-018) |

## 启动命令

### Wan2.2-T2V-A14B-Diffusers IFB BW1000 8x vLLM 0.18

#### 离线推理(Offline Inference)

```bash
cd vllm-omni/examples/offline_inference/text_to_video

python text_to_video.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --guidance-scale 4.0 \
    --flow-shift 5.0 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --tensor-parallel-size 2 \
    --ulysses-degree 2 \
    --cfg-parallel-size 2 \
    --vae-patch-parallel-size 8 \
    --output t2v_output.mp4
```

#### 在线推理(Online Inference)

##### 启动 vllm server

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --omni \
    --port 8098 \
    --tensor-parallel-size 2 \
    --ulysses-degree 2 \
    --cfg-parallel-size 2 \
    --vae-patch-parallel-size 8
```

##### API Calls

```bash
curl -s http://localhost:8098/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=81" \
  -F "negative_prompt=色调艳丽 ，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42"
```
##### Retrieve a job

```bash
curl -s http://localhost:8098/v1/videos/${video_id} | jq .
```

##### Download the completed video

```bash
curl -L http://localhost:8098/v1/videos/${video_id}/content -o wan22_output.mp4
```

### Wan2.2-T2V-A14B-Diffusers IFB BW1100 8x vLLM 0.18

#### 离线推理(Offline Inference)

```bash
cd vllm-omni/examples/offline_inference/text_to_video

python text_to_video.py --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
    --negative-prompt "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
    --height 720 \
    --width 1280 \
    --num-frames 81 \
    --fps 16 \
    --guidance-scale 4.0 \
    --flow-shift 5.0 \
    --num-inference-steps 40 \
    --boundary-ratio 0.875 \
    --tensor-parallel-size 1 \
    --ulysses-degree 4 \
    --cfg-parallel-size 2 \
    --vae-patch-parallel-size 8 \
    --output t2v_output.mp4
```

#### 在线推理(Online Inference)

##### 启动 vllm server

```bash
vllm serve Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --omni \
    --port 8098 \
    --tensor-parallel-size 1 \
    --ulysses-degree 4 \
    --cfg-parallel-size 2 \
    --vae-patch-parallel-size 8
```

##### API Calls

```bash
curl -s http://localhost:8098/v1/videos \
  -H "Accept: application/json" \
  -F "prompt=Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  -F "width=1280" \
  -F "height=720" \
  -F "num_frames=81" \
  -F "negative_prompt=色调艳丽 ，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走" \
  -F "fps=16" \
  -F "num_inference_steps=40" \
  -F "guidance_scale=4.0" \
  -F "boundary_ratio=0.875" \
  -F "flow_shift=5.0" \
  -F "seed=42"
```
##### Retrieve a job

```bash
curl -s http://localhost:8098/v1/videos/${video_id} | jq .
```

##### Download the completed video

```bash
curl -L http://localhost:8098/v1/videos/${video_id}/content -o wan22_output.mp4
```