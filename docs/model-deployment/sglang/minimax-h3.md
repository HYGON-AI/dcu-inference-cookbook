# MiniMax-H3 on SGLang Diffusion

## 模型简介

MiniMax-H3 是 MiniMax 推出的全模态音视频生成模型，采用 DiT 去噪网络和时间因果视频自编码器，可根据文本、关键帧或参考素材生成带原生音频的视频。

本文覆盖以下三个场景：

| 场景 | 模型分区 | 输入 | 输出 |
| ---- | -------- | ---- | ---- |
| T2VA | `fl2va` | 文本 | 视频 + 音频 |
| FL2VA | `fl2va` | 文本 + 1～2 张关键帧 | 视频 + 音频 |
| Ref2VA | `ref2va` | 文本 + 参考图像、音频或视频 | 视频 + 音频 |

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.15 | BW1100 | 8x | Online | [启动命令](#minimax-h3-t2va-online-bw1100-8x-sglang-0515) |

## 启动命令

### MiniMax-H3 T2VA Online BW1100 8x SGLang 0.5.15

以下推荐配置采用 SP8 并行。T2VA 和 FL2VA 共用 `fl2va` 分区；运行 Ref2VA 时，需要把 `MODEL_VARIANT` 改为 `ref2va` 并重启 Server。

```bash
export GPU_IDS=0,1,2,3,4,5,6,7
export MODEL_PATH=/path/to/MiniMax-H3
export OUTPUT_PATH=/path/to/outputs

# T2VA、FL2VA 使用 fl2va；Ref2VA 改为 ref2va。
export MODEL_VARIANT=fl2va
# export MODEL_VARIANT=ref2va

export PORT=30010
export SCHEDULER_PORT=30011
export MASTER_PORT=30012

# 使用 PyTorch 自动选择稳定的 Video VAE SDPA 后端；不要在未验证的 HCU 上强制 flash。
export MINIMAX_H3_TORCH_SDPA_BACKEND=auto
export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1

HIP_VISIBLE_DEVICES="$GPU_IDS" sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --model-variant "$MODEL_VARIANT" \
  --num-gpus 8 \
  --tp-size 1 \
  --sp-degree 8 \
  --ulysses-degree 8 \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --performance-mode manual \
  --dit-cpu-offload false \
  --dit-layerwise-offload false \
  --text-encoder-cpu-offload false \
  --image-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --pin-cpu-memory false \
  --use-fsdp-inference false \
  --trust-remote-code \
  --warmup-mode server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --scheduler-port "$SCHEDULER_PORT" \
  --master-port "$MASTER_PORT" \
  --strict-ports \
  --output-path "$OUTPUT_PATH"
```

## API 调用

### T2VA

Server ready 后，在同一容器的另一个终端执行：

```bash
export PORT=30010

curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside, soft indoor lighting, gentle ambient room tone.\noverall_soundscape: Quiet indoor ambience with soft snowfall.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

接口会先返回 `queued` 和任务 ID。生成任务在 Server 后台执行；日志出现 `Output saved to ...mp4` 和 `Pixel data generated successfully` 后，视频保存在 `--output-path` 指定的目录。

### FL2VA

FL2VA 与 T2VA 共用 `fl2va` Server。首帧使用 `frame_index=0`，尾帧使用 `frame_index=-1`；关键帧路径必须能在 Server 容器中访问。

```bash
export PORT=30010

curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "fl2va",
    "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced. Generate rotating track shots, a 360-degree orbit, bullet time, and water droplets surrounding the camera.",
    "conditions": [
      {
        "type": "image",
        "uri": "/path/to/inputs/fl2va_first_frame.png",
        "role": "keyframe",
        "frame_index": 0
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 2101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

### Ref2VA

Ref2VA 使用 `ref2va` 分区。先停止当前 Server，把 `MODEL_VARIANT` 改为 `ref2va`，再重新执行启动命令。下面示例以参考图片和参考音频驱动生成；两个输入路径都必须能在 Server 容器中访问。

```bash
export PORT=30010

curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "ref2va",
    "prompt": "<Subject 1> is the cat in <Picture 1>. <Audio 1> is the vocal and lip-sync reference. Create a realistic video in which <Subject 1> lip-syncs precisely to <Audio 1> while preserving its appearance.",
    "conditions": [
      {
        "type": "image",
        "uri": "/path/to/inputs/ref2va_image.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "/path/to/inputs/ref2va_audio.mp3",
        "role": "reference"
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 5
    },
    "seed": 3101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

## 性能参考

以下数据使用推荐镜像在 BW1100 8 卡节点上验证。输入为 768p、输出为 1344×768、124 帧、约 5.17 秒、50 inference steps。Server 内部 warmup 完成后执行 3 次 2-step 短 warmup，第 4 次执行 50-step 正式测量。

| 配置 | CacheDiT | Server 总耗时 | E2E | Denoise 耗时 | 单步耗时 | Decode 耗时 | 单卡峰值显存 |
| ---- | -------- | ------------- | --- | ------------ | -------- | ----------- | ------------ |
| SP8 | 不开启 | 133.45s | 136.35s | 113.02s | 2.3066s/step | 15.84s | 91,741 MiB（62.22%） |
| SP8 | 开启（Fn=1、Bn=0、RDT=0.24、MC=3） | 96.20s | 98.23s | 72.48s | 1.4791s/step | 18.93s | 84,458 MiB（57.28%） |

不开 CacheDiT 的结果是无近似缓存基线。开启 CacheDiT 的结果命中 33/49 个 Denoising step，属于近似加速；应根据目标业务的视频质量要求单独验收，不能视为与基线完全等价的无损优化。

不同节点负载、温度、频率和首次加载状态会带来小幅波动，建议以短 warmup 后的正式请求作为稳定性能参考。

## 关键参数说明

- `--model-variant`：T2VA、FL2VA 使用 `fl2va`，Ref2VA 使用 `ref2va`；切换分区后需要重启 Server。
- `--tp-size 1 --sp-degree 8 --ulysses-degree 8`：BW1100 8 卡推荐的 SP8 配置。
- `--attention-backend fa`：DiT 主干使用 HCU FlashAttention 后端。
- `MINIMAX_H3_TORCH_SDPA_BACKEND=auto`：由 PyTorch 自动选择 Video VAE SDPA 后端。不要在未完成目标卡型稳定性验证前强制设为 `flash`；HCU 上强制 Flash 可能在 Video VAE Decode 阶段触发 VMFault。
- `MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1`：Video VAE Decoder 使用流式 temporal concat，降低中间张量开销。
- `LD_LIBRARY_PATH`：选择镜像内固化的 hipBLASLt 库。BW1100 的 Video VAE Decode 用时不能直接按 DiT 算力比例估算。
- `--use-fsdp-inference false` 和所有 offload 参数为 `false`：模型组件常驻设备，避免 CPU/GPU 搬运开销。
- `--warmup-mode server`：在 Server ready 前执行内部短 warmup；正式性能测试仍需额外发送短 warmup 请求。
- `num_inference_steps=50`：SGLang 日志中实际执行 49 次 Denoising 迭代，表中单步耗时按 49 次计算。

## 显存不足时的兜底配置

如果发生 OOM，建议优先开启 FSDP，或把 Text Encoder offload 到 CPU。将启动命令中的对应参数改为：

```bash
# 方案一：开启 FSDP
--use-fsdp-inference true

# 方案二：将 Text Encoder offload 到 CPU
--text-encoder-cpu-offload true
```

这两个选项可以单独启用，也可以在显存仍不足时组合启用。FSDP 会切分模型权重，Text Encoder CPU offload 会降低编码阶段的设备显存占用，但两者都可能增加通信或 CPU/GPU 数据搬运开销，因此性能通常低于默认推荐配置。Ref2VA 的参考素材编码和更长输出更容易触发显存压力，可优先使用该兜底方案。

## 日志检查

启动和请求完成后，建议确认日志中包含以下关键行：

```text
Using HCU FlashAttention-2 backend on HCU.
Using fa attention backend
Pipeline instantiated
[MiniMaxH3DenoisingStage] finished in ... seconds
[MiniMaxH3DecodingStage] finished in ... seconds
Output saved to ...mp4
Pixel data generated successfully in ... seconds
```
