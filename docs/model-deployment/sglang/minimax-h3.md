# MiniMax-H3 on SGLang Diffusion

本文汇总 MiniMax-H3 在 BW1000 和 BW1100 上的 8 卡推荐部署策略。默认运行环境已包含 SGLang 0.5.18，本文不展开镜像安装和性能数据，只说明模型分区、8 卡并行方式、CacheDiT 使用边界和验证口径。

本文下方提供一份通用 8 卡启动、请求和验证模板；更细的平台策略见：

| 硬件 | 单卡显存 | 详细文档 |
| --- | ---: | --- |
| BW1000 | 64 GiB | [MiniMax-H3 on BW1000](./minimax-h3-bw1000.md) |
| BW1100 | 144 GiB | [MiniMax-H3 on BW1100](./minimax-h3-bw1100.md) |

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.18 | BW1000 | 8 | Online | [启动命令](#启动-server) |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.18 | BW1100 | 8 | Online | [启动命令](#启动-server) |


## 模型与场景

| 场景 | `model-variant` | 输入 | 输出 |
| --- | --- | --- | --- |
| T2VA | `fl2va` | 文本 | 视频 + 音频 |
| FL2VA | `fl2va` | 文本 + 首尾帧/关键帧 | 视频 + 音频 |
| Ref2VA | `ref2va` | 文本 + 参考图像、音频或视频 | 视频 + 音频 |

T2VA 与 FL2VA 可以共用 `fl2va` Server；切换到 Ref2VA 时需要以 `ref2va` 重新启动 Server。

## 8 卡推荐策略

总表只给一个稳定的 8 卡主配置。CacheDiT 是否开启属于请求/服务开关，不改变这里的推荐拓扑；需要专项调优时再查看对应硬件的详细文档。

| 硬件 | 推荐 8 卡布局 | 覆盖场景 | Offload 建议 | 说明 |
| --- | --- | --- | --- | --- |
| BW1000 | TP2 + SP4 | T2VA、FL2VA、Ref2VA | 默认全关；Ref2VA 复杂参考素材 OOM 时优先开启 Text Encoder offload | 64 GiB 卡的通用 8 卡部署入口 |
| BW1100 | SP8 | T2VA、FL2VA、Ref2VA | 默认全关 | 144 GiB 卡显存余量更大，先按纯序列并行作为统一 8 卡入口 |

## 运行变量

先设置通用路径和端口。`MODEL_PATH` 指向 MiniMax-H3 权重根目录，目录内应包含 `FL2VA` 和 `Ref2VA`。

```bash
export MODEL_PATH=/models/MiniMax-H3
export OUTPUT_PATH=/workspace/outputs

export PORT=30010
export SCHEDULER_PORT=30011
export MASTER_PORT=30012
export WARMUP_MODE=server

export OMP_NUM_THREADS=32
export AllTOAll_STREAM_WITH_COMPUTE=1
export MINIMAX_H3_TORCH_SDPA_BACKEND=flash
export MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT=1
```

再按硬件选择一组 8 卡布局。

BW1000 推荐 8 卡配置：

```bash
export GPU_IDS=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export TP_SIZE=2
export SP_DEGREE=4
export ULYSSES_DEGREE=4
```

BW1100 推荐 8 卡配置：

```bash
export GPU_IDS=0,1,2,3,4,5,6,7
export NUM_GPUS=8
export TP_SIZE=1
export SP_DEGREE=8
export ULYSSES_DEGREE=8
```

默认关闭组件 offload。BW1000 的 Ref2VA 复杂参考素材或更长视频 OOM 时，可以优先把 `TEXT_ENCODER_OFFLOAD` 改成 `true`。

```bash
export TEXT_ENCODER_OFFLOAD=false
export IMAGE_ENCODER_OFFLOAD=false
export DIT_OFFLOAD=false
export DIT_LAYERWISE_OFFLOAD=false
export VAE_OFFLOAD=false
export PIN_CPU_MEMORY=false
export USE_FSDP=false
```

## 启动 Server

T2VA 和 FL2VA 使用 `fl2va` 分区：

```bash
export MODEL_VARIANT=fl2va
```

Ref2VA 需要重启 Server，并改用 `ref2va` 分区：

```bash
export MODEL_VARIANT=ref2va
```

启动命令。MiniMax-H3 建议显式指定组件级 attention backend：`transformer=fa` 给 DiT 主干，`text_encoder=torch_sdpa` 给文本/视觉编码器，避免不同组件都继承同一个默认 backend。

```bash
HIP_VISIBLE_DEVICES="$GPU_IDS" sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --model-variant "$MODEL_VARIANT" \
  --num-gpus "$NUM_GPUS" \
  --tp-size "$TP_SIZE" \
  --sp-degree "$SP_DEGREE" \
  --ulysses-degree "$ULYSSES_DEGREE" \
  --ring-degree 1 \
  --encoder-parallel auto \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --performance-mode manual \
  --dit-cpu-offload "$DIT_OFFLOAD" \
  --dit-layerwise-offload "$DIT_LAYERWISE_OFFLOAD" \
  --text-encoder-cpu-offload "$TEXT_ENCODER_OFFLOAD" \
  --image-encoder-cpu-offload "$IMAGE_ENCODER_OFFLOAD" \
  --vae-cpu-offload "$VAE_OFFLOAD" \
  --pin-cpu-memory "$PIN_CPU_MEMORY" \
  --use-fsdp-inference "$USE_FSDP" \
  --trust-remote-code \
  --warmup-mode "$WARMUP_MODE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --scheduler-port "$SCHEDULER_PORT" \
  --master-port "$MASTER_PORT" \
  --strict-ports \
  --output-path "$OUTPUT_PATH"
```

Server 是前台常驻进程。服务启动完成后，在另一个 Shell 中验活：

```bash
curl -sS "http://127.0.0.1:${PORT}/health"
curl -sS "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool
```

## Warmup

Warmup 不能省略。严格验收建议把 `WARMUP_MODE` 设为 `off`，手动执行 3 次 2-step 请求，然后再发送正式请求。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "t2va",
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside.\noverall_soundscape: Quiet indoor ambience.\nnon_diegetic_music: N/A",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 2,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

## 三种场景请求

### T2VA

T2VA 使用 `MODEL_VARIANT=fl2va`。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -o create.json \
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

### FL2VA

FL2VA 使用 `MODEL_VARIANT=fl2va`。首帧使用 `frame_index=0`，尾帧使用 `frame_index=-1`；图片路径必须能在 Server 容器内访问。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -o create.json \
  -d '{
    "task": "fl2va",
    "prompt": "At 0.00 seconds, <Picture 1> is fully referenced. Generate a smooth cinematic camera orbit while preserving the subject and scene.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/fl2va_first_frame.png",
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

Ref2VA 使用 `MODEL_VARIANT=ref2va`，需要单独启动 `ref2va` Server。参考图片、音频或视频路径必须能在 Server 容器内访问。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -o create.json \
  -d '{
    "task": "ref2va",
    "prompt": "<Subject 1> is the subject in <Picture 1>. <Audio 1> is the voice reference. Create a realistic video with precise lip sync while preserving the subject appearance.",
    "conditions": [
      {
        "type": "image",
        "uri": "/inputs/ref2va_image.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "/inputs/ref2va_audio.mp3",
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

## 查询和下载结果

接口会先返回 `queued` 和任务 ID。可以从 `create.json` 取出 `VIDEO_ID`，再查询状态或下载文件。

```bash
export VIDEO_ID=$(python3 - <<'PY'
import json
print(json.load(open("create.json"))["id"])
PY
)

curl -sS "http://127.0.0.1:${PORT}/v1/videos/${VIDEO_ID}" | python3 -m json.tool
curl -sS -L "http://127.0.0.1:${PORT}/v1/videos/${VIDEO_ID}/content" -o output.mp4
```

如果只看 Server 落盘结果，日志出现 `Output saved to ...mp4` 和 `Pixel data generated successfully` 后，视频也会保存在 `OUTPUT_PATH` 下。

## 可选：开启 CacheDiT

CacheDiT 是 denoising 阶段的近似复用加速，不改变推荐并行拓扑。默认不开启；如果要开，需要在启动 Server 前设置：

```bash
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_FN=1
export SGLANG_CACHE_DIT_BN=0
export SGLANG_CACHE_DIT_WARMUP=4
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3



如果明确要关掉 CacheDiT，启动 Server 前执行：

```bash
export SGLANG_CACHE_DIT_ENABLED=false
```

## 可选：生成并启用 AdaLN cache

AdaLN cache 是给固定 T2VA 配置用的精确预计算 sidecar，不是 CacheDiT。它把固定 step/timestep plan 下的 AdaLN 结果提前算好，启动时用 sidecar 代替一部分 AdaLN 权重常驻显存，主要用于降低显存压力。

先准备模型权重，再生成 sidecar：

```bash
HIP_VISIBLE_DEVICES=0 python3 -m sglang.multimodal_gen.tools.build_minimax_h3_adaln_cache \
  --transformer-path "$MODEL_PATH/FL2VA/transformer" \
  --model-variant fl2va \
  --mode t2va \
  --num-inference-steps 50 \
  --flow-shift 12 \
  --audio-flow-shift 3 \
  --output /path/minimax_h3_t2va_50step_adaln.safetensors

ls -lh /path/minimax_h3_t2va_50step_adaln.safetensors
```

只有在本次请求仍是相同权重、T2VA、50 steps、`flow_shift=12`、`audio_flow_shift=3` 时，才可以在启动 Server 时额外加这一行：

```bash
  --minimax-h3-adaln-cache-path /path/minimax_h3_t2va_50step_adaln.safetensors \
```

例如把它放在 `--output-path "$OUTPUT_PATH"` 前面；注意前一行要保留反斜杠。默认不启用 AdaLN cache 时，不需要设置 `MINIMAX_H3_ADALN_CACHE_PATH`，也不需要加这行参数。

## 日志检查

启动和请求完成后，建议确认日志中包含以下关键行：

```text
Using HCU FlashAttention-2 backend on HCU.
Using fa attention backend
Pipeline instantiated
[MiniMaxH3TextEncodingStage] finished in ... seconds
[MiniMaxH3DenoisingStage] finished in ... seconds
[MiniMaxH3DecodingStage] finished in ... seconds
Output saved to ...mp4
Pixel data generated successfully in ... seconds
```

同时确认启动参数中包含 `--component-attention-backends text_encoder=torch_sdpa,transformer=fa`：文本/视觉编码器保持 torch SDPA 路径，DiT transformer 使用 FA 路径。

开启 CacheDiT 后还要确认命中数大于 0，并确认 `residual_diffs` 没有 `NaN`。
