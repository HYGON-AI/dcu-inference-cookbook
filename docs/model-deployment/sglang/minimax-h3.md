# MiniMax-H3 on SGLang Diffusion

## 模型简介

MiniMax-H3 是 MiniMax 推出的全模态音视频生成模型，能够统一理解文本、图像和音频条件，并输出带原生音频的视频。模型采用 DiT 去噪网络和时间因果视频自编码器，支持文生音视频、首帧约束音视频以及参考图像/音频驱动的音视频生成。

本文档给出 BW1101 单卡在线推理实践，覆盖以下三个任务：

| 任务 | 模型分区 | 输入 | 输出 |
| --- | --- | --- | --- |
| T2VA | `FL2VA` | 文本 | 视频 + 音频 |
| FL2VA | `FL2VA` | 文本 + 1～2 张关键帧 | 视频 + 音频 |
| Ref2VA | `Ref2VA` | 文本 + 参考图像/音频/视频 | 视频 + 音频 |


## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | ---: | --- | --- |
| [MiniMax/MiniMax-H3](https://www.modelscope.cn/models/MiniMax/MiniMax-H3) | BF16 | 0.5.15 | BW1101 | 1 | Online | [启动命令](#启动命令) |

验证环境：

| 软件 | 版本 |
| --- | --- |
| DTK | 26.04 |
| Python | 3.10.12 |
| SGLang | 0.5.15 |
| sglang-kernel | 0.4.4 |
| Transformers | 5.12.1 |

进入容器后安装视频处理工具，并将 Transformers 对齐到验证版本：

```bash
apt-get update
apt-get install -y ffmpeg
python3 -m pip install --no-deps "transformers==5.12.1"
```

## 启动命令

### MiniMax-H3 Online BW1101 1x

T2VA 和 FL2VA 共用 `fl2va` 分区；运行 Ref2VA 时需要重启 Server，并把
`MODEL_VARIANT` 改为 `ref2va`。`MODEL_PATH` 始终指向 MiniMax-H3 模型仓库根目录。

```bash
export GPU_ID=0
export PORT=30010

# 替换为 MiniMax-H3 模型仓库根目录，目录内包含 FL2VA 和 Ref2VA。
export MODEL_PATH=/path/to/models/MiniMax-H3

# T2VA、FL2VA 使用 fl2va；Ref2VA 改成 ref2va 后重启 Server。
export MODEL_VARIANT=fl2va
# export MODEL_VARIANT=ref2va

mkdir -p /path/to/workspace/outputs

HIP_VISIBLE_DEVICES="$GPU_ID" sglang serve \
  --model-type diffusion \
  --model-path "$MODEL_PATH" \
  --model-variant "$MODEL_VARIANT" \
  --attention-backend fa \
  --num-gpus 1 \
  --tp-size 1 \
  --sp-degree 1 \
  --ulysses-degree 1 \
  --ring-degree 1 \
  --trust-remote-code \
  --warmup-mode off \
  --host 0.0.0.0 \
  --port "$PORT" \
  --strict-ports \
  --output-path /path/to/workspace/outputs
```

Server 会持续占用当前终端。服务启动后，另开一个终端进入同一容器，并重新设置端口：

```bash
docker exec -it minimax-h3 bash
export PORT=30010
```

然后进行验活：

```bash
curl -sS "http://127.0.0.1:${PORT}/health"
curl -sS "http://127.0.0.1:${PORT}/v1/models" | python3 -m json.tool
```

## API 调用

### Online Inference

### T2VA

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "integrated_multimodal_description: A cat sitting on a windowsill watching snow fall outside, soft indoor lighting, gentle ambient room tone.\noverall_soundscape: Quiet indoor ambience with soft snowfall.\nnon_diegetic_music: N/A",
    "task": "t2va",
    "conditions": [],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "16:9",
      "duration_seconds": 5
    },
    "seed": 1101,
    "n": 1,
    "num_inference_steps": 20,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```


### FL2VA

首帧使用 `frame_index=0`，尾帧使用 `frame_index=-1`。

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "For the target video, at 0.00 seconds into the target video, <Picture 1> is fully referenced. Generate rotating track shots, a 360-degree orbit, bullet time, and water droplets surrounding the camera.",
    "task": "fl2va",
    "conditions": [
      {
        "type": "image",
        "uri": "/path/to/workspace/inputs/fl2va_first_frame.png",
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

```bash
curl -sS -X POST "http://127.0.0.1:${PORT}/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<Subject 1> is the cat in <Picture 1>. <Audio 1> is the vocal and lip-sync reference. Create a realistic video in which <Subject 1> lip-syncs precisely to <Audio 1> while preserving its appearance.",
    "task": "ref2va",
    "conditions": [
      {
        "type": "image",
        "uri": "/path/to/workspace/inputs/ref2va_image.png",
        "role": "reference"
      },
      {
        "type": "audio",
        "uri": "/path/to/workspace/inputs/ref2va_audio.mp3",
        "role": "reference"
      }
    ],
    "target": {
      "short_edge": 768,
      "aspect_ratio": "auto",
      "duration_seconds": 8
    },
    "seed": 3101,
    "n": 1,
    "num_inference_steps": 50,
    "flow_shift": 12,
    "audio_flow_shift": 3
  }'
```

`POST /v1/videos` 返回 `queued` 后，任务会在 Server 后台继续执行。本文直接观察 Server 终端日志；出现 `Output saved to ...mp4` 和 `Pixel data generated successfully` 表示生成完成，视频保存在 `--output-path` 指定的目录。

## 性能参考

以下数据在单张 BW1101上验证。每个场景连续运行 3 次完整 warmup，第 4 次作为正式测量。

| 场景 | 输出规格 | Steps | API 推理耗时 | Text Encode | Visual Encode | Audio Encode | Denoise | 单步 Denoise | Decode | 峰值显存 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T2VA | 1344×768，124 帧，24 FPS | 20 | 499.70s | 20.50s | 0.00s | 0.00s | 349.82s | 18.41s | 123.47s | 123,290 MiB（83.62%） |
| FL2VA | 1344×768，124 帧，24 FPS | 50 | 1107.69s | 41.85s | 1.16s | 0.00s | 944.16s | 19.27s | 114.56s | 123,696 MiB（83.90%） |
| Ref2VA | 1344×768，192 帧，24 FPS | 50 | 2675.97s | 27.41s | 6.34s | 0.53s | 2436.85s | 49.73s | 198.16s | 125,598 MiB（85.19%） |

## 关键参数说明

- `--model-path`：指向包含 `FL2VA`、`Ref2VA` 的 MiniMax-H3 模型仓库根目录。
- `--model-variant`：T2VA/FL2VA 使用 `fl2va`，Ref2VA 使用 `ref2va`；切换分区后需要重启 Server。
- `--attention-backend fa`：显式选择 HCU FlashAttention-2。HCU 环境默认可能选择 `aiter`，但 MiniMax-H3 的 Qwen3VL 文本编码器当前只支持 `fa` 或 `torch_sdpa`。
- `--num-gpus 1 --tp-size 1 --sp-degree 1 --ulysses-degree 1 --ring-degree 1`：本文验证的是 BW1101 单卡配置。
- `--warmup-mode off`：关闭 Server 启动阶段自动 warmup；正式测试仍需由 Client 连续发送 warmup 请求。
- `target.short_edge=768`：本文模型和性能数据使用的短边尺寸。
- `target.aspect_ratio`：T2VA 可指定固定比例；FL2VA 通常使用 `auto` 跟随关键帧语义。Ref2VA 的 `auto` 默认回退为 16:9，并不保证跟随参考图比例。
- `target.duration_seconds`：公开请求通常使用 5～15 秒；时长越长，序列长度、耗时和显存压力越高。
- `num_inference_steps`：控制扩散采样步数。减少步数可降低耗时，但需要重新验证视频与音频质量。
- `conditions`：FL2VA 支持 1～2 个有序关键帧；Ref2VA 至少提供一个 reference，且图像、音频、视频等条件的顺序具有语义。
- `--trust-remote-code`：会执行模型仓库中的自定义代码，只应加载可信权重和代码。

## 日志检查

启动完成后应看到：

```text
Pipeline instantiated
Worker 0: Scheduler loop started.
Starting FastAPI server.
Application startup complete.
```

提交请求后应看到：

```text
Sampling params:
Running pipeline stages: [...]
[MiniMaxH3TextEncodingStage] finished in ... seconds
[MiniMaxH3DenoisingStage] finished in ... seconds
[MiniMaxH3DecodingStage] finished in ... seconds
Output saved to ...mp4
Pixel data generated successfully in ... seconds
```
