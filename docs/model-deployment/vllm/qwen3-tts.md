# Qwen3-TTS on vLLM-Omni

## 模型简介

Qwen3-TTS 基于 12 Hz 语音 tokenizer 生成 codec codes，再由 Code2Wav 解码为音频。vLLM-Omni 将 Talker 和 Code2Wav 组织为两阶段推理流水线，并提供 OpenAI 兼容的 `POST /v1/audio/speech` 接口。


## 模型列表

| 模型权重 | 任务类型 | vLLM-Omni 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| --- | --- | --- | --- | --- | --- | --- |
| [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base) | Base，参考音频音色复刻 | 0.21 | BW1000 / BW1100 | 1 | 单卡场景服务 | [**`>_`**](#06b-base) |
| [Qwen/Qwen3-TTS-12Hz-1.7B-Base](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | Base，参考音频音色复刻 | 0.21 | BW1000 / BW1100 | 1 | 单卡场景服务 | [**`>_`**](#17b-base) |
| [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | CustomVoice，预置音色 | 0.21 | BW1000 / BW1100 | 1 | 单卡场景服务 | [**`>_`**](#06b-customvoice) |
| [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | CustomVoice，预置音色 | 0.21 | BW1000 / BW1100 | 1 | 单卡场景服务 | [**`>_`**](#17b-customvoice) |
| [Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) | VoiceDesign，自然语言设计音色 | 0.21 | BW1000 / BW1100 | 1 | 单卡场景服务 | [**`>_`**](#17b-voicedesign) |


## 部署准备


推荐通用环境变量：

| 模型场景 | 启动前设置的性能环境变量 |
| --- | --- |
| 0.6B Base | `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1`、`TORCHINDUCTOR_MAX_AUTOTUNE=1` |
| 1.7B Base | `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1`、`TORCHINDUCTOR_MAX_AUTOTUNE=1` |
| 0.6B CustomVoice | `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1`、`TORCHINDUCTOR_MAX_AUTOTUNE=1` |
| 1.7B CustomVoice | `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1`、`TORCHINDUCTOR_MAX_AUTOTUNE=1`、`TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1` |
| 1.7B VoiceDesign | `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1`；不设置 TorchInductor 环境变量，使用 YAML 中 stage0 的 `cudagraph_mode: FULL` |

## 启动 Server

`--deploy-config` 接收 YAML 文件路径。请先将本文 [YAML 配置](#yaml-配置) 中对应内容保存为启动命令里的文件名；Base、CustomVoice 和 VoiceDesign 的配置关系见该章节。

### 0.6B Base

启动前按上表设置 0.6B Base 的性能环境变量，使用 `qwen3_tts_base.yaml`：

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --deploy-config ./qwen3_tts_base.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --omni
```

### 1.7B Base

启动前按上表设置 1.7B Base 的性能环境变量，使用 `qwen3_tts_base.yaml`：

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --deploy-config ./qwen3_tts_base.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --omni
```

### 0.6B CustomVoice

启动前按上表设置 0.6B CustomVoice 的性能环境变量。使用 `qwen3_tts_customvoice.yaml`；该文件以 Base YAML 为基础，增加 CustomVoice 与 VoiceDesign 的公共配置。

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --deploy-config ./qwen3_tts_customvoice.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --omni
```

### 1.7B CustomVoice

启动前按上表设置 1.7B CustomVoice 的三项性能环境变量。使用与 0.6B CustomVoice 相同的 `qwen3_tts_customvoice.yaml`。

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    --deploy-config ./qwen3_tts_customvoice.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --omni
```

### 1.7B VoiceDesign

启动前只设置 HIP sampler 开关，不设置 `TORCHINDUCTOR_*` 环境变量。使用 `qwen3_tts_voicedesign.yaml`；该文件以 Base YAML 为基础，先增加公共配置，再在 stage0 设置 `cudagraph_mode: FULL`。

```bash
vllm-omni serve Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
    --deploy-config ./qwen3_tts_voicedesign.yaml \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --omni
```

服务启动完成后可检查：

```bash
curl http://127.0.0.1:8000/health
```

## API 调用示例

### CustomVoice

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
      "input": "你好，欢迎使用 Qwen3-TTS。",
      "voice": "Vivian",
      "task_type": "CustomVoice",
      "language": "Chinese",
      "instructions": "语气自然、清晰。",
      "response_format": "wav",
      "stream": false
    }' \
    -o customvoice.wav
```

### VoiceDesign

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
      "input": "今天的天气很好，我们一起出发吧。",
      "task_type": "VoiceDesign",
      "language": "Chinese",
      "instructions": "年轻女声，语速自然，声音温暖且富有活力。",
      "response_format": "wav",
      "stream": false
    }' \
    -o voicedesign.wav
```

### Base 音色复刻

下面示例使用 Linux `base64 -w 0` 将本地 WAV 参考音频编码为 data URL。`ref_text` 应与参考音频内容一致。

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"Qwen/Qwen3-TTS-12Hz-1.7B-Base\",
      \"input\": \"这是使用参考音频复刻音色生成的语音。\",
      \"task_type\": \"Base\",
      \"language\": \"Chinese\",
      \"ref_text\": \"参考音频对应的文本内容。\",
      \"ref_audio\": \"data:audio/wav;base64,$(base64 -w 0 reference.wav)\",
      \"response_format\": \"wav\",
      \"stream\": false
    }" \
    -o base.wav
```

### 流式请求

流式接口返回持续输出的 PCM 数据。调用端应按服务端的采样率和通道数进行播放或封装：

```bash
curl -N -X POST http://127.0.0.1:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
      "input": "这是一次流式语音生成请求。",
      "voice": "Vivian",
      "task_type": "CustomVoice",
      "language": "Chinese",
      "response_format": "pcm",
      "stream": true
    }' \
    -o stream.pcm
```

是否流式由请求 JSON 中的 `"stream": true` 决定；`curl -o stream.pcm` 只决定把响应写入哪个本地文件，并不改变服务端的流式行为。

## YAML 配置

五个单卡场景实际只有两类 YAML：

- 两种 Base 模型共用同一份完整 Base YAML。
- 两种 CustomVoice 模型和 VoiceDesign 都以 Base YAML 为基础，只在两个 stage 增加 `distributed_executor_backend: mp`。

### Base 推荐配置

将以下内容保存为 `qwen3_tts_base.yaml`。行内 `#` 之后是注释，不影响 YAML 解析。

```yaml
async_chunk: true  # 两阶段流水线异步传递 codec 数据。
connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector  # Talker 与 Code2Wav 之间使用共享内存传递数据。
    extra:
      shm_threshold_bytes: 65536  # 大于此阈值的数据通过共享内存传递。
      codec_streaming: true       # 允许 Code2Wav 按音频 chunk 输出。
      connector_get_sleep_s: 0.01
      connector_get_max_wait_first_chunk: 3000
      connector_get_max_wait: 300
      codec_chunk_frames: 150           # non-stream 与 stream 稳态使用 150 帧解码窗口。
      codec_left_context_frames: 72     # Code2Wav 解码时保留的左侧上下文。
      initial_codec_chunk_frames: 1     # 首个音频包先使用较小窗口。
      decode_enable_tf32: true          # 允许 Code2Wav 解码路径使用 TF32。
      codec_chunk_schedule_frames:      # 仅 stream 请求按顺序扩大解码窗口。
      - 2
      - 8
      - 32
      - 150
stages:
- stage_id: 0
  max_num_seqs: 16
  gpu_memory_utilization: 0.3
  trust_remote_code: true
  enable_prefix_caching: false
  async_scheduling: true
  max_num_batched_tokens: 4096
  max_model_len: 4096
  devices: '0'
  enforce_eager: false
  output_connectors:
    to_stage_1: connector_of_shared_memory
  default_sampling_params:
    temperature: 0.9
    top_k: 50
    max_tokens: 4096
    seed: 42
    repetition_penalty: 1.05
  subtalker_sampling_params:
    do_sample: true
    temperature: 0.9
    top_k: 50
    top_p: 1.0
- stage_id: 1
  max_num_seqs: 16
  gpu_memory_utilization: 0.2
  trust_remote_code: true
  enable_prefix_caching: false
  async_scheduling: true
  max_num_batched_tokens: 16384
  max_model_len: 32768
  devices: '0'
  enforce_eager: true
  input_connectors:
    from_stage_0: connector_of_shared_memory
  default_sampling_params:
    temperature: 0.0
    top_p: 1.0
    top_k: -1
    max_tokens: 32768
    seed: 42
    repetition_penalty: 1.0
platforms:
  npu:
    stages:
    - stage_id: 0
      enforce_eager: true
```

### CustomVoice 与 VoiceDesign 推荐公共配置

将 Base YAML 分别复制为 `qwen3_tts_customvoice.yaml` 和 `qwen3_tts_voicedesign.yaml`。在两个 stage 的 `enforce_eager` 后都增加：

```yaml
distributed_executor_backend: mp
```

### VoiceDesign 额外配置

在公共配置基础上，VoiceDesign 仅在 stage0 的 `enforce_eager` 后增加：

```yaml
compilation_config:
  cudagraph_mode: FULL
```

除公共配置和上述字段外，VoiceDesign 使用与 Base 相同的 YAML 内容。

## 关键配置说明

| 配置 | 作用 |
| --- | --- |
| `QWEN3_TTS_USE_HIP_TOPK_SAMPLER=1` | 在 top-k 结果产生后，用 HIP kernel 完成 softmax 和按概率采样，减少小粒度采样阶段的框架开销。 |
| `TORCHINDUCTOR_MAX_AUTOTUNE=1` | 允许 TorchInductor 在编译阶段选择更合适的内核实现。 |
| `TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1` | 仅用于 1.7B CustomVoice，允许 Pointwise 内核参与 autotune。 |
| `cudagraph_mode: FULL` | 仅在 1.7B VoiceDesign 的 stage0 YAML 中设置，用完整 CUDA Graph 捕获模式替代 TorchInductor 环境变量。 |
| `codec_chunk_frames=150` | non-stream 和流式稳态的 Code2Wav 解码窗口大小。 |
| `codec_chunk_schedule_frames=[2, 8, 32, 150]` | 只在 stream 请求中逐步增大解码窗口，同时兼顾首包延迟与后续生成效率。 |
| `distributed_executor_backend=mp` | CustomVoice 和 VoiceDesign 的两个 stage 使用多进程执行后端。 |

## 注意事项

- 请求中的 `stream` 字段决定是否启用阶梯式 chunk：`stream=true` 使用 `[2, 8, 32, 150]`；`stream=false` 始终使用 `codec_chunk_frames=150`。
- 同一物理 DCU 上不要同时运行多个独立服务或 benchmark；性能测试前后应检查 HCU 利用率、显存和相关进程。
- `TORCHINDUCTOR_*` 变量仅按模型场景设置，会增加首次编译耗时；服务完成 warmup 后再进行性能测量。
