# vLLM on DCU

## 安装

```bash
# 从源码安装（推荐，确保 DCU 兼容）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 或 pip 安装（需确认 DCU 兼容版本）
pip install vllm
```

## LLM 推理

### 单卡

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trust-remote-code
```

### 多卡张量并行

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --trust-remote-code
```

## VLM 推理（视觉语言模型）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --limit-mm-per-prompt image=5
```

VLM API 调用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
            {"type": "text", "text": "描述这张图片"},
        ],
    }],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

## vLLM Omni（全模态推理）

vLLM Omni 支持任意模态输入输出（文本、图像、音频、视频），通过统一的 API 接口实现多模态推理。

### 安装

```bash
pip install vllm  # vLLM 0.7+ 内置 Omni 支持
```

### 启动 Omni 服务

```bash
# Qwen2.5-Omni — 支持文本 + 音频输入/输出
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Omni-7B \
    --trust-remote-code \
    --limit-mm-per-prompt image=5,audio=3

# UltraVox — 语音理解模型
python -m vllm.entrypoints.openai.api_server \
    --model fixie-ai/ultravox-v0_5-8b \
    --trust-remote-code \
    --limit-mm-per-prompt audio=1
```

### 语音对话 API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 语音输入 → 语音输出
with open("question.wav", "rb") as f:
    audio_data = f.read()

import base64
audio_b64 = base64.b64encode(audio_data).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Omni-7B",
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
            {"type": "text", "text": "请用语音回答这个问题"},
        ],
    }],
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### 图像 + 文本混合输入

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-Omni-7B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/chart.png"}},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
            {"type": "text", "text": "分析这张图表并用语音总结"},
        ],
    }],
    max_tokens=1024,
)
```

### Omni 支持的模态

| 模态 | 输入 | 输出 | 支持模型 |
|------|------|------|---------|
| 文本 | ✅ | ✅ | 所有 LLM/VLM |
| 图像 | ✅ | ❌ | Qwen2.5-VL, LLaVA, InternVL |
| 音频 | ✅ | ✅ | Qwen2.5-Omni, UltraVox |
| 视频 | ✅ | ❌ | Qwen2.5-VL (逐帧) |

## OpenAI 兼容 API

vLLM 提供完整的 OpenAI 兼容接口，支持：

- `POST /v1/chat/completions` — 对话补全
- `POST /v1/completions` — 文本补全
- `POST /v1/embeddings` — 向量嵌入
- `GET /v1/models` — 模型列表

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好！"},
    ],
    max_tokens=128,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

## 高级特性

### 前缀缓存

```bash
--enable-prefix-caching
```

### 流式输出

```python
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "讲个故事"}],
    max_tokens=512,
    stream=True,
):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### KV Cache 量化

```bash
--kv-cache-dtype int8
```

## 已知限制

- 部分 CUDA 专属算子需要 DTK 适配
- Flash Attention 需要确认 DCU 兼容版本
- 某些量化方法（AWQ/GPTQ）在 DCU 上可能有兼容性问题
- VLM 多图场景显存消耗较大，注意 `--limit-mm-per-prompt` 设置
- Omni 音频模态依赖特定编解码器，需确认 DCU 兼容性

## 参考链接

- [vLLM 官方文档](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
