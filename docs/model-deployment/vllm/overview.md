# vLLM on DCU — 部署总览

## 安装

```bash
# 从源码安装（推荐，确保 DCU 兼容）
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .

# 或 pip 安装（需确认 DCU 兼容版本）
pip install vllm
```

## 支持的模型

| 模型 | 参数量 | 文档 |
|------|--------|------|
| Qwen3 | 0.6B ~ 235B | [qwen3.md](qwen3.md) |
| Qwen3.5 | 7B ~ 72B+ | [qwen3.5.md](qwen3.5.md) |
| GLM-5 | 9B ~ 130B+ | [glm-5.md](glm-5.md) |
| Kimi-K2 | 1.5B ~ 72B | [kimi-k2.md](kimi-k2.md) |
| MiniMax-2.x | 456B (MoE) | [minimax-2.x.md](minimax-2.x.md) |

## 通用启动参数

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <model_path> \
    --tensor-parallel-size <N> \
    --max-model-len <length> \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --dtype bfloat16
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--tensor-parallel-size` | 张量并行卡数 | 视模型大小 |
| `--max-model-len` | 最大上下文长度 | 按需，影响显存 |
| `--gpu-memory-utilization` | 显存利用率 | 0.90 - 0.95 |
| `--max-num-seqs` | 最大并发序列数 | 128 - 512 |
| `--dtype` | 数据类型 | bfloat16 |
| `--enable-prefix-caching` | 前缀缓存 | 推荐开启 |
| `--kv-cache-dtype int8` | KV Cache 量化 | 显存紧张时 |

## OpenAI 兼容 API

vLLM 提供完整的 OpenAI 兼容接口：

- `POST /v1/chat/completions` — 对话补全
- `POST /v1/completions` — 文本补全
- `POST /v1/embeddings` — 向量嵌入
- `GET /v1/models` — 模型列表

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="<model_name>",
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
    model="<model_name>",
    messages=[{"role": "user", "content": "讲个故事"}],
    max_tokens=512,
    stream=True,
):
    print(chunk.choices[0].delta.content, end="", flush=True)
```

### vLLM Omni（全模态）

```bash
# Qwen2.5-Omni — 文本+图像+音频
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Omni-7B \
    --trust-remote-code \
    --limit-mm-per-prompt image=5,audio=3
```

详见 → [../../frameworks/vllm-dcu.md](../../frameworks/vllm-dcu.md#vllm-omni全模态推理)

## 性能监控

```bash
# 查看 DCU 利用率
hy-smi

# 持续监控
watch -n 2 hy-smi
```

## 已知限制

- 部分 CUDA 专属算子需要 ROCm 适配
- Flash Attention 需要确认 DCU 兼容版本
- 某些量化方法（AWQ/GPTQ）在 DCU 上可能有兼容性问题
- VLM 多图场景显存消耗较大，注意 `--limit-mm-per-prompt` 设置
- Omni 音频模态依赖特定编解码器，需确认 DCU 兼容性
