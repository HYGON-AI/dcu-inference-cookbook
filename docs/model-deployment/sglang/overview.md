# SGLang on DCU — 部署总览

## 安装

```bash
pip install "sglang[all]"
```

## 支持的模型

| 模型 | 参数量 | 文档 |
|------|--------|------|
| Qwen3 | 0.6B ~ 235B | [qwen3.md](qwen3.md) |
| Qwen3.5 | 7B ~ 72B+ | [qwen3.5.md](qwen3.5.md) |
| GLM-5 | 9B ~ 130B+ | [glm-5.md](glm-5.md) |
| Kimi-K2 | 1.5B ~ 72B | [kimi-k2.md](kimi-k2.md) |
| DeepSeek-R1 | 7B ~ 671B (MoE) | [deepseek-r1.md](deepseek-r1.md) |
| DeepSeek-V3.2 | 671B (MoE) | [deepseek-v3.2.md](deepseek-v3.2.md) |
| MiniMax-2.x | 456B (MoE) | [minimax-2.x.md](minimax-2.x.md) |

## 通用启动参数

```bash
python -m sglang.launch_server \
    --model-path <model_path> \
    --tp-size <N> \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--tp-size` | 张量并行卡数 | 视模型大小 |
| `--mem-fraction-static` | 静态显存分配比例 | 0.80 - 0.90 |
| `--context-length` | 最大上下文长度 | 按需设置 |
| `--enable-torch-compile` | torch.compile 加速 | 生产环境推荐 |

## 核心特性

### RadixAttention

SGLang 的核心创新，基于基数树（Radix Tree）的 KV Cache 管理：

- 自动复用公共前缀的 KV Cache
- 多轮对话中 system prompt 只计算一次
- Fork 机制支持并行生成

### 结构化生成

```python
import json
from sglang import function_call

# 定义输出 schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "number"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "score"],
}

response = client.generate(
    prompt="分析这个产品的特点",
    response_format={"type": "json_object", "schema": schema},
)
result = json.loads(response.text)
```

## 与 vLLM 对比

| 特性 | SGLang | vLLM |
|------|--------|------|
| KV Cache 复用 | RadixAttention (更强) | Prefix Caching |
| 结构化生成 | 原生支持 | 有限支持 |
| 多轮对话 | 更高效 | 一般 |
| VLM 支持 | ✅ | ✅ |
| Omni 全模态 | ❌ | ✅ |
| 社区生态 | 快速增长 | 更成熟 |
| DCU 适配 | ✅ | ✅ |

## 性能监控

```bash
hy-smi
watch -n 2 hy-smi
```

## 参考链接

- [SGLang 官方文档](https://sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
