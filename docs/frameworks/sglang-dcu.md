# SGLang on DCU

## 安装

```bash
pip install "sglang[all]"
```

## LLM 推理

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --trust-remote-code
```

## VLM 推理（视觉语言模型）

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-VL-7B-Instruct \
    --tp-size 1 \
    --trust-remote-code
```

## 核心特性

### RadixAttention

SGLang 的核心创新，基于基数树（Radix Tree）的 KV Cache 管理：

- 自动复用公共前缀的 KV Cache
- 多轮对话中 system prompt 只计算一次
- Fork 机制支持并行生成

### 结构化生成

```python
from sglang import function_call

# 定义输出 schema
schema = {
    "name": "str",
    "age": "int",
    "skills": ["str"],
}

# 约束生成
response = client.generate(
    prompt="介绍一个程序员",
    response_format=schema,
)
```

## 性能调优

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-72B-Instruct \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --enable-torch-compile
```

## 与 vLLM 对比

| 特性 | SGLang | vLLM |
|------|--------|------|
| KV Cache 复用 | RadixAttention (更强) | Prefix Caching |
| 结构化生成 | 原生支持 | 有限支持 |
| 多轮对话 | 更高效 | 一般 |
| VLM 支持 | ✅ | ✅ |
| 社区生态 | 快速增长 | 更成熟 |
| DCU 适配 | ✅ | ✅ |

## 参考链接

- [SGLang 官方文档](https://sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
