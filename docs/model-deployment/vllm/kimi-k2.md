# Kimi-K2 on vLLM

## 模型简介

Kimi-K2 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2-Instruct](https://www.modelscope.cn/models/moonshotai/Kimi-K2-Instruct) | FP8 W8A8 | 0.15 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k2-instruct-ifb-bw1100-16x-vllm-015) |

## 启动命令

### Kimi-K2-Instruct IFB BW1100 16x vLLM 0.15

```bash
export VLLM_USE_MODELSCOPE=1

vllm serve moonshotai/Kimi-K2-Instruct \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="moonshotai/Kimi-K2-Instruct",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "moonshotai/Kimi-K2-Instruct", "messages": [{"role": "user", "content": "中国的首都是什么？"}], "max_tokens": 128}'
```

## DCU 适配注意

- Kimi-K2-Instruct 原生支持 bf16
- 超长上下文（>32K）场景 KV Cache 占用大，建议适当降低 `--max-model-len`
