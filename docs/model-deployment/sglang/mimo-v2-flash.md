# MiMo-V2-Flash on SGLang

## 模型简介

MiMo-V2-Flash 是小米推出的大规模 MoE（混合专家）语言模型，总参数量309B，采用混合注意力和 256 experts + top-8 路由策略，支持 262K 超长上下文。模型原生支持 EAGLE（MTP）投机解码，可显著提升 decode 吞吐。

## 模型列表

| 模型 | 量化方式 | 推荐硬件 |
|------|---------|---------|
| MiMo-V2-Flash | FP8 block (原始) | 8x DCU BW1100 |
| MiMo-V2-Flash-Channel-FP8-w8a8 | FP8 per-channel (MoE) | 8x DCU BW1100 |

## 镜像

```bash
10.16.6.35:5000/jenkins/model_test_env/sglang:daily-20260416-1454
```
## MiMo-V2-Flash启动命令(IFB)

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --page-size 64 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --disable-radix-cache \
    --context-length 262144 \
    --attention-backend triton \
    --chunked-prefill-size -1 \
    --enable-dp-attention \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

## MiMo-V2-Flash-Channel-FP8-w8a8启动命令(IFB)

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --page-size 64 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --disable-radix-cache \
    --context-length 262144 \
    --attention-backend triton \
    --chunked-prefill-size -1 \
    --enable-dp-attention \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="/data1/model/MiMo-V2-Flash",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"},
    ],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
```

## curl 调用
```bash
curl -X 'POST' \
    'http://localhost:30000/v1/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
    "model": "/data1/model/MiMo-V2-Flash",
    "prompt": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？",
    "max_tokens": 128,
    "stream": false,
    "ignore_eos": false,
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 1
    }'
```

