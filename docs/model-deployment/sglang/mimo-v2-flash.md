# MiMo-V2-Flash on SGLang

## 模型简介

MiMo-V2-Flash 是小米推出的大规模 MoE（混合专家）语言模型，总参数量309B，采用混合注意力和 256 experts + top-8 路由策略，支持 262K 超长上下文。模型原生支持 EAGLE（MTP）投机解码，可显著提升 decode 吞吐。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [XiaomiMiMo/MiMo-V2-Flash](https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2-Flash) | BF16 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#mimo-v2-flash-ifb-bw1100-8x-sglang-0510) |
| [hygon/MiMo-V2-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510) |
| [hygon/MiMo-V2-Flash-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiMo-V2-Flash-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1000 | 8 | IFB | [**`>_`**](#mimo-v2-flash-channel-int8-w8a8-ifb-bw1000-8x-sglang-0510) |

## 启动命令

### MiMo-V2-Flash IFB BW1100 8x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --page-size 64 \
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

### MiMo-V2-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.10

```bash
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-FP8-w8a8 \
    --pp-size 1 \
    --dp-size 2 \
    --tp-size 8 \
    --page-size 64 \
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

### MiMo-V2-Flash-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.10

```bash
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export HSA_NO_SCRATCH_RECLAIM=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_USE_MODELSCOPE=1

sglang serve \
    --model-path hygon/MiMo-V2-Flash-Channel-INT8-w8a8 \
    --pp-size 1 \
    --dp-size 1 \
    --tp-size 8 \
    --page-size 64 \
    --trust-remote-code \
    --mem-fraction-static 0.75 \
    --quantization slimquant_marlin \
    --max-running-requests 128 \
    --tool-call-parser mimo \
    --context-length 262144 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --port 11010 \
    --attention-backend triton \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="XiaomiMiMo/MiMo-V2-Flash",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"},
    ],
    max_tokens=128,
    temperature=0,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "XiaomiMiMo/MiMo-V2-Flash",
    "messages": [
      {"role": "system", "content": "你是一个专业的 AI 助手。"},
      {"role": "user", "content": "甲乙两班共有学生98人，甲班比乙班多6人，求两班各有多少人？"}
    ],
    "max_tokens": 128,
    "temperature": 0
  }'
```

