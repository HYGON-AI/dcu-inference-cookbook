# Gemma-4 on vLLM

## 模型简介

Gemma-4-31B-it 是 Gemma 系列指令模型，本文档提供其在 vLLM 上的部署示例。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/gemma-4-31B-it](https://www.modelscope.cn/models/hygon/gemma-4-31B-it) | BF16 | 0.21 | BW1000 | 1 | IFB | [**`>_`**](#gemma-4-31b-it-ifb-bw1000-1x-vllm-021) |

## 启动命令

### Gemma-4-31B-it IFB BW1000 2x vLLM 0.25

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_KV_CACHE_LAYOUT=NHD

vllm serve \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --attention-backend TRITON_ATTN \
  --hf-overrides '{"text_config":{"allow_global_per_layer_attribute_access":true,"global_head_dim":512,"num_global_key_value_heads":4,"use_bidirectional_attention":null}}' \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4
```

### Gemma-4-31B-it IFB BW1000 1x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/gemma-4-31B-it \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --reasoning-parser gemma4 \
    --chat-template tool_chat_template_gemma4.jinja \
    --attention-backend FLASH_ATTN_CUTLASS
```

## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hygon/gemma-4-31B-it",
        "messages": [{"role": "user", "content": "中国的首都是什么？"}],
        "temperature": 0,
        "max_tokens": 400
    }'
```
