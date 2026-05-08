# Kimi-K2.5 on vLLM

## 模型简介

Kimi-K2.5 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| Kimi-K2.5 | 1T | 256K | IFB:8x BW1100 144GB |

## 启动命令

### IFB【TP8】

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_LIGHTOP=1
export VLLM_USE_PIECEWISE=1
export VLLM_1D_MROPE=1
export USE_FUSED_RMS_QUANT=0 
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_W8A8_BACKEND=3
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_CAT_MLA=1 
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve /module/Kimi-K2.5/ \
    -tp 8 \
    --trust-remote-code   \
    --dtype bfloat16  \
    --max-model-len 65536  \
    --disable-log-requests  \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3
```
## API 调用

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "/module/Kimi-K2.5/", 
        "messages": [{"role": "user", "content": "中国的首都是什么？"}], 
        "temperature": 0, 
        "max_tokens": 400
    }'
```
