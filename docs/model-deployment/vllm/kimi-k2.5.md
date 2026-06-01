# Kimi-K2.5 on vLLM

## 模型简介

Kimi-K2.5 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.5](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5) | INT4 W4A16 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k25-ifb-bw1100-8x-vllm-015) |

## 启动命令

### Kimi-K2.5 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export VLLM_USE_MODELSCOPE=1
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

vllm serve moonshotai/Kimi-K2.5 \
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

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json"  \
  -d '{
      "model": "moonshotai/Kimi-K2.5", 
      "messages": [{"role": "user", "content": "中国的首都是什么？"}], 
      "temperature": 0, 
      "max_tokens": 400
  }'
```
