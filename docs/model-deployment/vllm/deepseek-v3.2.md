# Deepseek-v3.2-w8a8 on vLLM

## 模型简介

DeepSeek V3.2 是深度求索公司于 2025 年底发布的大语言模型，基于创新的 DeepSeek Sparse Attention (DSA) 稀疏注意力机制，支持 128K 上下文。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| Deepseek-v3.2 | 671B | 128K | 8x BW1100 144GB |

## 启动命令

### Deepseek-v3.2-w8a8（八卡 144G）

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     
export ALLREDUCE_STREAM_WITH_COMPUTE=1           
export NCCL_MIN_NCHANNELS=16                      
export NCCL_MAX_NCHANNELS=16                   
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1   
export VLLM_RPC_TIMEOUT=1800000  

export VLLM_SPEC_DECODE_EAGER=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=0
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 
export VLLM_DISABLE_DSA=0
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_CAT_MLA=1

vllm serve hygon/DeepSeek-V3.2-Channel-INT8-w8a8 \
 --dtype bfloat16 \
 --trust-remote-code \
 --max-model-len 40960 \
 -tp 8 \
 --gpu-memory-utilization 0.92 \
 --max-num-seqs 256 \
 --disable-log-requests \
 --enable-chunked-prefill \
 --max-num-batched-tokens 16384 \
 --no-enable-prefix-caching \
 -cc '{"pass_config": {"fuse_act_quant": false}}' \
 --speculative_config '{"method": "mtp", "num_speculative_tokens": 2}' \
 --kv-cache-dtype fp8_ds_mla

```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V3.2-Channel-INT8-w8a8",
    messages=[
        {"role": "user", "content": "请总结以下长文档的关键要点..."},
    ],
    max_tokens=4096,
    temperature=0,

)
print(response.choices[0].message.content)
 
```

## curl 示例

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json"  \
    -d '{
        "model": "hygon/DeepSeek-V3.2-Channel-INT8-w8a8", 
        "messages": [{"role": "user", "content": "中国的首都是什么？"}], 
        "temperature": 0, 
        "max_tokens": 100
    }'
```