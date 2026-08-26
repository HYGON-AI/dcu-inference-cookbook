# DeepSeek-V3 on SGLang

## 模型简介

DeepSeek-V3-0324 是 DeepSeek V3 系列的 MoE 大模型版本，适用于对话、推理与代码生成场景。本页提供其 INT8 W8A8 量化模型在 HCU 平台上的 SGLang IFB 部署方法。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V3-0324-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3-0324-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 144GB | 8 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
|  | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 64GB | 16 | IFB | [**`>_`**](#deepseek-v3-0324-channel-int8-w8a8-ifb-bw1000-16x-sglang-0512) |

## 启动命令

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-running-requests 32 \
  --context-length 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --cuda-graph-max-bs 32 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.9 \
  --attention-backend hcu_mla
```

### DeepSeek-V3-0324-Channel-INT8-w8a8 IFB BW1000 16x SGLang 0.5.12


#### Node 0

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \ #按照实际
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-running-requests 32 \
  --context-length 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --cuda-graph-max-bs 32 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --host xxxxx\
  --dist-init-addr "<node0_ip>:xxxxx" \
  --nnodes 2 \
  --node-rank 0 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla
```

#### Node 1

`--dist-init-addr` 中的 `<node0_ip>` 请替换为 Node 0 的实际 IP。

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export SGLANG_USE_FUSED_RMS_QUANT=0
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/DeepSeek-V3-0324-Channel-INT8-w8a8 \
  --numa-node 0 0 0 0 1 1 1 1 \ #按照实际
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --max-running-requests 32 \
  --context-length 32768 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 1 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 2 \
  --cuda-graph-max-bs 32 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --host xxxxx\
  --dist-init-addr "<node0_ip>:xxxxx" \
  --nnodes 2 \
  --node-rank 1 \
  --dtype bfloat16 \
  --tp-size 8 \
  --pp-size 1 \
  --mem-fraction-static 0.85 \
  --attention-backend hcu_mla
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V3-0324-Channel-INT8-w8a8",
    messages=[
        {"role": "user", "content": "请介绍一下你自己。"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"hygon/DeepSeek-V3-0324-Channel-INT8-w8a8","messages":[{"role":"user","content":"请介绍一下你自己。"}],"max_tokens":128}'
```
