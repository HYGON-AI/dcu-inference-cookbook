# DeepSeek-R1 on SGLang

## 模型简介

DeepSeek-R1 是 DeepSeek 推出的推理强化模型系列，面向复杂推理、数学与代码场景。SGLang 在 HCU 平台可通过张量并行部署 R1 系列模型，并兼容 OpenAI API 接口。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | BF16 | 0.5.12 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1100-8x-sglang-0512) |
|  | BF16 | 0.5.12 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1000-8x-sglang-0512) |
|  | BF16 | 0.5.12 | K100_AI | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-k100_ai-4x-sglang-0512) |
| [hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.12 | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-bw1100-2x-sglang-0512) |
|  | INT8 W8A8 | 0.5.12 | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-bw1000-4x-sglang-0512) |
|  | INT8 W8A8 | 0.5.12 | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-channel-int8-w8a8-ifb-k100_ai-8x-sglang-0512) |
| [hygon/DeepSeek-R1-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 8x | IFB | [**\`>_\`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510) |

## 启动命令

### DeepSeek-R1-Distill-Llama-70B IFB BW1100 8x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1000 8x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192
```

### DeepSeek-R1-Distill-Llama-70B IFB K100_AI 4x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

sglang serve \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --trust-remote-code \
  --tp-size 4 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB BW1100 2x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3
export HIP_VISIBLE_DEVICES=6,7

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 2 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB BW1000 4x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 4 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 IFB K100_AI 8x SGLang 0.5.12

```bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_GID_INDEX=3

sglang serve \
  --model-path hygon/DeepSeek-R1-Distill-Llama-70B-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --attention-backend fa3 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --disable-custom-all-reduce \
  --quantization slimquant_marlin
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.10
```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1

export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

export HIP_H2D_DISABLE_COPY_BUFFER=0 # 同步异步强制走WriteBuffer
export HIP_D2H_DISABLE_COPY_BUFFER=0 # 同步异步强制走ReadBuffer
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 # 小于此值走CPUCopy
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 # 大于此值走HSACOPY（CopyBuffer）
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 # 小于此值走CPUCopy
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 # 大于此值走HSACOPY（CopyBuffer）

export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072

export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

python3 -m sglang.launch_server \
    --model-path hygon/DeepSeek-R1-Channel-FP8-w8a8 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests 256 \
    --speculative-algorithm EAGLE --speculative-num-steps 2  --speculative-eagle-topk 1  --speculative-num-draft-tokens 1 \
    --context-length 65536 \
    --quantization w8a8_fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --dtype bfloat16 \
    --tp-size 8 \
    --pp-size 1 \
    --mem-fraction-static 0.9 \
    --attention-backend dcu_mla

```


## API 调用

### IFB

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Channel-FP8",
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请给出一个高并发服务限流方案。"}
    ]
  }'
```
