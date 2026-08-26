# GLM-5.1

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [RedHatAI/GLM-5.1](https://www.modelscope.cn/models/RedHatAI/GLM-5.1) | BF16 | [0.5.12](../docker_images.md) | BW1100 | 16 | IFB | [**`>_`**](#redhatai-glm-51-ifb-bw1100-16x-sglang-0512) |
| [hygon/GLM-5.1-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.1-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-51-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
| [hygon/GLM-5.1-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.1-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-51-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|                                                                                                 | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-51-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8ep8cp8) |
| [hygon/GLM-5.1-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5.1-Channel-INT4-w4a8) | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-51-channel-int4-w4a8-ifb-bw1100-8x-sglang-0512) |
|                                                                                                 | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#glm-51-channel-int4-w4a8-ifb-bw1000-8x-sglang-0512) |

## 启动命令
 
### GLM-5.1-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
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
export W8A8_SUPPORT_METHODS=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.1-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --custom-all-reduce-backen aiter
~~~

### GLM-5.1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5.1-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --custom-all-reduce-backen aiter
~~~

### GLM-5.1-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8ep8cp8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
export HSA_ENABLE_COREDUMP=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export AMDGCN_USE_BUFFER_OPS=0
export ROCSHMEM_IB_GID_INDEX=0
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1

sglang serve \
  --model-path hygon/GLM-5.1-Channel-FP8-w8a8 \
  --trust-remote-code \
  --port 30001 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 100000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --attn-cp-size 8 \
  --nsa-prefill-cp-mode round-robin-split \
  --enable-nsa-prefill-context-parallel \
  --context-length 140000 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode auto \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --chunked-prefill-size 16384
~~~

### GLM-5.1-Channel-INT4-w4a8 IFB BW1100 8x SGLang 0.5.12

~~~bash
export NCCL_IB_GID_INDEX=3
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export SGLANG_USE_LIGHTOP=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export W8A8_SUPPORT_METHODS=3
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
sysctl -w kernel.numa_balancing=0 || true

sglang serve \
  --model-path hygon/GLM-5.1-Channel-INT4-w4a8 \
  --port 30000 \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --chunked-prefill-size 8192 \
  --quantization slimquant_w4a8_marlin \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --custom-all-reduce-backen aiter
~~~

### GLM-5.1-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.12

~~~bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_ENABLE_COREDUMP=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_USE_SVM=0
unset MC_GID_INDEX
unset NCCL_DEBUG
unset NCCL_DEBUG_FILE
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
unset NCCL_IB_GID_INDEX
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
unset RCCL_DEBUG
unset RCCL_DEBUG_SUBSYS
unset ROCSHMEM_ALLOWED_IBV_DEVICES
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MODELSCOPE=1
export SGLANG_W4A8_TPMOE_BACKEND=aiter
export TORCH_CPP_LOG_LEVEL=ERROR
unset TORCH_NCCL_DEBUG
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --context-length 32768 \
  --cuda-graph-max-bs 16 \
  --disable-radix-cache \
  --dist-timeout 10000 \
  --dp-size 1 \
  --dtype bfloat16 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 \
  --mem-fraction-static 0.90 \
  --model-path hygon/GLM-5.1-Channel-INT4-w4a8 \
  --moe-dense-tp-size 1 \
  --nsa-decode-backend flashmla_kv \
  --nsa-prefill-backend flashmla_auto \
  --page-size 64 \
  --quantization slimquant_w4a8_marlin \
  --reasoning-parser glm45 \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-num-steps 3 \
  --tool-call-parser glm47 \
  --tp-size 8 \
  --trust-remote-code \
  --watchdog-timeout 3600 \
  --port 30000
~~~


## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.1-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.1-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
