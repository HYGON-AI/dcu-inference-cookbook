# GLM-5.2

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.2-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#glm-52-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#glm-52-channel-int8-w8a8-ifb-bw1000-16x-sglang-0512) |
| [hygon/GLM-5.2-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8) | [**`>_`**](#glm-52-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|                                                                                                 | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8ep8) | [**`>_`**](#glm-52-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8ep8) |
|                                                                                                 | FP8 W8A8 | [0.5.12](../docker_images.md) | ScaleX40 | 24 | PD | [**`>_`**](#glm-52-channel-fp8-w8a8-pd-scalex40-24x-sglang-0512) |
| [hygon/GLM-5.2-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5.2-Channel-INT4-w4a8) | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#glm-52-channel-int4-w4a8-ifb-bw1100-4x-sglang-0512) |
|                                                                                                 | INT4 W4A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#glm-52-channel-int4-w4a8-ifb-bw1000-8x-sglang-0512) |

## 启动命令

### GLM-5.2-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
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
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.9
~~~

### GLM-5.2-Channel-INT8-w8a8 IFB BW1000 16x SGLang 0.5.12

#### Node 0

~~~bash
#!/bin/bash
set -e
export GLOO_SOCKET_IFNAME=ens61f1np1
export NCCL_SOCKET_IFNAME=ens61f1np1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
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
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 16 \
  --pp-size 1 \
  --nnodes 2 \
  --node-rank 0 \
  --dist-init-addr <P_node0_ip>:<port0> \
  --host xxxxx \
  --port <port1>  \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.8
~~~

#### Node 1

~~~bash
#!/bin/bash
set -e
export GLOO_SOCKET_IFNAME=ens61f1np1
export NCCL_SOCKET_IFNAME=ens61f1np1
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
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
  --model-path hygon/GLM-5.2-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 16 \
  --pp-size 1 \
  --nnodes 2 \
  --node-rank 1 \
  --dist-init-addr <P_node0_ip>:<port0> \
  --host xxxxx \
  --port <port1>  \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 64 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_marlin \
  --mem-fraction-static 0.8
~~~

### GLM-5.2-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
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
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
~~~

### GLM-5.2-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8ep8)

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
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
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_USE_DEEPGEMM_MOE=1

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 16 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --ep-size 8 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode auto
~~~

### GLM-5.2-Channel-FP8-w8a8 PD ScaleX40 24x SGLang 0.5.12
#### DeepEP 配置

以下 `ep_config.json` 仅作为参考配置。使用时请将其保存为 `ep_config.json`

```json
{
  "normal_dispatch": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 6,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  },
  "normal_combine": {
    "num_sms": 48,
    "num_max_nvl_chunked_send_tokens": 4,
    "num_max_nvl_chunked_recv_tokens": 256,
    "num_max_rdma_chunked_send_tokens": 6,
    "num_max_rdma_chunked_recv_tokens": 128
  }
}
```

#### EPLB 配置参考：[EPLB](../../optimization/static-eplb-sglang.md)

#### P node 0

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profiling
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
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_IPC_MNVL=1
export NCCL_IB_DISABLE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export AMDGCN_USE_BUFFER_OPS=0
export ROCSHMEM_IB_GID_INDEX=0
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --attn-cp-size 8 \
  --deepep-config /xxxxx/ep_config.json \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --disable-cuda-graph \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode normal \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 16 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --dist-init-addr <P_node0_ip>:<port0> \
  --nnodes 2 \
  --node-rank 0 \
  --host <P_node0_ip> \
  --port <port1> \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --dtype bfloat16 \
  --dist-timeout 100000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1
~~~

#### P node 1

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
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export DEEP_EP_NORMAL_MNVL=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
export PYTHONUNBUFFERED=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_IPC_MNVL=1
export NCCL_IB_DISABLE=0
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export AMDGCN_USE_BUFFER_OPS=0
export ROCSHMEM_IB_GID_INDEX=0
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048

sglang serve \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --attn-cp-size 8 \
  --deepep-config /xxxxx/ep_config.json \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --disable-cuda-graph \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --enable-dp-attention \
  --deepep-mode normal \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 16 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --disaggregation-mode prefill \
  --dist-init-addr <P_node0_ip>:<port0> \
  --nnodes 2 \
  --node-rank 1 \
  --host <P_node1_ip> \
  --port <port1> \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 8 \
  --dtype bfloat16 \
  --dist-timeout 100000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1
~~~

#### D node 0

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
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
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
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
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node0_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 0 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --ep-num-redundant-experts 64
~~~

#### D node 1

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
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
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
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
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node1_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 1 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --ep-num-redundant-experts 64
~~~

#### D node 2

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
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
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
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
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node2_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 2 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --ep-num-redundant-experts 64
~~~

#### D node 3

~~~bash
export NCCL_IB_DISABLE=1
export HIP_BUFFER_EXTRA_SIZE=0
export ROCSHMEM_GDR_DISABLE_XDP=1
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
export SGLANG_NSA_FUSE_TOPK=1
export SGLANG_USE_DEEPGEMM_MOE=1
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
export ROCSHMEM_HEAP_SIZE=4737418240
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export ROCSHMEM_IPC_MNVL=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS=shca_0,shca_1,shca_2,shca_4
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_4
export NCCL_SOCKET_IFNAME=em1
export GLOO_SOCKET_IFNAME=em1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1

sglang serve \
  --init-expert-location /xxxxx/expert_distribution.pt  \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host <D_node3_ip> \
  --port <port1> \
  --dist-init-addr <D_node0_ip>:<port0> \
  --nnodes 4 \
  --node-rank 3 \
  --tp-size 16 \
  --dp-size 16 \
  --ep-size 16 \
  --moe-dense-tp-size 1 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --enable-dp-lm-head \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length 524288 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_4 \
  --ep-num-redundant-experts 64
~~~

#### Router

~~~bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill \
  --decode http://decode \
  --policy round_robin \
  --port 30020
~~~


### GLM-5.2-Channel-INT4-w4a8 IFB BW1100 4x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
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
export SGLANG_W4A8_TPMOE_BACKEND=aiter

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 4 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_w4a8_marlin \
  --mem-fraction-static 0.9
~~~

### GLM-5.2-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.12

~~~bash
export SGLANG_ENABLE_SPEC_V2=1
export HSA_ENABLE_COREDUMP=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_USE_AITER_AR=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=0
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
export SGLANG_W4A8_TPMOE_BACKEND=aiter

sglang serve \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 16384 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 32 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization slimquant_w4a8_marlin \
  --mem-fraction-static 0.85
~~~

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.2-Channel-INT8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.2-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router。示例中 Router 端口为 `30020`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30020/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.2-Channel-FP8-w8a8",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=2048,
)
```

```bash
curl http://<router_ip>:30020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5.2-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
