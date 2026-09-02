# GLM-5.2 on SGLang

## 模型简介

GLM-5.2 智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.2-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.12| BW1000 | 16 | IFB | [**`>_`**](#glm-52-channel-int4-w4a8-ifb-bw1000-16x-sglang-0512)  |
|                                                                                                 | INT4 W4A8 | 0.5.12| BW1000 | 32 | 2P2D| [**`>_`**](#glm-52-channel-int4-w4a8-2p2d-bw1000-32x-sglang-0512) |
| [hygon/GLM-5.2-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5.1-Channel-FP8-w8a8) | INT4 W4A8 | 0.5.12| BW1000 | 16 | IFB | [**`>_`**](#glm-52-channel-fp8-w8a8-1p1d-bw1000-16x-sglang-0512)  |

## 启动命令

### GLM-5.2-Channel-INT4-w4a8 IFB BW1000 16x SGLang 0.5.12

```bash
#!/usr/bin/env bash
set -euo pipefail

# IFB 环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GLOO_SOCKET_IFNAME=ib0
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
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=ib0
export PYTHONWARNINGS=ignore
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
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

# 节点配置。
# group=0, node_rank=0, host=10.2.14.71, master=10.2.14.71
exec python -m sglang.launch_server \
  --chunked-prefill-size -1 \
  --context-length 131072 \
  --cuda-graph-max-bs 16 \
  --disable-radix-cache \
  --dist-timeout 10000 \
  --dp-size 1 \
  --dtype bfloat16 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 256 \
  --mem-fraction-static 0.80 \
  --model-path hygon//GLM-5.2-Channel-INT4-w4a8 \
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
  --host 10.2.14.71 \
  --port 30000 \
  --dist-init-addr 10.2.14.71:5002 \
  --nnodes 2 \
  --node-rank 0
```

### GLM-5.2-Channel-INT4-w4a8 2P2D BW1000 32x SGLang 0.5.12

#### P node 0

```bash
#!/usr/bin/env bash
# 环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GLOO_SOCKET_IFNAME=eno1
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
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=8388608
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=eno1
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

# 节点配置。
# role=prefill, node_rank=0, host=10.2.14.71, master=10.2.14.71
exec python -m sglang.launch_server \
  --attn-cp-size 8 \
  --chunked-prefill-size 16384 \
  --context-length 133120 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode prefill \
  --dtype bfloat16 \
  --enable-nsa-prefill-context-parallel \
  --kv-cache-dtype fp8_e4m3 \
  --max-prefill-tokens 65536 \
  --mem-fraction-static 0.75 \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --nsa-decode-backend flashmla_kv \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-prefill-cp-mode round-robin-split \
  --page-size 64 \
  --pp-max-micro-batch-size 1 \c
  --pp-size 2 \
  --quantization slimquant_w4a8_marlin \
  --tp-size 8 \
  --trust-remote-code \
  --host 10.2.14.71 \
  --port 30000 \
  --dist-init-addr 10.2.14.71:5000 \
  --nnodes 2 \
  --node-rank 0

```
                                                                                                        
#### P node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.2.14.71`。

```bash
#!/usr/bin/env bash
# IFB 环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GLOO_SOCKET_IFNAME=ib0
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
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=ib0
export PYTHONWARNINGS=ignore
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
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

# 节点配置。
# group=0, node_rank=1, host=10.2.14.72, master=10.2.14.71
exec python -m sglang.launch_server \
  --chunked-prefill-size -1 \
  --context-length 131072 \
  --cuda-graph-max-bs 16 \
  --disable-radix-cache \
  --dist-timeout 10000 \
  --dp-size 1 \
  --dtype bfloat16 \
  --ep-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 256 \
  --mem-fraction-static 0.80 \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
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
  --host 10.2.14.72 \
  --port 30000 \
  --dist-init-addr 10.2.14.71:5002 \
  --nnodes 2 \
  --node-rank 1
```

#### D node 0

```bash
#!/usr/bin/env bash
# 环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GLOO_SOCKET_IFNAME=eno1
export GPU_FORCE_BLIT_COPY_SIZE=16
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_ENABLE_COREDUMP=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_USE_SVM=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=eno1
export PYTHONWARNINGS=ignore
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_HEAP_SIZE=1073741824
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_TOPO_FILE_FORCE=/home/shca_topo.config
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MODELSCOPE=1
export TORCH_CPP_LOG_LEVEL=ERROR
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

# 节点配置。
# role=decode, node_rank=0, host=10.2.14.73, master=10.2.14.73
exec python -m sglang.launch_server \
  --chunked-prefill-size -1 \
  --context-length 133120 \
  --cuda-graph-max-bs 8 \
  --deepep-mode low_latency \
  --disable-radix-cache \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --dist-timeout 10000 \
  --dp-size 16 \
  --dtype bfloat16 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --ep-size 16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 256 \
  --mem-fraction-static 0.75 \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --nsa-decode-backend flashmla_kv \
  --page-size 64 \
  --quantization slimquant_w4a8_marlin \
  --reasoning-parser glm45 \
  --served-model-name glm \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-num-steps 3 \
  --tool-call-parser glm47 \
  --tp-size 16 \
  --trust-remote-code \
  --watchdog-timeout 3600 \
  --host 10.2.14.73 \
  --port 30000 \
  --dist-init-addr 10.2.14.73:5000 \
  --nnodes 2 \
  --node-rank 0
```

#### D node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.2.14.73`。

```bash
#!/usr/bin/env bash
# 环境变量
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export DEEPEP_ENABLE_LL_LAYERED_OPT=1
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GLOO_SOCKET_IFNAME=eno1
export GPU_FORCE_BLIT_COPY_SIZE=16
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_ENABLE_COREDUMP=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_USE_SVM=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_NET_PLUGIN=shca
export NCCL_PLUGIN_P2P=ib
export NCCL_SOCKET_IFNAME=eno1
export PYTHONWARNINGS=ignore
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_HEAP_SIZE=1073741824
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_TOPO_FILE_FORCE=/home/shca_topo.config
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_USE_AITER_AR=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_MODELSCOPE=1
export TORCH_CPP_LOG_LEVEL=ERROR
export USE_DCU_CUSTOM_ALLREDUCE=0
export W8A8_SUPPORT_METHODS=3

# 节点配置。
# role=decode, node_rank=1, host=10.2.14.74, master=10.2.14.73
exec python -m sglang.launch_server \
  --chunked-prefill-size -1 \
  --context-length 133120 \
  --cuda-graph-max-bs 8 \
  --deepep-mode low_latency \
  --disable-radix-cache \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode decode \
  --dist-timeout 10000 \
  --dp-size 16 \
  --dtype bfloat16 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --ep-size 16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 256 \
  --mem-fraction-static 0.75 \
  --model-path hygon/GLM-5.2-Channel-INT4-w4a8 \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --nsa-decode-backend flashmla_kv \
  --page-size 64 \
  --quantization slimquant_w4a8_marlin \
  --reasoning-parser glm45 \
  --served-model-name glm \
  --speculative-algorithm EAGLE \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --speculative-num-steps 3 \
  --tool-call-parser glm47 \
  --tp-size 16 \
  --trust-remote-code \
  --watchdog-timeout 3600 \
  --host 10.2.14.74 \
  --port 30000 \
  --dist-init-addr 10.2.14.73:5000 \
  --nnodes 2 \
  --node-rank 1
```

#### Router

```bash
sglang-router launch \
  --host 0.0.0.0 \
  --port 30002 \
  --pd-disaggregation \
  --prometheus-port 29001 \
  --prefill http://10.2.14.71:30000 8998 \
  --decode http://10.2.14.73:30000 \
```

### GLM-5.2-Channel-FP8-w8a8 1P1D BW1000 16x SGLang 0.5.12
#### P node 0
```bash
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

export SGLANG_NSA_HCU_REUSE_SORTED_TOPK=1
export SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB=2
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=2048

export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_USE_MODELSCOPE=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200

export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824

sglang serve \
    --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
    --trust-remote-code \
    --served-model-name GLM-5.2-Channel-FP8-w8a8 \
    --host 10.41.101.127 \
    --port 30000 \
    --dist-init-addr "10.41.101.127:5001" \
    --nnodes 1 \
    --node-rank 0 \
    --tp-size 8 \
    --ep-size 8 \
    --attn-cp-size 8 \
    --moe-dense-tp-size 1 \
    --moe-a2a-backend deepep \
    --deepep-mode normal \
    --enable-single-batch-overlap \
    --enable-nsa-prefill-context-parallel \
    --nsa-prefill-cp-mode round-robin-split \
    --nsa-prefill-backend flashmla_auto \
    --nsa-decode-backend flashmla_kv \
    --dtype bfloat16 \
    --dist-timeout 10000 \
    --watchdog-timeout 3600 \
    --page-size 64 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.9 \
    --disable-radix-cache \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --max-running-requests 128 \
    --disable-cuda-graph \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake
```

#### D node 0
```bash

export SGLANG_NSA_HCU_PERSISTENT_MQA_FASTPATH=1
export SGLANG_NSA_HCU_PERSISTENT_MQA_CTAS=0

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

export SGLANG_USE_LIGHTOP_MASK_TOPK=1
export SGLANG_ENABLE_HCU_CONCAT_MLA_ABSORB_Q=1

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
export ROCSHMEM_IPC_MNVL=1

export NCCL_SOCKET_IFNAME=enp23s0u2c2
export GLOO_SOCKET_IFNAME=enp23s0u2c2
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_MODELSCOPE=1
export W8A8_SUPPORT_METHODS=3
export GPU_MAX_HW_QUEUES=3
export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1

sglang serve \
    --model-path hygon/GLM-5.2-Channel-FP8-w8a8 \
    --trust-remote-code \
    --served-model-name GLM-5.2-Channel-FP8-w8a8 \
    --host 10.41.101.126 \
    --port 30023 \
    --dist-init-addr 10.41.101.126:5000 \
    --nnodes 1 \
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
    --speculative-algorithm EAGLE \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --cuda-graph-max-bs 16 \
    --max-running-requests 256 \
    --disaggregation-mode decode \
```
