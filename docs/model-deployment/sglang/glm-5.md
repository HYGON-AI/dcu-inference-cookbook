# GLM-5 on SGLang

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.10 | BW1000 |  8 | IFB | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x-sglang-0510) |
|                                                                                                 | INT4 W4A8 | 0.5.10 | BW1000 | 32 | 2P2D| [**`>_`**](#glm-5-channel-int4-w4a8-2p2d-bw1000-32x-sglang-0510) |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1100 |  8 | IFB | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-sglang-0510) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D| [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-sglang-0510) |
| [hygon/GLM-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-FP8-w8a8)   |  FP8 W8A8 | 0.5.10 | BW1100 |  8 | IFB | [**`>_`**](#glm-5-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510)  |
|                                                                                                 |  FP8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D| [**`>_`**](#glm-5-channel-fp8-w8a8-1p2d-bw1100-24x-sglang-0510)  |

## 启动命令

### GLM-5-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.10

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 8 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_sparse \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.8 \
  --chunked-prefill-size 8192 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
```

### GLM-5-Channel-INT4-w4a8 2P2D BW1000 32x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 8 \
  --pp-size 2 \
  --attn-cp-size 8 \
  --pp-max-micro-batch-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --context-length 131072 \
  --kv-cache-dtype bf16 \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 65536 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --quantization slimquant_w4a8_marlin \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --disaggregation-mode prefill
```

#### P node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.16.1.36`。

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "10.16.1.36:5000" \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 8 \
  --pp-size 2 \
  --attn-cp-size 8 \
  --pp-max-micro-batch-size 1 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --context-length 131072 \
  --kv-cache-dtype bf16 \
  --dtype bfloat16 \
  --mem-fraction-static 0.9 \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 65536 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --quantization slimquant_w4a8_marlin \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --disaggregation-mode prefill
```

#### D node 0

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --nnodes 2 \
  --node-rank 0 \
  --tp-size 16 \
  --moe-dense-tp-size 1 \
  --dp-size 16 \
  --ep-size 16 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_sparse \
  --context-length 131072 \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.83 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --disaggregation-mode decode
```

#### D node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.16.1.46`。

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "10.16.1.46:5000" \
  --nnodes 2 \
  --node-rank 1 \
  --tp-size 16 \
  --moe-dense-tp-size 1 \
  --dp-size 16 \
  --ep-size 16 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --deepep-mode low_latency \
  --page-size 64 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_sparse \
  --context-length 131072 \
  --quantization slimquant_w4a8_marlin \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.83 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --disaggregation-ib-device mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3 \
  --disaggregation-mode decode
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.16.1.36:30000 \
  --decode http://10.16.1.46:30000 \
  --policy cache_aware \
  --port 30001
```

### GLM-5-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.10

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HIP_GRAPH_USE_CMD_CACHE=0
export NCCL_SOCKET_IFNAME=ens19f0
export GLOO_SOCKET_IFNAME=ens19f0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --page-size 64 \
  --quantization slimquant_marlin \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --mem-fraction-static 0.8
```

### GLM-5-Channel-INT8-w8a8 1P2D BW1100 24x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_GRAPH_USE_CMD_CACHE=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export W8A8_SUPPORT_METHODS=3

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --tp-size 8 \
  --pp-size 1 \
  --attn-cp-size 8 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_marlin \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill
```

#### D node 0

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
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
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_TOPO_FILE_FORCE=/home/mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export ROCBLAS_TENSILE_LIBPATH=/home/library_gpu6_glm5_int8
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --nnodes 2 \
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
  --context-length 131072 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

#### D node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.16.1.46`。

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1
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
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_TOPO_FILE_FORCE=/home/mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export ROCBLAS_TENSILE_LIBPATH=/home/library_gpu6_glm5_int8
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "10.16.1.46:5000" \
  --nnodes 2 \
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
  --context-length 131072 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.16.1.36:30000 \
  --decode http://10.16.1.46:30000 \
  --policy cache_aware \
  --port 30001
```

### GLM-5-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.10

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export HIP_GRAPH_USE_CMD_CACHE=0
export NCCL_SOCKET_IFNAME=ens19f0
export GLOO_SOCKET_IFNAME=ens19f0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0

sglang serve \
  --model-path hygon/GLM-5-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --page-size 64 \
  --quantization slimquant_marlin \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --mem-fraction-static 0.8
```

### GLM-5-Channel-FP8-w8a8 1P2D BW1100 24x SGLang 0.5.10

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_USE_MODELSCOPE=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_GRAPH_USE_CMD_CACHE=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0

sglang serve \
  --model-path hygon/GLM-5-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --tp-size 8 \
  --pp-size 1 \
  --attn-cp-size 8 \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --kv-cache-dtype fp8_e4m3 \
  --dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --page-size 64 \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_marlin \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --disaggregation-mode prefill
```

#### D node 0

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_TOPO_FILE_FORCE=/home/mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export ROCBLAS_TENSILE_LIBPATH=/home/library_gpu6_glm5_int8
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\'):5000" \
  --nnodes 2 \
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
  --context-length 131072 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

#### D node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.16.1.46`。

```bash
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export MC_TOPO_FILE_FORCE=/home/mc_topo.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export ROCBLAS_TENSILE_LIBPATH=/home/library_gpu6_glm5_int8
export SGLANG_USE_MODELSCOPE=1

sglang serve \
  --model-path hygon/GLM-5-Channel-FP8-w8a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk \'/src/{print $7}\')" \
  --port 30000 \
  --dist-init-addr "10.16.1.46:5000" \
  --nnodes 2 \
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
  --context-length 131072 \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs 32 \
  --max-running-requests 512 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.16.1.36:30000 \
  --decode http://10.16.1.46:30000 \
  --policy cache_aware \
  --port 30001
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT4-w4a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/GLM-5-Channel-INT4-w4a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 默认端口为 `30000`，若与 P node 0 部署在同一机器上需指定其他端口（示例中为 `30001`）。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT4-w4a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://<router_ip>:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/GLM-5-Channel-INT4-w4a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```
