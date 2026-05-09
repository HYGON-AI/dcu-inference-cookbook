# GLM-5 on SGLang

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型         | 参数量  | 上下文  | 量化方式 | 推荐硬件                             |
| ---------- | ---- | ---- |---------| -------------------------------- |
| GLM-5-9B   | 9B   | 128K | BF16 | 1x BW1000 64GB                      |
| GLM-5-25B  | 25B  | 128K | BF16 | 1x BW1100 144GB / 2x DCU TP         |
| GLM-5-72B  | 72B  | 128K | BF16 | 2x BW1100 144GB TP / 4x BW1000 64GB TP |
| GLM-5-130B | 130B | 128K | BF16 | 4x BW1100 144GB TP                  |

## 启动命令

### GLM-5-9B

#### IFB【单卡】

```bash
sglang serve \
  --model-path THUDM/GLM-5-9B \
  --trust-remote-code \
  --tp-size 1 \
  --mem-fraction-static 0.85
```


### GLM-5-72B

#### IFB【tp4】

```bash
sglang serve \
  --model-path THUDM/GLM-5-72B \
  --trust-remote-code \
  --tp-size 4 \
  --mem-fraction-static 0.85
```


### GLM-5-W4A8

#### IFB【tp8】【bw1000】

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


#### PD 分离

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

##### P 节点【pp2tp8cp8】【bw1000】

###### node0

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
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="${HOST}"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

###### node1

说明：node1 的 `MASTER_IP` 需要填写当前分组 node0 的 `HOST`/IP，确保 `DIST_INIT_ADDR` 指向 node0。下面示例使用 `MASTER_IP="10.16.1.36"`。

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
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_GRAPH_USE_CMD_CACHE=0

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="10.16.1.36"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

##### D 节点【ep16dp16-mtp3】

###### node0

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

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="${HOST}"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

###### node1

说明：node1 的 `MASTER_IP` 需要填写当前分组 node0 的 `HOST`/IP，确保 `DIST_INIT_ADDR` 指向 node0。下面示例使用 `MASTER_IP="10.16.1.46"`。

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

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="10.16.1.46"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

##### SGLang-Router

`--prefill` 填写 P 节点 HTTP 服务地址，`--decode` 填写 D 节点 HTTP 服务地址。这里的 `30000` 是 SGLang 服务端口，不是 `--dist-init-addr` 使用的 `5000`；`--port 30005` 是 Router 对外服务端口。

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.16.1.36:30000 \
  --decode http://10.16.1.46:30000 \
  --policy cache_aware \
  --port 30005
```


### GLM-5-W8A8

#### IFB【tp8】

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


#### PD 分离

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

##### P 节点【单节点 tp8cp8】

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

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="${HOST}"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

##### D 节点【ep16dp16-mtp3】

###### node0

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

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="${HOST}"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

###### node1

说明：node1 的 `MASTER_IP` 需要填写当前分组 node0 的 `HOST`/IP，确保 `DIST_INIT_ADDR` 指向 node0。下面示例使用 `MASTER_IP="10.16.1.46"`。

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

HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
  HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
PORT="30000"
MASTER_IP="10.16.1.46"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

sglang serve \
  --model-path hygon/GLM-5-Channel-INT8-w8a8 \
  --trust-remote-code \
  --host "${HOST}" \
  --port "${PORT}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
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

##### SGLang-Router

`--prefill` 填写 P 节点 HTTP 服务地址，`--decode` 填写 D 节点 HTTP 服务地址。这里的 `30000` 是 SGLang 服务端口，不是 `--dist-init-addr` 使用的 `5000`；`--port 30005` 是 Router 对外服务端口。

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://10.16.1.36:30000 \
  --decode http://10.16.1.46:30000 \
  --policy cache_aware \
  --port 30005
```


### GLM-5-FP8

同上 GLM-5-W8A8。

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="THUDM/GLM-5-9B",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

## curl 调用

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "THUDM/GLM-5-9B",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```
