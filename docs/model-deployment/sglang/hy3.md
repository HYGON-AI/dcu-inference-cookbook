# Hy3

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Hy3-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB (tp8dp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512-tp8dp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB(tp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1000-8x-sglang-0512-tp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB(tp8dp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1000-8x-sglang-0512-tp8dp8) |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#hy3-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|                                                                                      | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 24 | 1P2D | [**`>_`**](#hy3-fp8-1p2d-bw1100-24x-sglang-0512) |

## 启动命令

### Hy3-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_RMS_ROTARY=0
export W8A8_SUPPORT_METHODS=2
# export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH #按照实际需求

sglang serve \
  --model hygon/Hy3-Channel-INT8-w8a8 \
  --tp-size 8 \
  --trust-remote-code \
  --enable-cache-report \
  --mem-fraction-static 0.80 \
  --port 8000 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8dp8)

~~~bash
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ROCM_USE_AITER_MOE=1
unset SGLANG_USE_FP8_W8A8_MOE
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_MAX_NUM_CONTEXTS=32
#export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH 按照实际

sglang serve \
  --model-path hygon/Hy3-Channel-INT8-w8a8 \
  --dp-size 8 \
  --tp-size 8 \
  --deepep-mode auto \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size=1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --kv-cache-dtype fp8_e4m3 \
  --watchdog-timeout 36000 \
  --cuda-graph-max-bs 16 \
  --enforce-shared-experts-fusion \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --quantization slimquant_marlin
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.12 (tp8)

~~~bash
export SGLANG_ROCM_USE_AITER_MOE=false
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FUSED_RMS_ROTARY=0
export W8A8_SUPPORT_METHODS=2
# export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH 按照实际

sglang serve \
  --model hygon/Hy3-Channel-INT8-w8a8 \
  --tp-size 8 \
  --trust-remote-code \
  --enable-cache-report \
  --mem-fraction-static 0.80 \
  --port 8000 \
  --attention-backend fa3 \
  --page-size 64 \
  --kv-cache-dtype fp8_e5m2 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.12 (tp8dp8)

~~~bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_INT8_DEEPGEMM_ASM=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FUSED_RMS_ROTARY=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_MAX_NUM_CONTEXTS=32
#export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:$LD_LIBRARY_PATH 按照实际

sglang serve \
  --model-path hygon/Hy3-Channel-INT8-w8a8 \
  --dp-size 8 \
  --tp-size 8 \
  --enable-dp-attention \
  --deepep-mode auto \
  --moe-a2a-backend deepep \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --kv-cache-dtype fp8_e5m2 \
  --watchdog-timeout 36000 \
  --cuda-graph-max-bs 16 \
  --enforce-shared-experts-fusion \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan \
  --quantization slimquant_marlin
~~~

### Hy3-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1

sglang serve \
  --model-path hygon/Hy3-Channel-FP8-w8a8 \
  --tp-size 8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --port 3009 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disable-radix-cache \
  --kv-cache-dtype fp8_e4m3 \
  --tool-call-parser hunyuan \
  --reasoning-parser hunyuan
~~~

### Hy3-FP8 1P2D BW1100 24x SGLang 0.5.12

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。部署包含一个 8 卡 P 节点和两个 8 卡 D 节点，共使用 24 张 HCU。P 节点使用 PP8，两个 D 节点共同组成一个 EP16 服务。

离线 EPLB 优化参考：[Static EPLB 原理与使用最佳实践](../../optimization/static-eplb-sglang.md)。

#### P node

将 `P_NODE_IP` 设置为 P 节点的实际 IP。

~~~bash
P_NODE_IP="10.x.x.x"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export SGLANG_USE_FUSED_RMS_ROTARY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export MC_GID_INDEX=0

sysctl -w kernel.numa_balancing=0

sglang serve \
  --model-path hygon/Hy3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --tp-size 1 \
  --dp-size 1 \
  --pp-size 8 \
  --mem-fraction-static 0.80 \
  --attention-backend fa3 \
  --max-running-requests 100 \
  --context-length 16384 \
  --skip-server-warmup \
  --disable-radix-cache \
  --chunked-prefill-size 16384 \
  --pp-async-batch-depth 1 \
  --load-balance-method round_robin \
  --disaggregation-mode prefill \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --disaggregation-bootstrap-port 8998 \
  --host "${P_NODE_IP}" \
  --port 30000 \
  --dist-init-addr "${P_NODE_IP}:5000" \
  --nnodes 1 \
  --node-rank 0
~~~

#### D node 0

将 `D_NODE0_IP` 设置为 D node 0 的实际 IP。

~~~bash
D_NODE0_IP="10.x.x.x"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_HOST_IP="${D_NODE0_IP}"
export NCCL_SOCKET_IFNAME=enp10s0f4u1
export GLOO_SOCKET_IFNAME=enp10s0f4u1
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export SGLANG_USE_FUSED_RMS_ROTARY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_MAX_NUM_CONTEXTS=32
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1

sysctl -w kernel.numa_balancing=0

sglang serve \
  --model-path hygon/Hy3-Channel-FP8-w8a8 \
  --dp-size 16 \
  --tp-size 16 \
  --ep-size 16 \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dist-init-addr "${D_NODE0_IP}:5000" \
  --host "${D_NODE0_IP}" \
  --nnodes 2 \
  --node-rank 0 \
  --dtype bfloat16 \
  --port 30000 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.80 \
  --context-length 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --enforce-shared-experts-fusion
~~~

#### D node 1

将 `D_NODE0_IP` 设置为 D node 0 的实际 IP，将 `D_NODE1_IP` 设置为 D node 1 的实际 IP。

~~~bash
D_NODE0_IP="10.x.x.x"
D_NODE1_IP="10.x.x.x"

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_HOST_IP="${D_NODE1_IP}"
export NCCL_SOCKET_IFNAME=enp10s0f4u1
export GLOO_SOCKET_IFNAME=enp10s0f4u1
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export SGLANG_USE_FUSED_RMS_ROTARY=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export GPU_MAX_HW_QUEUES=3
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export ROCSHMEM_HEAP_SIZE=3737418240
export ROCSHMEM_MAX_NUM_CONTEXTS=32
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1

sysctl -w kernel.numa_balancing=0

sglang serve \
  --model-path hygon/Hy3-Channel-FP8-w8a8 \
  --dp-size 16 \
  --tp-size 16 \
  --ep-size 16 \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dist-init-addr "${D_NODE0_IP}:5000" \
  --host "${D_NODE1_IP}" \
  --nnodes 2 \
  --node-rank 1 \
  --dtype bfloat16 \
  --port 30000 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.80 \
  --context-length 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  --enforce-shared-experts-fusion
~~~

#### Router

将 `P_NODE_IP` 和 `D_NODE0_IP` 分别设置为 P 节点和 D node 0 的实际 IP。

~~~bash
P_NODE_IP="10.x.x.x"
D_NODE0_IP="10.x.x.x"

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://${P_NODE_IP}:30000" 8998 \
  --decode "http://${D_NODE0_IP}:30000" \
  --policy round_robin \
  --host 0.0.0.0 \
  --port 30001
~~~

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-INT8-w8a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy3-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 Router，而不是直接发送到 P/D 节点。以下示例中 Router 使用端口 `30001`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache。"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://<router_ip>:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/Hy3-Channel-FP8-w8a8",
    "messages": [
      {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "max_tokens": 128
  }'
```
