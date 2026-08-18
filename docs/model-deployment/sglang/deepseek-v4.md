# DeepSeek-V4 on SGLang

## 模型简介

DeepSeek-V4 是 DeepSeek 系列的混合专家模型。本页汇总 DeepSeek-V4 系列模型在 HCU 平台上使用 SGLang 的部署方式.

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.12 | BW1100 | 8 | IFB(CP8EP8) | [**`>_`**](#deepseek-v4-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512) |
|  | FP8 W8A8 | 0.5.12 | BW1100 | 8 | IFB(DP8EP8) | [**`>_`**](#deepseek-v4-flash-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512) |
| [hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.12 | BW1100 | 16 | IFB(CP8EP8PP2) | [**`>_`**](#deepseek-v4-pro-channel-fp8-w8a8-ifb-p-bw1100-16x-sglang-0512) |
|  | FP8 W8A8 | 0.5.12 | BW1100 | 16 | IFB(EP16DP16) | [**`>_`**](#deepseek-v4-pro-channel-fp8-w8a8-ifb-d-bw1100-16x-sglang-0512) |
|  | FP8 W8A8 | 0.5.12 | BW1100 | 32 | PD | [**`>_`**](#deepseek-v4-pro-channel-fp8-w8a8-pd-bw1100-32x-sglang-0512) |

## DeepEP 配置

以下 `ep_config.json` 仅作为参考配置。使用时请将其保存为 `ep_config.json`，并根据实际保存路径设置启动命令中的 `--deepep-config` 参数。

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
## EPLB配置参考：[EPLB](../../optimization/static-eplb-sglang.md)。

## HIPBLASLt

 [`ep16.config`](./configs/deepseek-v4/ep16.config) 部署时请下载该文件，将其放到主节点和从节点均可访问的位置，并把 `HIPBLASLT_TUNING_OVERRIDE_FILE=/XXX/ep16.config` 中的 `/XXX/ep16.config` 替换为文件的实际绝对路径。

该配置基于 **64 CU** 环境生成，其他 CU 配置不建议直接使用，可需要根据实际生成对应配置。

## topo 配置

[`topo.config`](./configs/deepseek-v4/topo.config) 是拓扑映射参考。部署时请下载该文件，将其放到主节点和从节点均可访问的位置，并将 `ROCSHMEM_TOPO_FILE_FORCE=/XXXXX/topo.config` 中的路径替换为文件的实际绝对路径。文件中的 PCI 设备地址、IB 网卡名称和映射编号仅适用于生成该配置的环境，使用前必须按照实际硬件拓扑修改。

## 启动命令

### DeepSeek-V4-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export SGLANG_HEALTH_CHECK_TIMEOUT=360
export NCCL_SOCKET_IFNAME=xxx
export GLOO_SOCKET_IFNAME=xxx
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1
export SGLANG_USE_LIGHTOP_EP_GATHER=1
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=1
export SGLANG_USE_LIGHTOP_EP_SCATTER=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export GPU_MAX_HW_QUEUES=2
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp-size 8 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --chunked-prefill-size 32768 \
  --disable-flashinfer-autotune \
  --disable-cuda-graph \
  --disable-chunked-prefix-cache \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --disable-custom-all-reduce \
  --deepep-config /xxxxx/ep_config.json \
  --mem-fraction-static 0.88 \
  --init-expert-location  /xxx/expert_distribution.pt  \ #见上述EPLB配置参考
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 64 \
  --eplb-algorithm deepseek_vec \
  --kv-cache-dtype bfloat16 \
  --host <host_ip>
```

### DeepSeek-V4-Flash-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export SGLANG_HEALTH_CHECK_TIMEOUT=180
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export UCX_NET_DEVICES=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export GPU_MAX_HW_QUEUES=3
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_NO_SCRATCH_RECLAIM=1
export HSA_SCRATCH_SINGLE_LIMIT=1073741824
export K3_USE_ASM_TAIL_REDUCE=0
export ROCSHMEM_HEAP_SIZE=3173741824
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DSV4_CHANNEL_FP8_SCALE=1
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=0
export SGLANG_OPT_SWIGLU_CLAMP_FUSION=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_OPT_CAT=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
unset SGLANG_USE_AITER
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800

sglang serve \
  --model-path hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8 \
  --trust-remote-code \
  --tp 8 \
  --dp 8 \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --moe-dense-tp-size=1 \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --chunked-prefill-size 32768 \
  --cuda-graph-max-bs 128 \
  --page-size 256 \
  --max-running-requests 256 \
  --mem-fraction-static 0.9 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --disable-flashinfer-autotune \
  --disable-chunked-prefix-cache \
  --deepep-config /xxxxx/ep_config.json \
  --kv-cache-dtype bfloat16
```

### DeepSeek-V4-Pro-Channel-FP8-w8a8 IFB  BW1100 16x SGLang 0.5.12

主节点：`NODE_RANK="${NODE_RANK:-0}"`

从节点：`NODE_RANK="${NODE_RANK:-1}"`

```bash
#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:-0}"
if [[ $# -gt 0 ]]; then
  NODE_RANK="$1"
  shift
fi

export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export ROCSHMEM_TOPO_FILE_FORCE=/xxx/topo.config #按照实际
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export ROCSHMEM_IB_GID_INDEX=0
export NCCL_SOCKET_IFNAME=xxxxx  #按照实际
export GLOO_SOCKET_IFNAME=xxxxx  #按照实际
export MODEL_PATH="${MODEL_PATH:-/hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_TORCH_PROFILER_DIR="/home/work/prof"
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export GPU_MAX_HW_QUEUES=2
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export PP="${PP:-2}"
export DP="${DP:-1}"
export TP="${TP:-8}"
export EP="${EP:-8}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-13.13.4.20:20002}" #按照实际
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-30002}" #按照实际
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:--1}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-128}"

export SERVE_LOG="./logs/sgl_rank${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
SERVE_LOG_DIR="${SERVE_LOG%/*}"
mkdir -p "$SERVE_LOG_DIR" "$SGLANG_TORCH_PROFILER_DIR"
: > "${SERVE_LOG}"
exec > >(tee -a "${SERVE_LOG}") 2>&1

echo "== Starting DeepSeek V4 Pro multi-node =="
echo "host: $(hostname)"
echo "node_rank: ${NODE_RANK}"
echo "nnodes: ${NNODES}"
echo "dist_init_addr: ${DIST_INIT_ADDR}"
echo "model: ${MODEL_PATH}"
echo "tokenizer: ${TOKENIZER_PATH}"
echo "pp: ${PP}"
echo "tp: ${TP}"
echo "ep: ${EP}"
echo "dp: ${DP}"
echo "devices: ${HIP_VISIBLE_DEVICES}"
echo "port: ${PORT}"
echo "log: ${SERVE_LOG}"

sglang serve \
  --trust-remote-code \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --pp-size "${PP}" \
  --tp-size "${TP}" \
  --ep-size "${EP}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 1800 \
  --watchdog-timeout 3600 \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --disable-flashinfer-autotune \
  --disable-radix-cache \
  --deepep-config /xxx/ep_config.json \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --max-total-tokens 1048576 \
  --kv-cache-dtype fp8_e4m3 \
  --init-expert-location /xxx/expert_distribution.pt \ #见上述EPLB配置参考 
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 8 \
  --eplb-algorithm deepseek \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

### DeepSeek-V4-Pro-Channel-FP8-w8a8 IFB D BW1100 16x SGLang 0.5.12



主节点：`NODE_RANK="${NODE_RANK:-0}"`

从节点：`NODE_RANK="${NODE_RANK:-1}"`

```bash
#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

log() {
  local level="$1"
  shift
  local message="$*"
  local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
  echo "[${timestamp}] [${level}] ${message}"
}

log_info() { log "INFO" "$@"; }
log_warn() { log "WARN" "$@"; }
log_error() { log "ERROR" "$@"; }

log_info "Initializing system environment..."
hy-smi

NODE_RANK="${1:-${NODE_RANK:-0}}"
if [[ $# -gt 0 ]]; then
  shift
fi

export HIPBLASLT_TUNING_OVERRIDE_FILE=/XXX/ep16.config
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false
export MODEL_PATH="${MODEL_PATH:-/hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_HEALTH_CHECK_TIMEOUT=1000
export HSA_ENABLE_COREDUMP=1
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3173741824
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export ROCSHMEM_TOPO_FILE_FORCE=/XXX/topo.config
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_SOCKET_IFNAME=ens61f1np1 #按照实际
export GLOO_SOCKET_IFNAME=ens61f1np1 #按照实际
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export SGLANG_USE_LIGHTOP_EP_SCATTER=false
export SGLANG_USE_LIGHTOP_EP_GATHER=false
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=false

export TP="${TP:-16}"
export PP="${PP:-1}"
export EP_SIZE="${EP_SIZE:-16}"
export DP_SIZE="${DP_SIZE:-16}"
export MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-13.13.4.20:21000}" #按照实际
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-10030}" #按照实际
export SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
export SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-2}"
export SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
export SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-2}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.89}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
export SERVE_LOG="./pro_multinode_rank${NODE_RANK}_${RUN_TS}.log"

mkdir -p "$(dirname "${SERVE_LOG}")"
: > "${SERVE_LOG}"
exec > >(tee -a "${SERVE_LOG}") 2>&1


sglang serve \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tp-size "${TP}" \
  --ep-size "${EP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  --moe-dense-tp-size "${MOE_DENSE_TP_SIZE}" \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --speculative-algo "${SPEC_ALGO}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
  --disable-flashinfer-autotune \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --skip-server-warmup \
  --moe-a2a-backend deepep \
  --deepep-mode auto \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

### DeepSeek-V4-Pro-Channel-FP8-w8a8 PD BW1100 32x SGLang 0.5.12


以下示例为 PD 部署：P 集群和 D 集群各由 2 个节点组成，每个节点使用 8 张卡，总计 32 张卡。请将节点 IP、网卡名称、DeepEP 配置和 EPLB 文件路径等替换为实际值。

同一 P 集群的主节点和从节点使用相同的 `DIST_INIT_ADDR`，均填写 `<P_node0_ip>:<P_dist_port>`；同一 D 集群的主节点和从节点也使用相同的 `DIST_INIT_ADDR`，均填写 `<D_node0_ip>:<D_dist_port>`。`<P_dist_port>` 和 `<D_dist_port>` 是分布式初始化端口，`<P_service_port>` 和 `<D_service_port>` 是服务监听端口，两类端口不要混用。Router 连接服务监听端口。

#### P node 0

说明：P 主节点使用 `NODE_RANK=0`；
```bash
#!/usr/bin/env bash
set -euo pipefail

export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1

NODE_RANK="${1:-${NODE_RANK:-0}}"
if [[ $# -gt 0 ]]; then
  shift
fi
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL=true
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export MODEL_PATH="${MODEL_PATH:-hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export GPU_MAX_HW_QUEUES=2
export SGLANG_USE_LIGHTOP_EP_SCATTER=false
export SGLANG_USE_LIGHTOP_EP_GATHER=false
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=false
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ROCSHMEM_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export NCCL_SOCKET_IFNAME=XXXX #按照实际
export GLOO_SOCKET_IFNAME=XXXX #按照实际
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9  #根据实际
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #根据实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #根据实际
export TP="${TP:-8}"
export PP="${PP:-2}"
export EP_SIZE="${EP_SIZE:-8}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-<P_node0_ip>:<P_dist_port>}" # 主从节点保持一致，均指向 P node 0
export HOST="${HOST:-<current_node_ip>}" # 按照实际修改
export PORT="${PORT:-<P_service_port>}" # 服务监听端口
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-512}"
option+=" --disaggregation-mode prefill "
option+=" --disable-cuda-graph "
# option+=" --disable-radix-cache " #按照实际
option+=" --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 "
option+=" --skip-server-warmup "

sglang serve  ${option} \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tp-size "${TP}" \
  --pp-size "${PP}" \
  --ep-size "${EP_SIZE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --disable-flashinfer-autotune \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --deepep-config /xxxxx/ep_config.json \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --init-expert-location  /xxx/expert_distribution.pt  \ #见上述EPLB配置参考
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 8 \
  --eplb-algorithm deepseek \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

#### P node 1

说明：P 从节点使用 `NODE_RANK=1`；

```bash
#!/usr/bin/env bash
set -euo pipefail

export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1

NODE_RANK="${1:-${NODE_RANK:-1}}"
if [[ $# -gt 0 ]]; then
  shift
fi
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL=true
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export MODEL_PATH="${MODEL_PATH:-hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export GPU_MAX_HW_QUEUES=2
export SGLANG_USE_LIGHTOP_EP_SCATTER=false
export SGLANG_USE_LIGHTOP_EP_GATHER=false
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=false
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export ROCSHMEM_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export NCCL_SOCKET_IFNAME=XXXX #按照实际
export GLOO_SOCKET_IFNAME=XXXX #按照实际
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9  #根据实际
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #根据实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #根据实际
export TP="${TP:-8}"
export PP="${PP:-2}"
export EP_SIZE="${EP_SIZE:-8}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-<P_node0_ip>:<P_dist_port>}" # 主从节点保持一致，均指向 P node 0
export HOST="${HOST:-<current_node_ip>}" # 按照实际修改
export PORT="${PORT:-<P_service_port>}" # 服务监听端口
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
export MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-512}"
option+=" --disaggregation-mode prefill "
option+=" --disable-cuda-graph "
# option+=" --disable-radix-cache " #按照实际
option+=" --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 "
option+=" --skip-server-warmup "

sglang serve  ${option} \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tp-size "${TP}" \
  --pp-size "${PP}" \
  --ep-size "${EP_SIZE}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --disable-flashinfer-autotune \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --deepep-config /xxxxx/ep_config.json \
  --moe-a2a-backend deepep \
  --deepep-mode normal \
  --init-expert-location  /xxx/expert_distribution.pt  \ #见上述EPLB配置参考
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 8 \
  --eplb-algorithm deepseek \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

#### D node 0

说明：D 主节点使用 `NODE_RANK=0`；

```bash
#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false

NODE_RANK="${1:-${NODE_RANK:-0}}"
if [[ $# -gt 0 ]]; then
  shift
fi
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL=true
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export HIPBLASLT_TUNING_OVERRIDE_FILE=/xxx/ep16.config
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export MODEL_PATH="${MODEL_PATH:-hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export HSA_ENABLE_COREDUMP=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3173741824
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_SOCKET_IFNAME=XXXXX #按照实际
export GLOO_SOCKET_IFNAME=XXXXX #按照实际
export ROCSHMEM_TOPO_FILE_FORCE=/XXXXX/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export SGLANG_USE_LIGHTOP_EP_SCATTER=false
export SGLANG_USE_LIGHTOP_EP_GATHER=false
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=false
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export TP="${TP:-16}"
export PP="${PP:-1}"
export EP_SIZE="${EP_SIZE:-16}"
export DP_SIZE="${DP_SIZE:-16}"
export MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-<D_node0_ip>:<D_dist_port>}" # 主从节点保持一致，均指向 D node 0
export HOST="${HOST:-<current_node_ip>}" # 按照实际修改
export PORT="${PORT:-<D_service_port>}" # 服务监听端口
export SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
export SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
export SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
export SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.91}"
sglang serve \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tp-size "${TP}" \
  --ep-size "${EP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  --moe-dense-tp-size "${MOE_DENSE_TP_SIZE}" \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --speculative-algo "${SPEC_ALGO}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
  --disable-flashinfer-autotune \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --context-length 1048576 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --skip-server-warmup \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

#### D node 1

说明：D 从节点使用 `NODE_RANK=1`；

```bash
#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export SGLANG_LIGHTOP_KVALLOC_KERNEL=1
export HIP_GRAPH_ACCUMULATE_DISPATCH=0
export SGLANG_LIGHTOP_TOPK=true
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=false

NODE_RANK="${1:-${NODE_RANK:-1}}"
if [[ $# -gt 0 ]]; then
  shift
fi
export SGLANG_DSV4_REQUEST_SCOPED_C128_STATE=true
export SGLANG_OPT_USE_ONLINE_COMPRESS=false
export SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL=true
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1800
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800
export HIPBLASLT_TUNING_OVERRIDE_FILE=/xxx/ep16.config
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH=1
export MODEL_PATH="${MODEL_PATH:-hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export SGLANG_HEALTH_CHECK_TIMEOUT=10000
export HSA_ENABLE_COREDUMP=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export ROCSHMEM_MAX_NUM_CONTEXTS=60
export ROCSHMEM_HEAP_SIZE=3173741824
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export MC_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_SOCKET_IFNAME=XXXXX #按照实际
export GLOO_SOCKET_IFNAME=XXXXX #按照实际
export ROCSHMEM_TOPO_FILE_FORCE=/XXXXX/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 #按照实际
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1 #按照实际
export ROCSHMEM_IB_GID_INDEX=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export HIP_KERNEL_BATCH_CEILING=100
export GPU_FORCE_BLIT_COPY_SIZE=16
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE=1
export SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_OPT_USE_FUSED_STORE_CACHE="${SGLANG_OPT_USE_FUSED_STORE_CACHE:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-true}"
export SGLANG_OPT_SWIGLU_CLAMP_FUSION="${SGLANG_OPT_SWIGLU_CLAMP_FUSION:-false}"
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA="${SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA:-0}"
export SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA="${SGLANG_DSV4_SPLIT_HCA_NONSPARSE_MLA:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL="${SGLANG_DSV4_SPARSE_PREFILL_SINGLE_CALL:-false}"
export SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER="${SGLANG_DSV4_SPARSE_PREFILL_TRITON_GATHER:-false}"
export SGLANG_JIT_DEEPGEMM_PRECOMPILE="${SGLANG_JIT_DEEPGEMM_PRECOMPILE:-0}"
export SGLANG_DISABLED_MODEL_ARCHS="${SGLANG_DISABLED_MODEL_ARCHS:-midashenglm}"
export SGLANG_DEBUG_DSV4_LOAD="${SGLANG_DEBUG_DSV4_LOAD:-0}"
export SGLANG_APPLY_CONFIG_BACKUP="${SGLANG_APPLY_CONFIG_BACKUP:-none}"
export SGLANG_USE_LIGHTOP_EP_SCATTER=false
export SGLANG_USE_LIGHTOP_EP_GATHER=false
export SGLANG_USE_LIGHTOP_EP_MOE_ALIGN=false
export SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0
export TP="${TP:-16}"
export PP="${PP:-1}"
export EP_SIZE="${EP_SIZE:-16}"
export DP_SIZE="${DP_SIZE:-16}"
export MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"
export NNODES="${NNODES:-2}"
export DIST_INIT_ADDR="${DIST_INIT_ADDR:-<D_node0_ip>:<D_dist_port>}" # 主从节点保持一致，均指向 D node 0
export HOST="${HOST:-<current_node_ip>}" # 按照实际修改
export PORT="${PORT:-<D_service_port>}" # 服务监听端口
export SPEC_ALGO="${SPEC_ALGO:-EAGLE}"
export SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-3}"
export SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
export SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-4}"
export CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.91}"
sglang serve \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tp-size "${TP}" \
  --ep-size "${EP_SIZE}" \
  --dp-size "${DP_SIZE}" \
  --moe-dense-tp-size "${MOE_DENSE_TP_SIZE}" \
  --enable-dp-attention \
  --enable-dp-lm-head \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --trust-remote-code \
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
  --speculative-algo "${SPEC_ALGO}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
  --disable-flashinfer-autotune \
  --cuda-graph-max-bs 16 \
  --max-running-requests 256 \
  --context-length 1048576 \
  --moe-a2a-backend deepep \
  --deepep-mode low_latency \
  --disaggregation-mode decode \
  --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
  --skip-server-warmup \
  --host "${HOST}" \
  --port "${PORT}" \
  "$@"
```

#### Router

Router 只需填写 P node 0 和 D node 0（即 P/D rank 0）的服务地址，多节点中的其他节点无需填写。

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "http://<P_node0_ip>:<P_service_port>" \
  --decode "http://<D_node0_ip>:<D_service_port>" \
  --policy cache_aware \
  --port 30001
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好，请介绍一下你自己。"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/DeepSeek-V4-Flash-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。示例中 Router 端口为 `30001`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8",
    messages=[
        {"role": "user", "content": "你好"},
    ],
    max_tokens=2048,
)

print(response.choices[0].message.content)
```

```bash
curl "http://<router_ip>:30001/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/DeepSeek-V4-Pro-Channel-FP8-w8a8", "messages": [{"role": "user", "content": "你好"}], "max_tokens": 128}'
```
