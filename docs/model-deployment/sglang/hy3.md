# hy3

# Hy3

## 模型列表

|模型权重|量化方式|SGLang 版本|推荐硬件|卡数|部署方式|启动命令|
|---|---|---|---|---|---|---|
|[hygon/Hy3\-Channel\-INT8\-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-INT8-w8a8)|INT8 W8A8|0\.5\.12|BW1100|8|IFB\(tp8\)|**`>_`**|
||INT8 W8A8|0\.5\.12|BW1100|8|IFB \(tp8dp8\)|**`>_`**|
||INT8 W8A8|0\.5\.12|BW1000|8|IFB\(tp8\)|**`>_`**|
||INT8 W8A8|0\.5\.12|BW1000|8|IFB\(tp8dp8\)|**`>_`**|
||INT8 W8A8|0\.5\.12|BW1000|8|PD分离\(tp8\+tp8\)|**`>_`**|
||INT8 W8A8|0\.5\.12|BW1000|8|PD分离\(pp8\+ep8\)|**`>_`**|
|[hygon/Hy3\-Channel\-FP8\-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8)|FP8 W8A8|0\.5\.12|BW1100|8|IFB|**`>_`**|

## 启动命令

### Hy3\-Channel\-INT8\-w8a8 IFB BW1100 8x SGLang 0\.5\.12 \(tp8\)

```Bash
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
```

### Hy3\-Channel\-INT8\-w8a8 IFB BW1100 8x SGLang 0\.5\.12 \(tp8dp8\)

```Bash
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
```

### Hy3\-Channel\-INT8\-w8a8 IFB BW1000 8x SGLang 0\.5\.12 \(tp8\)

```Bash
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
```

### Hy3\-Channel\-INT8\-w8a8 IFB BW1000 8x SGLang 0\.5\.12 \(tp8dp8\)

```Bash
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
```

### Hy3\-Channel\-INT8\-w8a8 PD分离 BW1000 8x SGLang 0\.5\.12 \(tp8\+tp8\)

#### P节点

```Bash
#!/usr/bin/env bash
set -euo pipefail
export MC_TCP_ENABLE_CONNECTION_POOL=1
# Kunshan Hunyuan V3 PD prefill node, 8 GPUs, PP=8.
# Run on the prefill node:
#   bash hunyuan_v3_pd_single_p.sh
#export PYTHONPATH=/workspace/hyv3/sgl_0714/sglang-das/python:$PYTHONPATH
WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

#PREFILL_HOST="${PREFILL_HOST:-10.101.0.55}"
PREFILL_HOST="${PREFILL_HOST:-10.211.9.35}"
PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-${PREFILL_HOST}}"
#PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-5000}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8998}"

MODEL_PATH="${MODEL_PATH:-/module1/Hy3-preview-W8A8}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${PREFILL_HOST}}"

# 消除空泡
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
# export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp10s0f4u1}"
# export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp10s0f4u1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
# export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
# rocblaslt config for tuning
#export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/hyv3/gemm_config/hipblaslt.config
#export ROCBLAS_TENSILE_LIBPATH=/workspace/hyv3/gemm_config/rocblas_hy3_fp8_zmy_sglang

export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"

export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export HIP_KERNEL_EVENT_SYSTENFENCE="${HIP_KERNEL_EVENT_SYSTENFENCE:-1}"
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-16}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
# sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

#export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/prof}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export VLLM_USE_LIGHTOP_MOE_ALIGN="${VLLM_USE_LIGHTOP_MOE_ALIGN:-1}"
export LMSLIM_USE_LIGHTOP="${LMSLIM_USE_LIGHTOP:-1}"
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:-0}"

export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"

export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export W8A8_SUPPORT_METHODS=1
export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
# export SGLANG_USE_FP8_W8A8_MOE=1 # 开启 lightop的 Marlin MOE W8A8 算子
# 不要开启 FP8 W8A8 MoE dispatch
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
fi

mkdir -p "${LOG_DIR}"

# python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --trust-remote-code \
#   --page-size "${PAGE_SIZE:-64}" \
#   --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
#   --tp-size "${PREFILL_TP_SIZE:-1}" \
#   --dp-size "${PREFILL_DP_SIZE:-1}" \
#   --pp-size "${PREFILL_PP_SIZE:-8}" \
#   --enable-cache-report \
#   --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC:-0.80}" \
#   --attention-backend "${ATTENTION_BACKEND:-fa3}" \
#   --context-length "${PREFILL_CONTEXT_LENGTH:-8192}" \
#   --chunked-prefill-size 4096 \
#   --disable-radix-cache \
#   --max-running-requests 128 \
#   --disaggregation-mode prefill \
#   --disaggregation-ib-device "${IB_DEVICES}" \
#   --disaggregation-bootstrap-port "${PREFILL_BOOTSTRAP_PORT}" \
#   --host "${PREFILL_BIND_HOST}" \
#   --port "${PREFILL_PORT}" \
#   --dist-init-addr "${PREFILL_HOST}:${PREFILL_DIST_PORT}" \
#   "$@" \
#   2>&1 | tee "${LOG_DIR}/hunyuan_v3_p_${PREFILL_HOST}_${PREFILL_PORT}.log"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size 8 \
  --nnodes 1 \
  --node-rank 0 \
  --page-size 64 \
  --skip-server-warmup \
  --cuda-graph-max-bs 64 \
  --kv-cache-dtype fp8_e5m2 \
  --mem-fraction-static 0.80 \
  --attention-backend fa3 \
  --context-length 16384 \
  --chunked-prefill-size 16384 \
  --max-running-requests 128 \
  --pp-async-batch-depth 1 \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-bootstrap-port "${PREFILL_BOOTSTRAP_PORT}" \
  --host "${PREFILL_BIND_HOST}" \
  --port "${PREFILL_PORT}" \
  --log-requests \
  --log-level debug \
  --show-time-cost \
  --enable-metrics \
  --decode-log-interval 1 \
  "$@" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_p_${PREFILL_HOST}_${PREFILL_PORT}.log"
```

#### D节点

```Bash
#!/usr/bin/env bash
set -euo pipefail
export MC_TCP_ENABLE_CONNECTION_POOL=1
# Kunshan Hunyuan V3 PD decode node, 8 GPUs, EP=8.
# Run on the decode node:
#   bash hunyuan_v3_pd_single_d.sh
#
# EP16: run on both decode nodes as one 16-GPU service.
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 NODE_RANK=0 bash hunyuan_v3_pd_single_d.sh
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 NODE_RANK=1 bash hunyuan_v3_pd_single_d.sh
#export PYTHONPATH=/workspace/hyv3/sgl_0714/sglang-das/python:$PYTHONPATH
WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

DECODE_HOSTS="${DECODE_HOSTS:-10.211.9.36}"
read -r -a DECODE_HOST_ARRAY <<< "${DECODE_HOSTS}"

EP_SIZE="${EP_SIZE:-8}"
if [[ "${EP_SIZE}" == "16" ]]; then
  if [[ "${#DECODE_HOST_ARRAY[@]}" -lt 2 ]]; then
    echo "EP_SIZE=16 requires DECODE_HOSTS to contain two decode node IPs." >&2
    exit 1
  fi
  DEFAULT_NNODES=2
  DEFAULT_TP_SIZE=16
  DEFAULT_DP_SIZE=16
  DEFAULT_NODE_RANK="${NODE_RANK:-0}"
else
  DEFAULT_NNODES=1
  DEFAULT_TP_SIZE=8
  DEFAULT_DP_SIZE=8
  DEFAULT_NODE_RANK=0
fi

NODE_RANK="${NODE_RANK:-${DEFAULT_NODE_RANK}}"

DECODE_MASTER_HOST="${DECODE_MASTER_HOST:-${DECODE_HOST_ARRAY[0]}}"
if [[ -z "${DECODE_HOST:-}" ]]; then
  if command -v hostname >/dev/null 2>&1; then
    LOCAL_IPS="$(hostname -I 2>/dev/null || true)"
  else
    LOCAL_IPS=""
  fi
  for host in "${DECODE_HOST_ARRAY[@]}"; do
    if [[ " ${LOCAL_IPS} " == *" ${host} "* ]]; then
      DECODE_HOST="${host}"
      break
    fi
  done
fi
DECODE_HOST="${DECODE_HOST:-${DECODE_HOST_ARRAY[${NODE_RANK}]:-${DECODE_MASTER_HOST}}}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
#DECODE_BIND_HOST="${DECODE_BIND_HOST:-0.0.0.0}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_DIST_PORT="${DECODE_DIST_PORT:-5000}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DECODE_MASTER_HOST}:${DECODE_DIST_PORT}}"

MODEL_PATH="${MODEL_PATH:-/module1/hy3-int8-w8a8}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"


# export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
# export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
# export NCCL_NET_PLUGIN=shca
#export LD_LIBRARY_PATH=/zkjh/drivers/ib_plugin/topo_lib/lib:$LD_LIBRARY_PATH



export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"

export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
# export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0

# export SGLANG_USE_TRITON_VLLM_FA=1
# export SGLANG_DISAGG_KV_DEBUG=1
##blas
#export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/hyv3/gemm_config/hipblaslt.config
#export ROCBLAS_TENSILE_LIBPATH=/workspace/hyv3/gemm_config/rocblas_hy3_fp8_zmy_sglang
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"
export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export HIP_KERNEL_EVENT_SYSTENFENCE="${HIP_KERNEL_EVENT_SYSTENFENCE:-1}"
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-16}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-3737418240}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-32}"
export ROCSHMEM_GDR_DISABLE_XDP="${ROCSHMEM_GDR_DISABLE_XDP:-1}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-${IB_DEVICES}}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"

if [[ -n "${ROCSHMEM_TOPO_FILE_FORCE:-}" ]]; then
  export ROCSHMEM_TOPO_FILE_FORCE
elif [[ -f /workspace/hyv3/s129_topo.config ]]; then
  export ROCSHMEM_TOPO_FILE_FORCE=/workspace/hyv3/s129_topo.config
fi

export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/prof}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_USE_DEEPGEMM_MOE="${SGLANG_USE_DEEPGEMM_MOE:-1}"

export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export W8A8_SUPPORT_METHODS=1

export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
# export SGLANG_USE_FP8_W8A8_MOE=1 # 开启 lightop的 Marlin MOE W8A8 算子
# 不要开启 FP8 W8A8 MoE dispatch
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
fi

NNODES="${NNODES:-${DEFAULT_NNODES}}"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
DP_SIZE="${DP_SIZE:-${DEFAULT_DP_SIZE}}"
MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"

SPEC_ALGO="${SPEC_ALGO:-NEXTN}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-2}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-3}"
mkdir -p "${LOG_DIR}"

  # --speculative-algorithm "${SPEC_ALGO}" \
  # --speculative-num-steps "${SPEC_NUM_STEPS}" \
  # --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
  # --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
#export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/hyv3/eplb_trace
# python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --tp-size "${PREFILL_TP_SIZE:-8}" \
#   --pp-size "${PREFILL_PP_SIZE:-8}" \
#   --trust-remote-code \
#   --enable-cache-report \
#   --dist-init-addr "${DIST_INIT_ADDR}" \
#   --host "${DECODE_BIND_HOST}" \
#   --port "${DECODE_PORT}" \
#   --attention-backend "${ATTENTION_BACKEND:-fa3}" \
#   --page-size "${PAGE_SIZE:-64}" \
#   --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC:-0.80}" \
#   --context-length "${DECODE_CONTEXT_LENGTH:-8192}" \
#   --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
#   --max-running-requests 128 \
#   --chunked-prefill-size 4096 \
#   --disable-radix-cache \
#   --disaggregation-mode decode \
#   --disaggregation-ib-device "${IB_DEVICES}" \
#   "$@" \
#   2>&1 | tee "${LOG_DIR}/hunyuan_v3_d_${DECODE_HOST}_${DECODE_PORT}-1.log"


python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size 8 \
  --nnodes 1 \
  --node-rank 0 \
  --trust-remote-code \
  --enable-cache-report \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.80 \
  --context-length 8192 \
  --kv-cache-dtype fp8_e5m2 \
  --max-running-requests 320 \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --disable-radix-cache \
  --disaggregation-mode decode \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --watchdog-timeout 36000 \
  --log-requests \
  --log-level debug \
  --show-time-cost \
  --enable-metrics \
  --decode-log-interval 1 \
  "$@" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_d_${DECODE_HOST}_${DECODE_PORT}.log"
```

#### Router

```Bash
#!/usr/bin/env bash
set -euo pipefail

# Kunshan Hunyuan V3 PD router.
# EP8 default routes to the single decode service on 10.101.0.54.
# EP16 routes to the rank0 decode service only, for example:
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 bash hunyuan_v3_pd_single_router.sh
# or:
#   DECODE_URLS="http://10.101.0.54:30001" bash hunyuan_v3_pd_single_router.sh

WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

PREFILL_HOST="${PREFILL_HOST:-10.211.9.35}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8998}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_HOSTS="${DECODE_HOSTS:-10.211.9.36}"
EP_SIZE="${EP_SIZE:-8}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ -z "${DECODE_URLS:-}" ]]; then
  DECODE_URLS=""
  if [[ "${EP_SIZE}" == "16" ]]; then
    read -r decode_master_host _ <<< "${DECODE_HOSTS}"
    DECODE_URLS="http://${decode_master_host}:${DECODE_PORT}"
  else
    for host in ${DECODE_HOSTS}; do
      DECODE_URLS="${DECODE_URLS} http://${host}:${DECODE_PORT}"
    done
  fi
fi

mkdir -p "${LOG_DIR}"

ROUTER_ARGS=(
  --pd-disaggregation
  --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "${PREFILL_BOOTSTRAP_PORT}"
  --policy "${ROUTER_POLICY:-round_robin}"
  --host "${ROUTER_HOST}"
  --port "${ROUTER_PORT}"
)

for decode_url in ${DECODE_URLS}; do
  ROUTER_ARGS+=(--decode "${decode_url}")
done

python -m sglang_router.launch_router "${ROUTER_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_router_${ROUTER_PORT}.log"

```

### Hy3\-Channel\-INT8\-w8a8 PD分离 BW1000 8x SGLang 0\.5\.12 \(pp8\+ep8\)

#### P节点

```Bash
#!/usr/bin/env bash
set -euo pipefail
export MC_TCP_ENABLE_CONNECTION_POOL=1
# Kunshan Hunyuan V3 PD prefill node, 8 GPUs, PP=8.
# Run on the prefill node:
#   bash hunyuan_v3_pd_single_p.sh
#export PYTHONPATH=/workspace/hyv3/sgl_0714/sglang-das/python:$PYTHONPATH
WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

#PREFILL_HOST="${PREFILL_HOST:-10.101.0.55}"
PREFILL_HOST="${PREFILL_HOST:-10.211.9.35}"
PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-${PREFILL_HOST}}"
#PREFILL_BIND_HOST="${PREFILL_BIND_HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_DIST_PORT="${PREFILL_DIST_PORT:-5000}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8998}"

MODEL_PATH="${MODEL_PATH:-/module/Hy3-preview-W8A8}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${PREFILL_HOST}}"

# 消除空泡
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
# export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp10s0f4u1}"
# export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp10s0f4u1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
# export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
# rocblaslt config for tuning
#export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/hyv3/gemm_config/hipblaslt.config
#export ROCBLAS_TENSILE_LIBPATH=/workspace/hyv3/gemm_config/rocblas_hy3_fp8_zmy_sglang

export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"

export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export HIP_KERNEL_EVENT_SYSTENFENCE="${HIP_KERNEL_EVENT_SYSTENFENCE:-1}"
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-16}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
# sysctl -w kernel.numa_balancing=0
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

#export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/prof}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export VLLM_USE_LIGHTOP_MOE_ALIGN="${VLLM_USE_LIGHTOP_MOE_ALIGN:-1}"
export LMSLIM_USE_LIGHTOP="${LMSLIM_USE_LIGHTOP:-1}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO="${SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO:-1}"
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS="${SGLANG_ASSIGN_EXTEND_CACHE_LOCS:-1}"
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL="${SGLANG_ASSIGN_REQ_TO_TOKEN_POOL:-1}"
export SGLANG_GET_LAST_LOC="${SGLANG_GET_LAST_LOC:-1}"
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON="${SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON:-1}"
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES="${SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES:-1}"
export MC_GID_INDEX="${MC_GID_INDEX:-0}"

export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"

export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export W8A8_SUPPORT_METHODS="${W8A8_SUPPORT_METHODS:-2}"

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
fi

mkdir -p "${LOG_DIR}"

# python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --trust-remote-code \
#   --page-size "${PAGE_SIZE:-64}" \
#   --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
#   --tp-size "${PREFILL_TP_SIZE:-1}" \
#   --dp-size "${PREFILL_DP_SIZE:-1}" \
#   --pp-size "${PREFILL_PP_SIZE:-8}" \
#   --enable-cache-report \
#   --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC:-0.80}" \
#   --attention-backend "${ATTENTION_BACKEND:-fa3}" \
#   --context-length "${PREFILL_CONTEXT_LENGTH:-8192}" \
#   --chunked-prefill-size 4096 \
#   --disable-radix-cache \
#   --max-running-requests 128 \
#   --disaggregation-mode prefill \
#   --disaggregation-ib-device "${IB_DEVICES}" \
#   --disaggregation-bootstrap-port "${PREFILL_BOOTSTRAP_PORT}" \
#   --host "${PREFILL_BIND_HOST}" \
#   --port "${PREFILL_PORT}" \
#   --dist-init-addr "${PREFILL_HOST}:${PREFILL_DIST_PORT}" \
#   "$@" \
#   2>&1 | tee "${LOG_DIR}/hunyuan_v3_p_${PREFILL_HOST}_${PREFILL_PORT}.log"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --tp-size "${PREFILL_TP_SIZE:-1}" \
  --pp-size "${PREFILL_PP_SIZE:-8}" \
  --dp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --page-size "${PAGE_SIZE:-64}" \
  --skip-server-warmup \
  --cuda-graph-max-bs 64 \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
  --mem-fraction-static "${PREFILL_MEM_FRACTION_STATIC:-0.85}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  --context-length 16384 \
  --chunked-prefill-size 16384 \
  --max-running-requests "${PREFILL_MAX_RUNNING_REQUESTS:-128}" \
  --pp-async-batch-depth 1 \
  --disable-radix-cache \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend "${DISAGG_TRANSFER_BACKEND:-mooncake}" \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-bootstrap-port "${PREFILL_BOOTSTRAP_PORT}" \
  --host "${PREFILL_BIND_HOST}" \
  --port "${PREFILL_PORT}" \
  --log-requests \
  --log-level debug \
  --show-time-cost \
  --enable-metrics \
  --decode-log-interval 1 \
  "$@" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_p_${PREFILL_HOST}_${PREFILL_PORT}.log"
```

#### D节点

```Bash
#!/usr/bin/env bash
set -euo pipefail
export MC_TCP_ENABLE_CONNECTION_POOL=1
# Kunshan Hunyuan V3 PD decode node, 8 GPUs, EP=8.
# Run on the decode node:
#   bash hunyuan_v3_pd_single_d.sh
#
# EP16: run on both decode nodes as one 16-GPU service.
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 NODE_RANK=0 bash hunyuan_v3_pd_single_d.sh
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 NODE_RANK=1 bash hunyuan_v3_pd_single_d.sh
#export PYTHONPATH=/workspace/hyv3/sgl_0714/sglang-das/python:$PYTHONPATH
WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

DECODE_HOSTS="${DECODE_HOSTS:-10.211.9.36}"
read -r -a DECODE_HOST_ARRAY <<< "${DECODE_HOSTS}"

EP_SIZE="${EP_SIZE:-8}"
if [[ "${EP_SIZE}" == "16" ]]; then
  if [[ "${#DECODE_HOST_ARRAY[@]}" -lt 2 ]]; then
    echo "EP_SIZE=16 requires DECODE_HOSTS to contain two decode node IPs." >&2
    exit 1
  fi
  DEFAULT_NNODES=2
  DEFAULT_TP_SIZE=16
  DEFAULT_DP_SIZE=16
  DEFAULT_NODE_RANK="${NODE_RANK:-0}"
else
  DEFAULT_NNODES=1
  DEFAULT_TP_SIZE=8
  DEFAULT_DP_SIZE=8
  DEFAULT_NODE_RANK=0
fi

NODE_RANK="${NODE_RANK:-${DEFAULT_NODE_RANK}}"

DECODE_MASTER_HOST="${DECODE_MASTER_HOST:-${DECODE_HOST_ARRAY[0]}}"
if [[ -z "${DECODE_HOST:-}" ]]; then
  if command -v hostname >/dev/null 2>&1; then
    LOCAL_IPS="$(hostname -I 2>/dev/null || true)"
  else
    LOCAL_IPS=""
  fi
  for host in "${DECODE_HOST_ARRAY[@]}"; do
    if [[ " ${LOCAL_IPS} " == *" ${host} "* ]]; then
      DECODE_HOST="${host}"
      break
    fi
  done
fi
DECODE_HOST="${DECODE_HOST:-${DECODE_HOST_ARRAY[${NODE_RANK}]:-${DECODE_MASTER_HOST}}}"
DECODE_BIND_HOST="${DECODE_BIND_HOST:-${DECODE_HOST}}"
#DECODE_BIND_HOST="${DECODE_BIND_HOST:-0.0.0.0}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_DIST_PORT="${DECODE_DIST_PORT:-5000}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-${DECODE_MASTER_HOST}:${DECODE_DIST_PORT}}"

MODEL_PATH="${MODEL_PATH:-/module/Hy3-preview-W8A8}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
IB_DEVICES="${IB_DEVICES:-shca_0,shca_1,shca_2,shca_3}"

# export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
# export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
# export NCCL_NET_PLUGIN=shca
#export LD_LIBRARY_PATH=/zkjh/drivers/ib_plugin/topo_lib/lib:$LD_LIBRARY_PATH

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-${DECODE_HOST}}"

export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-16}"
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-16}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"
# export USE_DCU_CUSTOM_ALLREDUCE="${USE_DCU_CUSTOM_ALLREDUCE:-1}"
export SGLANG_USE_AITER_AR=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_IB_GID_INDEX=0

# export SGLANG_USE_TRITON_VLLM_FA=1
# export SGLANG_DISAGG_KV_DEBUG=1
##blas
#export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/hyv3/gemm_config/hipblaslt.config
#export ROCBLAS_TENSILE_LIBPATH=/workspace/hyv3/gemm_config/rocblas_hy3_fp8_zmy_sglang
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"
export ALLREDUCE_STREAM_WITH_COMPUTE="${ALLREDUCE_STREAM_WITH_COMPUTE:-1}"
export HIP_KERNEL_EVENT_SYSTENFENCE="${HIP_KERNEL_EVENT_SYSTENFENCE:-1}"
export HIP_KERNEL_EVENT_SYSTEMFENCE="${HIP_KERNEL_EVENT_SYSTEMFENCE:-1}"
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD="${SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD:-0}"
export GLIBC_TUNABLES="${GLIBC_TUNABLES:-glibc.rtld.optional_static_tls=0x40000}"
export SGLANG_KVALLOC_KERNEL="${SGLANG_KVALLOC_KERNEL:-1}"
export SGLANG_SET_CPU_AFFINITY="${SGLANG_SET_CPU_AFFINITY:-1}"
export HIP_KERNEL_BATCH_CEILING="${HIP_KERNEL_BATCH_CEILING:-100}"
export GPU_FORCE_BLIT_COPY_SIZE="${GPU_FORCE_BLIT_COPY_SIZE:-16}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-3}"
export HSA_KERNARG_POOL_SIZE="${HSA_KERNARG_POOL_SIZE:-8388608}"
export ROC_AQL_QUEUE_SIZE="${ROC_AQL_QUEUE_SIZE:-131072}"
export HIP_GRAPH_ACCUMULATE_DISPATCH="${HIP_GRAPH_ACCUMULATE_DISPATCH:-0}"
export HIP_H2D_DISABLE_COPY_BUFFER="${HIP_H2D_DISABLE_COPY_BUFFER:-0}"
export HIP_D2H_DISABLE_COPY_BUFFER="${HIP_D2H_DISABLE_COPY_BUFFER:-0}"
export HIP_H2D_DIRECT_COPY_THRESHOLD="${HIP_H2D_DIRECT_COPY_THRESHOLD:-32768}"
export HIP_H2D_HSAAPI_COPY_THRESHOLD="${HIP_H2D_HSAAPI_COPY_THRESHOLD:-32768}"
export HIP_D2H_DIRECT_COPY_THRESHOLD="${HIP_D2H_DIRECT_COPY_THRESHOLD:-512}"
export HIP_D2H_HSAAPI_COPY_THRESHOLD="${HIP_D2H_HSAAPI_COPY_THRESHOLD:-512}"

export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-3737418240}"
export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-32}"
export ROCSHMEM_GDR_DISABLE_XDP="${ROCSHMEM_GDR_DISABLE_XDP:-1}"
export ROCSHMEM_IB_GID_INDEX="${ROCSHMEM_IB_GID_INDEX:-0}"
export ROCSHMEM_ALLOWED_IBV_DEVICES="${ROCSHMEM_ALLOWED_IBV_DEVICES:-${IB_DEVICES}}"
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
export SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH="${SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH:-1}"

if [[ -n "${ROCSHMEM_TOPO_FILE_FORCE:-}" ]]; then
  export ROCSHMEM_TOPO_FILE_FORCE
elif [[ -f /workspace/hyv3/s129_topo.config ]]; then
  export ROCSHMEM_TOPO_FILE_FORCE=/workspace/hyv3/s129_topo.config
fi

export SGLANG_TORCH_PROFILER_DIR="${SGLANG_TORCH_PROFILER_DIR:-/workspace/prof}"
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT="${SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT:-1200}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export SGLANG_USE_FP8_W8A8_MOE="${SGLANG_USE_FP8_W8A8_MOE:-1}"
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"   # rmsnorm+rope+kv 
export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_ENABLE_SPEC_V2="${SGLANG_ENABLE_SPEC_V2:-1}"
export SGLANG_USE_DEEPGEMM_MOE="${SGLANG_USE_DEEPGEMM_MOE:-1}"

export SGLANG_USE_LIGHTOP="${SGLANG_USE_LIGHTOP:-1}"
export SGLANG_USE_FUSED_RMS_ROTARY="${SGLANG_USE_FUSED_RMS_ROTARY:-0}"
export SGLANG_ROCM_USE_AITER_MOE="${SGLANG_ROCM_USE_AITER_MOE:-false}"
export W8A8_SUPPORT_METHODS="${W8A8_SUPPORT_METHODS:-2}"

export USE_DCU_CUSTOM_ALLREDUCE=0
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
export HIP_KERNEL_EVENT_SYSTENFENCE=1
# export SGLANG_USE_FP8_W8A8_MOE=1 # 开启 lightop的 Marlin MOE W8A8 算子
# 不要开启 FP8 W8A8 MoE dispatch
export SGLANG_USE_FP8_W8A8_MOE=0
unset SGLANG_DEEPEP_BF16_DISPATCH
export SGLANG_GROUPGEMM=true

if command -v sysctl >/dev/null 2>&1; then
  sysctl -w kernel.numa_balancing=0 >/dev/null 2>&1 || true
fi

NNODES="${NNODES:-${DEFAULT_NNODES}}"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
DP_SIZE="${DP_SIZE:-${DEFAULT_DP_SIZE}}"
MOE_DENSE_TP_SIZE="${MOE_DENSE_TP_SIZE:-1}"

SPEC_ALGO="${SPEC_ALGO:-NEXTN}"
SPEC_NUM_STEPS="${SPEC_NUM_STEPS:-2}"
SPEC_EAGLE_TOPK="${SPEC_EAGLE_TOPK:-1}"
SPEC_NUM_DRAFT_TOKENS="${SPEC_NUM_DRAFT_TOKENS:-3}"
mkdir -p "${LOG_DIR}"

  # --speculative-algorithm "${SPEC_ALGO}" \
  # --speculative-num-steps "${SPEC_NUM_STEPS}" \
  # --speculative-eagle-topk "${SPEC_EAGLE_TOPK}" \
  # --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
#export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/workspace/hyv3/eplb_trace
# python -m sglang.launch_server \
#   --model-path "${MODEL_PATH}" \
#   --tp-size "${PREFILL_TP_SIZE:-8}" \
#   --pp-size "${PREFILL_PP_SIZE:-8}" \
#   --trust-remote-code \
#   --enable-cache-report \
#   --dist-init-addr "${DIST_INIT_ADDR}" \
#   --host "${DECODE_BIND_HOST}" \
#   --port "${DECODE_PORT}" \
#   --attention-backend "${ATTENTION_BACKEND:-fa3}" \
#   --page-size "${PAGE_SIZE:-64}" \
#   --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC:-0.80}" \
#   --context-length "${DECODE_CONTEXT_LENGTH:-8192}" \
#   --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e5m2}" \
#   --max-running-requests 128 \
#   --chunked-prefill-size 4096 \
#   --disable-radix-cache \
#   --disaggregation-mode decode \
#   --disaggregation-ib-device "${IB_DEVICES}" \
#   "$@" \
#   2>&1 | tee "${LOG_DIR}/hunyuan_v3_d_${DECODE_HOST}_${DECODE_PORT}-1.log"

python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size "${DECODE_TP_SIZE:-8}" \
  --ep-size "${DECODE_EP_SIZE:-8}" \
  --dp-size 8 \
  --pp-size 1 \
  --nnodes 1 \
  --node-rank 0 \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size 1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --enable-cache-report \
  --host "${DECODE_BIND_HOST}" \
  --port "${DECODE_PORT}" \
  --attention-backend "${ATTENTION_BACKEND:-fa3}" \
  --page-size "${PAGE_SIZE:-64}" \
  --mem-fraction-static "${DECODE_MEM_FRACTION_STATIC:-0.80}" \
  --context-length "${DECODE_CONTEXT_LENGTH:-8192}" \
  --kv-cache-dtype fp8_e5m2 \
  --max-running-requests "${DECODE_MAX_RUNNING_REQUESTS:-320}" \
  --dtype bfloat16 \
  --quantization slimquant_marlin \
  --disable-radix-cache \
  --disaggregation-mode decode \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --ep-dispatch-algorithm static \
  --ep-num-redundant-experts 16 \
  --eplb-algorithm auto \
  --watchdog-timeout "${WATCHDOG_TIMEOUT:-36000}" \
  --log-requests \
  --log-level debug \
  --show-time-cost \
  --enable-metrics \
  --decode-log-interval 1 \
  "$@" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_d_${DECODE_HOST}_${DECODE_PORT}.log"
```

#### Router

```Bash
#!/usr/bin/env bash
set -euo pipefail

# Kunshan Hunyuan V3 PD router.
# EP8 default routes to the single decode service on 10.101.0.54.
# EP16 routes to the rank0 decode service only, for example:
#   DECODE_HOSTS="10.101.0.54 <second_decode_ip>" EP_SIZE=16 bash hunyuan_v3_pd_single_router.sh
# or:
#   DECODE_URLS="http://10.101.0.54:30001" bash hunyuan_v3_pd_single_router.sh

WORKDIR="${WORKDIR:-/workspace/project/test/pd-test/py_scr}"
SCRIPT_DIR="${SCRIPT_DIR:-${WORKDIR}}"
cd "${SCRIPT_DIR}"

PREFILL_HOST="${PREFILL_HOST:-10.211.9.35}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_BOOTSTRAP_PORT="${PREFILL_BOOTSTRAP_PORT:-8998}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_HOSTS="${DECODE_HOSTS:-10.211.9.36}"
EP_SIZE="${EP_SIZE:-8}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

if [[ -z "${DECODE_URLS:-}" ]]; then
  DECODE_URLS=""
  if [[ "${EP_SIZE}" == "16" ]]; then
    read -r decode_master_host _ <<< "${DECODE_HOSTS}"
    DECODE_URLS="http://${decode_master_host}:${DECODE_PORT}"
  else
    for host in ${DECODE_HOSTS}; do
      DECODE_URLS="${DECODE_URLS} http://${host}:${DECODE_PORT}"
    done
  fi
fi

mkdir -p "${LOG_DIR}"

ROUTER_ARGS=(
  --pd-disaggregation
  --prefill "http://${PREFILL_HOST}:${PREFILL_PORT}" "${PREFILL_BOOTSTRAP_PORT}"
  --policy "${ROUTER_POLICY:-round_robin}"
  --host "${ROUTER_HOST}"
  --port "${ROUTER_PORT}"
)

for decode_url in ${DECODE_URLS}; do
  ROUTER_ARGS+=(--decode "${decode_url}")
done

python -m sglang_router.launch_router "${ROUTER_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/hunyuan_v3_router_${ROUTER_PORT}.log"

```

### Hy3\-Channel\-FP8\-w8a8 IFB BW1100 8x SGLang 0\.5\.12 \(tp8\)

```Bash
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
```

## API 调用

### IFB

```Python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-INT8-w8a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```Bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy3-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```

