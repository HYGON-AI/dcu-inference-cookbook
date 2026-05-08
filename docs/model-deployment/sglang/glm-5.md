# GLM-5 on SGLang

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型         | 参数量  | 上下文  | 量化方式 | 推荐硬件                             |
| ---------- | ---- | ---- |---------| -------------------------------- |
| GLM-5-9B   | 9B   | 128K | BF16 | 1x DCU 64GB                      |
| GLM-5-25B  | 25B  | 128K | BF16 | 1x DCU 128GB / 2x DCU TP         |
| GLM-5-72B  | 72B  | 128K | BF16 | 2x DCU 128GB TP / 4x DCU 64GB TP |
| GLM-5-130B | 130B | 128K | BF16 | 4x DCU 128GB TP                  |

## 启动命令

### GLM-5-9B（单卡）

```bash
python -m sglang.launch_server \
    --model-path THUDM/GLM-5-9B \
    --tp-size 1 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### GLM-5-72B（四卡）

```bash
python -m sglang.launch_server \
    --model-path THUDM/GLM-5-72B \
    --tp-size 4 \
    --trust-remote-code \
    --mem-fraction-static 0.85
```

### GLM-5-W8A8

#### IFB【tp8】

```
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0

#mtp overlap
export SGLANG_ENABLE_SPEC_V2=1
# # triton改写算子
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

export HIP_GRAPH_ACCUMULATE_DISPATCH=0 #torchprof需要
export HIP_H2D_DISABLE_COPY_BUFFER=0 # 同步异步强制走WriteBuffer
export HIP_D2H_DISABLE_COPY_BUFFER=0 # 同步异步强制走ReadBuffer
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 # 小于此值走CPUCopy
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 # 大于此值走HSACOPY（CopyBuffer）
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 # 小于此值走CPUCopy
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 # 大于此值走HSACOPY（CopyBuffer）
#hiplaunch
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
# deep_ep
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
# mooncake
export MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ALLREDUCE_STREAM_WITH_COMPUTE=1
# 融合算子
export SGLANG_USE_LIGHTOP=1  # 使用lightop的rope和topk算子，可用

export HIP_GRAPH_ACCUMULATE_DISPATCH=0 #torchprof需要
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_MOE=0
#export TVM_HOME=/workspace/tilelang/3rdparty/tvm
#export TVM_LIBRARY_PATH=$TVM_HOME/build
#export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH


model_path=/module/GLM-5-W8A8
model=${model_path##*/}
tp=8
pp=1
dp=1
ep=1
nodes=1
rank=0
host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${host_ip}" ]; then
    host_ip=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
master_ip=$host_ip
max_model_len=6000
gpu_mem=0.8
port=30000
dist_port=5000
time=$(date "+%m%d-%H%M")
logpath="dserver/${model}-tp${tp}-dp${dp}-ep${ep}-pp${pp}-cp${attn_cp_size}-$(hostname)"
logfile="${logpath}/glm5_int8_${time}.log"

mkdir -p "${logpath}"

option="--numa-node 0 0 0 0 1 1 1 1 "
option+=" --disable-radix-cache "
#option+=" --chunked-prefill-size -1 "
option+=" --page-size 64 "
option+=" --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv "
option+=" --quantization slimquant_marlin "
option+=" --disable-cuda-graph"


python3 -m sglang.launch_server --model-path "${model_path}" ${option} \
                                --trust-remote-code \
                                --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --mem-fraction-static "${gpu_mem}" \
                                --host "${host_ip}" --port "${port}" --dist-init-addr "${master_ip}:${dist_port}" \
                                --nnodes "${nodes}" --node-rank "${rank}" \
                                --tp-size "${tp}" --pp-size "${pp}" --dp-size "${dp}" --ep-size "${ep}" \
                                2>&1 | tee "${logfile}"

```

#### PD分离

##### P节点【tp8cp8】：

```
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_TORCH_PROFILER_DIR=/workspace/profile
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
sysctl -w kernel.numa_balancing=0

#mtp overlap
export SGLANG_ENABLE_SPEC_V2=1
# # triton改写算子
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

export HIP_GRAPH_ACCUMULATE_DISPATCH=0 #torchprof需要
export HIP_H2D_DISABLE_COPY_BUFFER=0 # 同步异步强制走WriteBuffer
export HIP_D2H_DISABLE_COPY_BUFFER=0 # 同步异步强制走ReadBuffer
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 # 小于此值走CPUCopy
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 # 大于此值走HSACOPY（CopyBuffer）
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 # 小于此值走CPUCopy
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 # 大于此值走HSACOPY（CopyBuffer）
#hiplaunch
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
# deep_ep
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_TOPO_FILE_FORCE=/home/topo.config
# mooncake
export MC_TOPO_FILE_FORCE=/home/mc_topo_400g.config
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ALLREDUCE_STREAM_WITH_COMPUTE=1
# 融合算子
export SGLANG_USE_LIGHTOP=1  # 使用lightop的rope和topk算子，可用

export HIP_GRAPH_ACCUMULATE_DISPATCH=0 #torchprof需要
export HIP_GRAPH_USE_CMD_CACHE=0
export SGLANG_ROCM_USE_AITER_MOE=0


model_path=/module/GLM-5-W8A8
model=${model_path##*/}
tp=8
pp=1
dp=1
ep=1
attn_cp_size=8
nodes=1
rank=0
host_ip=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${host_ip}" ]; then
    host_ip=$(hostname -i 2>/dev/null | awk '{print $1}')
fi
master_ip=$host_ip
max_model_len=6000
gpu_mem=0.8
port=30000
dist_port=5000
time=$(date "+%m%d-%H%M")
logpath="dserver/${model}-tp${tp}-dp${dp}-ep${ep}-pp${pp}-cp${attn_cp_size}-$(hostname)"
logfile="${logpath}/glm5_int8_${time}.log"

mkdir -p "${logpath}"

option="--numa-node 0 0 0 0 1 1 1 1 "
option+=" --disable-radix-cache "
option+=" --chunked-prefill-size -1 "
option+=" --page-size 64 "
option+=" --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv "
option+=" --quantization slimquant_marlin "
option+=" --attn-cp-size ${attn_cp_size} "
option+=" --enable-nsa-prefill-context-parallel  --nsa-prefill-cp-mode round-robin-split "
option+=" --disable-cuda-graph"
option+=" --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
option+=" --disaggregation-mode prefill "

python3 -m sglang.launch_server --model-path "${model_path}" ${option} \
                                --trust-remote-code \
                                --kv-cache-dtype fp8_e4m3 --dtype bfloat16 --mem-fraction-static "${gpu_mem}" \
                                --host "${host_ip}" --port "${port}" --dist-init-addr "${master_ip}:${dist_port}" \
                                --nnodes "${nodes}" --node-rank "${rank}" \
                                --tp-size "${tp}" --pp-size "${pp}" --dp-size "${dp}" --ep-size "${ep}" \
                                2>&1 | tee "${logfile}"

```

##### D节点【ep16dp16-mtp3】

**tp1-rocblas【72cu】**

```
curl -f -C - -o library_gpu6_glm5_int8.tar.gz https://wuzh01.hpccube.com:65015/efile/s/d/Z3VvYmo=/33a916b99f968820  
```

**node0**

```
#!/usr/bin/env bash
set -euo pipefail

# =========================
# 环境变量
# =========================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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

# =========================
# 可改配置
# =========================
MODEL_PATH="/module/GLM-5-W8A8/"

# 自动获取本机 IP
HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
    HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

PORT="30023"

NNODES="2"
NODE_RANK="0"

MASTER_IP="${HOST}"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

TP_SIZE="16"
PP_SIZE="1"
DP_SIZE="16"
EP_SIZE="16"

CUDA_GRAPH_MAX_BS="32"
MAX_RUNNING_REQUESTS="512"

SPEC_ALGO="EAGLE"
SPEC_NUM_STEPS="3"
SPEC_TOPK="1"
SPEC_NUM_DRAFT_TOKENS="4"

MEM_FRACTION_STATIC="0.8"
CONTEXT_LENGTH="131072"
PAGE_SIZE="64"
KV_CACHE_DTYPE="fp8_e4m3"

DISAGG_MODE="decode"
IB_DEVICES="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"

# =========================
# 日志配置
# =========================
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIME_STR=$(date "+%Y%m%d-%H%M%S")
HOST_NAME=$(hostname)

LOG_FILE="${LOG_DIR}/decode_${SPEC_ALGO,,}_s${SPEC_NUM_STEPS}_k${SPEC_TOPK}_d${SPEC_NUM_DRAFT_TOKENS}_tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_ep${EP_SIZE}_node${NODE_RANK}_${HOST_NAME}_${TIME_STR}.log"

echo "========================================"
echo "log file: $LOG_FILE"
echo "host    : $HOST"
echo "port    : $PORT"
echo "master  : $MASTER_IP"
echo "dist    : $DIST_INIT_ADDR"
echo "node    : $NODE_RANK/$NNODES"
echo "tp/pp/dp/ep: ${TP_SIZE}/${PP_SIZE}/${DP_SIZE}/${EP_SIZE}"
echo "spec    : ${SPEC_ALGO} steps=${SPEC_NUM_STEPS} topk=${SPEC_TOPK} draft=${SPEC_NUM_DRAFT_TOKENS}"
echo "========================================"

python3 -m sglang.launch_server \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length "${CONTEXT_LENGTH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size "${PAGE_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --model-path "${MODEL_PATH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --moe-dense-tp-size 1 \
  --dp-size "${DP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --port "${PORT}" \
  --host "${HOST}" \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-mode "${DISAGG_MODE}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --speculative-algorithm "${SPEC_ALGO}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  2>&1 | tee "$LOG_FILE"
```

**node1**

```
#!/usr/bin/env bash
set -euo pipefail

# =========================
# 环境变量
# =========================
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
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
# export TVM_HOME=/workspace/tilelang/3rdparty/tvm
# export TVM_LIBRARY_PATH=$TVM_HOME/build
# export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
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


# =========================
# 可改配置
# =========================
MODEL_PATH="/module/GLM-5-W8A8/"

# 自动获取本机 IP
HOST=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST}" ]; then
    HOST=$(hostname -i 2>/dev/null | awk '{print $1}')
fi

PORT="30023"

NNODES="2"
NODE_RANK="1"

MASTER_IP="10.16.1.42"
DIST_PORT="5000"
DIST_INIT_ADDR="${MASTER_IP}:${DIST_PORT}"

TP_SIZE="16"
PP_SIZE="1"
DP_SIZE="16"
EP_SIZE="16"

CUDA_GRAPH_MAX_BS="32"
MAX_RUNNING_REQUESTS="512"

SPEC_ALGO="EAGLE"
SPEC_NUM_STEPS="3"
SPEC_TOPK="1"
SPEC_NUM_DRAFT_TOKENS="4"

MEM_FRACTION_STATIC="0.8"
CONTEXT_LENGTH="131072"
PAGE_SIZE="64"
KV_CACHE_DTYPE="fp8_e4m3"

DISAGG_MODE="decode"
IB_DEVICES="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"

# =========================
# 日志配置
# =========================
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIME_STR=$(date "+%Y%m%d-%H%M%S")
HOST_NAME=$(hostname)

LOG_FILE="${LOG_DIR}/decode_${SPEC_ALGO,,}_s${SPEC_NUM_STEPS}_k${SPEC_TOPK}_d${SPEC_NUM_DRAFT_TOKENS}_tp${TP_SIZE}_pp${PP_SIZE}_dp${DP_SIZE}_ep${EP_SIZE}_node${NODE_RANK}_${HOST_NAME}_${TIME_STR}.log"

echo "========================================"
echo "log file: $LOG_FILE"
echo "host    : $HOST"
echo "port    : $PORT"
echo "master  : $MASTER_IP"
echo "dist    : $DIST_INIT_ADDR"
echo "node    : $NODE_RANK/$NNODES"
echo "tp/pp/dp/ep: ${TP_SIZE}/${PP_SIZE}/${DP_SIZE}/${EP_SIZE}"
echo "spec    : ${SPEC_ALGO} steps=${SPEC_NUM_STEPS} topk=${SPEC_TOPK} draft=${SPEC_NUM_DRAFT_TOKENS}"
echo "========================================"

python3 -m sglang.launch_server \
  --pp-size "${PP_SIZE}" \
  --tp-size "${TP_SIZE}" \
  --nsa-prefill-backend flashmla_auto \
  --nsa-decode-backend flashmla_kv \
  --context-length "${CONTEXT_LENGTH}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --page-size "${PAGE_SIZE}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --model-path "${MODEL_PATH}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --quantization slimquant_marlin \
  --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
  --moe-dense-tp-size 1 \
  --dp-size "${DP_SIZE}" \
  --ep-size "${EP_SIZE}" \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --enable-dp-lm-head \
  --port "${PORT}" \
  --host "${HOST}" \
  --disaggregation-ib-device "${IB_DEVICES}" \
  --disaggregation-mode "${DISAGG_MODE}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --speculative-algorithm "${SPEC_ALGO}" \
  --speculative-num-steps "${SPEC_NUM_STEPS}" \
  --speculative-eagle-topk "${SPEC_TOPK}" \
  --speculative-num-draft-tokens "${SPEC_NUM_DRAFT_TOKENS}" \
  --nnodes "${NNODES}" \
  --node-rank "${NODE_RANK}" \
  --dist-init-addr "${DIST_INIT_ADDR}" \
  2>&1 | tee "$LOG_FILE"
```

##### SGLang-Router

```
python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://10.16.1.36:30000 --decode http://10.16.1.46:30023 --policy cache_aware --port 30005
```

### GLM-5-FP8

同上glm-5-int8

**tp1-rocblas【72cu】**

```
curl -f -C - -o library_gpu6_glm5_fp8.tar.gz https://wuzh01.hpccube.com:65015/efile/s/d/Z3VvYmo=/42f45aa74226c4b3  
```





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

## DCU 适配注意

- GLM-5 原生支持 bf16，DCU 兼容性良好
- 需要 `--trust-remote-code`
- 中文场景下 RadixAttention 对重复 system prompt 优化明显
