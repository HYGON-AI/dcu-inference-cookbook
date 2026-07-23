# SGLang HiCache



## HiCache简介

SGLang HiCache 在传统 RadixAttention 基础上，扩展了一种三层分层的键值（KV）缓存系统，显著提升了长上下文和多轮对话场景下的性能。通过智能管理 GPU 内存、主机内存以及外部存储后端的 KV 缓存，HiCache 解决了传统系统中限制缓存命中率的根本性容量瓶颈问题。



## 参数列表

|                 参数名                 |    默认值     |                            可选项                            |                             作用                             |
| :------------------------------------: | :-----------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|      --enable-hierarchical-cache       |     False     |                  bool flag (set to enable)                   |                         开启 HiCache                         |
|            --hicache-ratio             |      2.0      |                         Type: float                          | Host KV cache 相对 GPU KV cache 的容量比例。比如 2.0 表示 host cache token 容量约为 device cache 的 2 倍。 |
|             --hicache-size             |       0       |                          Type: int                           | 直接指定 host KV cache 大小，单位 GB。非 0 时优先级高于 --hicache-ratio。 |
|         --hicache-write-policy         | write_through |      write_back，write_through，write_through_selective      |              控制 GPU KV 什么时候写到 host/L2。              |
|          --hicache-io-backend          |    kernel     |                direct， kernel，kernel_ascend                |             CPU/GPU 之间搬 KV cache 的 IO 实现。             |
|          --hicache-mem-layout          |  page_first   | layer_first ，page_first，page_first_direct，page_first_kv_split，page_head | Host KV cache 内存布局。影响搬运 kernel、storage zero-copy 和 backend 兼容性。 |
|       --hicache-storage-backend        |     None      |   file, mooncake, hf3fs, nixl, aibrix, dynamic, eic, simm    | L3 storage backend。不开就是只有 GPU + host L2；设置后启用外部存储层。 |
|   --hicache-storage-prefetch-policy    |    timeout    |             best_effort, wait_complete, timeout              |             从 L3 storage 预取 KV 时的等待策略。             |
| --hicache-storage-backend-extra-config |     None      |                          Type: str                           | storage backend 的额外配置。可以是 JSON 字符串，也可以是 @config.toml/json/yaml 文件路径。 |



## 启动命令

### GLM-5-w4a8 IFB BW1000 8x

```
time=$(date "+%m%d-%H%M")
export NODE1_IP="xxx"   
export MODEL_PATH="/models/GLM-w4a8-V2_6_test"
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
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export NCCL_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=3
export SGLANG_HOST_IP=xxx

export MOONCAKE_TE_META_DATA_SERVER="http://xxx:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="64GB"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export MOONCAKE_MASTER=xxx:50051

sysctl -w kernel.numa_balancing=0 || true

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${NODE1_IP}" \
  --port 30000 \
  --nnodes 1 \
  --node-rank 0 \
  --dist-init-addr "${NODE1_IP}:5000" \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype bf16 \
  --mem-fraction-static 0.80 \
  --context-length 6000 \
  --page-size 64 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --nsa-prefill-backend flashmla_sparse  \
  --nsa-decode-backend flashmla_sparse \
  --quantization slimquant_w4a8_marlin \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --max-running-requests 512 \
  --enable-hierarchical-cache
  --hicache-ratio 1 \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy best_effort \
  2>&1 | tee tp_logs/tp8-$time.log
```

#### mooncake

```
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=xxx --http_metadata_server_port=8080
```



### Qwen3.5-397B-A17B TP16 BW1000 16x 

说明：MHA、GQA类注意力需要设置export SGLANG_KV_LAYOUT_DCU_FA=1、--hicache-mem-layout dcu_layout

#### node 0

```
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=1900000000
export ROCSHMEM_TOPO_FILE_FORCE="topo.config"
export USE_SPE_MQP=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1 
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export MC_TOPO_FILE_FORCE=mc_topo.config

export MOONCAKE_TE_META_DATA_SERVER="http://xxx:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="128GB"
export MOONCAKE_PROTOCOL="rdma"
export DEVICE_LIST="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=xxx:50051
#kv layout
export SGLANG_KV_LAYOUT_DCU_FA=1

sysctl -w kernel.numa_balancing=0

model_path=/models/Qwen3.5-397B-A17B
model=${model_path##*/}
tp=16 
pp=1
dp=1
ep=1
nodes=2
rank=0
#master_ip=
host_ip=$(hostname -I | awk '{print $1}')
hostname=$(hostname)
master_ip=$host_ip
max_model_len=6000
gpu_mem=0.80
time=$(date "+%m%d-%H%M")
mode="cudagraph"
logpath="server/$model-tp$tp-dp$dp-ep$ep-$hostname"


if [ ! -f ${logpath} ]; then
    mkdir ${logpath} -p
fi

python3 -m sglang.launch_server \
  --model-path "${model_path}" \
  --host "${host_ip}" \
  --port 30000 \
  --nnodes "${nodes}" \
  --node-rank "${rank}" \
  --dist-init-addr "${master_ip}:5000" \
  --tp-size "${tp}" \
  --pp-size "${pp}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --mem-fraction-static "${gpu_mem}" \
  --context-length "${max_model_len}" \
  --attention-backend fa3 \
  --page-size 64 \
  --numa-node 3 1 1 0 7 5 5 4 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size 16384 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout dcu_layout \
  --hicache-io-backend kernel \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  2>&1 | tee "${logpath}/${time}.log"
```

#### node 1

```
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export ROCSHMEM_MAX_NUM_CONTEXTS=48
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=1900000000
export ROCSHMEM_TOPO_FILE_FORCE="topo.config"
export USE_SPE_MQP=1
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export SGLANG_USE_LIGHTOP=1 
export MC_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export MC_TOPO_FILE_FORCE=mc_topo.config

export MOONCAKE_TE_META_DATA_SERVER="http://xxx:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="128GB"
export MOONCAKE_PROTOCOL="rdma"
export DEVICE_LIST="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9"
export MOONCAKE_DEVICE="$DEVICE_LIST"
export MOONCAKE_MASTER=xxx:50051
#kv layout
export SGLANG_KV_LAYOUT_DCU_FA=1

sysctl -w kernel.numa_balancing=0

model_path=/models/Qwen3.5-397B-A17B
model=${model_path##*/}
tp=16 
pp=1
dp=1
ep=1
nodes=2
rank=1
#master_ip=
host_ip=$(hostname -I | awk '{print $1}')
hostname=$(hostname)
master_ip=xxxx
max_model_len=6000
gpu_mem=0.80
time=$(date "+%m%d-%H%M")
mode="cudagraph"
logpath="server/$model-tp$tp-dp$dp-ep$ep-$hostname"


if [ ! -f ${logpath} ]; then
    mkdir ${logpath} -p
fi

python3 -m sglang.launch_server \
  --model-path "${model_path}" \
  --host "${host_ip}" \
  --port 30000 \
  --nnodes "${nodes}" \
  --node-rank "${rank}" \
  --dist-init-addr "${master_ip}:5000" \
  --tp-size "${tp}" \
  --pp-size "${pp}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --mem-fraction-static "${gpu_mem}" \
  --context-length "${max_model_len}" \
  --attention-backend fa3 \
  --page-size 64 \
  --numa-node 3 1 1 0 7 5 5 4 \
  --mamba-scheduler-strategy extra_buffer \
  --chunked-prefill-size 16384 \
  --enable-hierarchical-cache \
  --hicache-ratio 2 \
  --hicache-mem-layout dcu_layout \
  --hicache-io-backend kernel \
  --hicache-storage-backend mooncake \
  --hicache-write-policy write_through \
  --hicache-storage-prefetch-policy timeout \
  2>&1 | tee "${logpath}/${time}.log"

```

####  mooncake

node1或者node2中任选一个节点开启mooncake_master,--http_metadata_server_host 、MOONCAKE_TE_META_DATA_SERVER、MOONCAKE_MASTER设置为选择的节点的ip

```
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=xxx --http_metadata_server_port=8080
```



### GLM-5-W8A8 1P1D BW1100 16x 

说明:p节点设置--enable-hierarchical-cache，而d节点设置--disaggregation-decode-enable-offload-kvcache

#### P node 

```
mkdir -p prefill_logs
time=$(date "+%m%d-%H%M")
export NODE0_IP="xxx"
export MODEL_PATH="/models/GLM-5-W8A8"
export NCCL_SOCKET_IFNAME="ens14f0"
export GLOO_SOCKET_IFNAME="ens14f0"
export NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9"
export NCCL_IB_GID_INDEX=3
export USE_DCU_CUSTOM_ALLREDUCE=1
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
export MC_GID_INDEX=0
export SGLANG_USE_LIGHTOP=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_HOST_IP=10.16.1.22
# export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600

export MOONCAKE_TE_META_DATA_SERVER="http://xxx:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="64GB"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9"
export MOONCAKE_MASTER=xxx:50051

sysctl -w kernel.numa_balancing=0 || true

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${NODE0_IP}" \
  --port 30000 \
  --nnodes 1 \
  --node-rank 0 \
  --dist-init-addr "${NODE0_IP}:5000" \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --attn-cp-size 8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --context-length 131072 \
  --page-size 64 \
  --chunked-prefill-size 8192 \
  --pp-max-micro-batch-size 1 \
  --disable-cuda-graph \
  --enable-nsa-prefill-context-parallel \
  --nsa-prefill-cp-mode round-robin-split \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_marlin \
  --disaggregation-ib-device "${NCCL_IB_HCA}" \
  --disaggregation-mode prefill \
  --enable-hierarchical-cache \
  --hicache-ratio 1 \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy best_effort \
  2>&1 | tee prefill_logs/$time.log

```

#### D node 

```
mkdir -p decode_logs
time=$(date "+%m%d-%H%M")
export NODE2_IP="xxx"   # Decode master IP
export MODEL_PATH="/models/GLM-5-W8A8"

export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
# export SGLANG_ENABLE_SPEC_V2=1
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
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_GID_INDEX=0
export SGLANG_HOST_IP=xxx

export MOONCAKE_TE_META_DATA_SERVER="http://xxxx:8080/metadata"
export MOONCAKE_GLOBAL_SEGMENT_SIZE="64GB"
export MOONCAKE_PROTOCOL="rdma"
export MOONCAKE_DEVICE="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9"
export MOONCAKE_MASTER=xxxx:50051

sysctl -w kernel.numa_balancing=0 || true

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${NODE2_IP}" \
  --port 30003 \
  --nnodes 1 \
  --node-rank 0 \
  --dist-init-addr "${NODE2_IP}:5000" \
  --tp-size 8 \
  --pp-size 1 \
  --dp-size 1 \
  --ep-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --context-length 131072 \
  --page-size 64 \
  --disable-radix-cache \
  --chunked-prefill-size -1 \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_kv \
  --quantization slimquant_marlin \
  --dist-timeout 10000 \
  --watchdog-timeout 3600 \
  --moe-dense-tp-size 1 \
  --cuda-graph-max-bs 128 \
  --max-running-requests 512 \
  --disaggregation-ib-device "${NCCL_IB_HCA}" \
  --disaggregation-mode decode \
  --hicache-ratio 1 \
  --hicache-mem-layout page_first \
  --hicache-io-backend kernel \
  --hicache-write-policy write_through \
  --hicache-storage-backend mooncake \
  --hicache-storage-prefetch-policy best_effort \
  --disaggregation-decode-enable-offload-kvcache \
  2>&1 | tee decode_logs/$time.log

```

#### Mooncake

```
mooncake_master --enable_http_metadata_server=true --http_metadata_server_host=xxx --http_metadata_server_port=8080
```

