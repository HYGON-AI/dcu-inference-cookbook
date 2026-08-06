# Mooncake on HCU

> 本文档涵盖 Mooncake 在 HCU 环境下的安装、配置、环境变量说明以及各框架（SGLang / vLLM）的 PD 分离部署和 Mooncake Store 集成方案。

- [简介](#简介)
- [安装](#安装)
- [环境变量](#环境变量)
  - [LOG](#log)
  - [RDMA](#rdma)
  - [HIP IPC/RPC](#HIP-IPCRPC)
  - [Topology](#Topology)
- [Mooncake bench 测试](#mooncake-bench-测试)
  - [Transfer engine bench(RDMA)](#Transfer-engine-benchRDMA)
  - [Transfer engine bench(HIP IPC)](#Transfer-engine-benchHIP-IPC)
  - [Transfer engine bench(HIP RPC)](#Transfer-engine-benchHIP-RPC)
- [SGLang PD 分离](#sglang-pd-分离)
  - [SGLang 单节点 1P1D 测试](#SGLang-单节点-1P1D-测试)
  - [SGLang 双节点 1P1D 测试](#SGLang-双节点-1P1D-测试)
- [vLLM PD 分离](#vllm-pd-分离)
  - [P、D 单实例单节点(1P1D)](#pd-单实例单节点1P1D)
  - [P：TP8  D：DP8EP8 (1P1D)](#ptp8--ddp8ep8-1p1d)
  - [P：PP16  D：TP8 (1P1D)](#ppp16--dtp8-1p1d)
  - [P：SP8  D：DP16EP16 (1P1D)](#psp8--ddp16ep16-1p1d)
  - [P：SP8EP8  D：DP16EP16(2P1D)](#PSP8EP8--DDP16EP162P1D)
  - [环境变量](#vllm环境变量)
- [SGLang HiCache with Mooncake Backend](#sglang-hicache-with-mooncake-backend)
  - [内存与拓扑配置](#内存与拓扑配置)
  - [释放页缓存](#释放页缓存)
  - [Mooncake Master 配置](#mooncake-master-配置)
  - [跨节点内存交错](#跨节点内存交错)
  - [动态扩容共享内存](#动态扩容共享内存)
  - [HiCache CPU 内存使用](#hicache-cpu-内存使用)
- [LMCache with Mooncake Backend](#lmcache-with-mooncake-backend)
  - [启动 Mooncake Store](#启动-mooncake-store)
  - [LMCache 配置文件](#lmcache-配置文件)
  - [启动 vLLM](#启动-vllm)
- [TransferEngine API](#transferengine-api)
  - [创建与初始化](#创建与初始化)
  - [注册内存](#注册内存)
  - [数据传输（同步写）](#数据传输同步写)
  - [注销内存](#注销内存)
  - [示例：跨节点传输](#示例跨节点传输)

## 简介

Mooncake 是一个面向大语言模型（LLM）推理场景的 KV Cache 传输与缓存卸载框架，核心目标是解决 PD 分离架构下 KV Cache 在 Prefill 和 Decode 节点间的高效传输问题，以及将 KV Cache 从 GPU 显存卸载到主机内存或 SSD 以降低推理成本。

Mooncake 的核心能力包括：

- **多协议传输**：支持 RDMA、TCP 等多种传输协议，在 HCU 环境下还可启用 HIP IPC（节点内）和 HIP RPC（跨节点）传输，灵活适配不同网络硬件
- **多级缓存池**：在 GPU 显存（L1）、主机 DRAM（L2）、SSD（L3）上构建分层缓存，平衡推理速度与容量
- **即插即用**：已深度集成至 SGLang 和 vLLM，通过 `--kv-transfer-config` 或 `--disaggregation-mode` 即可启用 PD 分离
- **缓存后端**：可作为 LMCache 的远端存储后端（Mooncake Store），支持内存卸载与 SSD 卸载

官方文档：<https://kvcache-ai.github.io/Mooncake/>

## 安装

hcu mooncake 代码仓库：https://developer.sourcefind.cn/codes/OpenDAS/mooncake

hcu mooncake whl 包（从 DAS PyPI 下载）：

```bash
# mooncake_transfer_engine：通用版本，普通网卡，不带 hylink 支持
https://pypi.sourcefind.cn/das_nightly/dtk2604-rc4/+f/2d3/fb98366b63bfc/mooncake_transfer_engine-0.3.10.post1+das.dtk2604.2606261349.g1476ea-cp310-cp310-manylinux_2_35_x86_64.whl#sha256=2d3fb98366b63bfc38bd34f8814b4a25f2e17a56e692cff7bff476c259a3165a

# mooncake_transfer_engine_rpc：带 hylink 支持的版本，普通网卡
https://pypi.sourcefind.cn/das_nightly/dtk26041-rc1/+f/82d/edbff272495c8/mooncake_transfer_engine_rpc-0.3.10.post1+das.dtk26041.2606261403.g1476ea-cp310-cp310-manylinux_2_35_x86_64.whl#sha256=82dedbff272495c84accfd2b4f8c306b38bb2f356fbc914dd3fef2bd06dbf31f

# mooncake_transfer_engine_shca：支持天龙网卡的版本，不带 hylink 支持（超节点天龙网卡要用正常版本）
https://pypi.sourcefind.cn/das_nightly/dtk2604-rc4/+f/cde/6179babb6b70b/mooncake_transfer_engine_shca-0.3.10.post1+das.dtk2604.2606261356.g1476ea-cp310-cp310-manylinux_2_35_x86_64.whl#sha256=cde6179babb6b70bbeaf206a0221abe150a11f0754f23e4d65a6b117b451d8ce

#安装 mooncake：
pip install mooncake_transfer_engine*.whl

#安装前请先卸载旧版本
pip uninstall mooncake_transfer_engine
```

## 环境变量

### LOG

```bash
# 可选：TRACE / INFO / WARNING / ERROR，默认 INFO
export MC_LOG_LEVEL=TRACE
```

### RDMA

```bash
# sglang 握手失败时可以通过设置 host ip 解决
export SGLANG_HOST_IP=${HOST_IP}
# vllm 握手失败时可以通过设置 host ip 解决
export VLLM_HOST_IP=${HOST_IP}

# 存在跨 SM IB NIC transfer 问题时，启用设备亲和性。非必要不设置
# 启用后，Transfer Engine 将优先选择和本地网卡同名的远端网卡进行通信。
# 默认值为 false。存在跨sm ib nic transfer 问题时，可尝试设置。
# 双平面需要设置
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

# 同一交换机内通信可以尝试切换 GID
# 默认为3（global 路由），让 RDMA 通信走带全局路由头 GRH 的 GID 地址，交换机识别后按 GID 跨三层转发
# 0（link-local），只能在同一交换机/本地子网进行通信，不支持跨交换机、跨子网通信
export MC_IB_GID_INDEX=0

# 只有某些网卡可见时，可以设置
export MC_TE_FILTERS=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
# 显式启用MC_IB_PCI_RELAXED_ORDERING时，可能会导致Failed to register memory，此时不支持该特性。
```

### HIP IPC/RPC

```bash
# 启用 HIP IPC Transport（仅用于节点内通信）
export MC_FOCRE_MNNVL=1

# 启用 HIP RPC Transport（用于节点内和节点间通信，需安装 mooncake_transfer_engine_rpc）
export MC_FOCRE_MNNVL=1
export MC_USE_HIP_IPC=0

# 修改 copy kernel cu number
export MC_HIP_COPY_BLOCKS=xxx
```

### Topology 

查看自动发现 Topology，自动发现会将和 CPU 或 GPU 同 NUMA 的 NIC 放 preferred_hca，其余放 avail_hca；对于 GPU，若没有同 NUMA 的 NIC，则会将距离最近（两个 PCI 设备在 Linux PCI 树上的拓扑距离估计，并非NUMA distance）的 NIC 放 preferred_hca。
有关 topo 的信息请参阅显存与计算拓扑和主机内存与拓扑部分。

```bash
#查看自动发现 Topology
transfer_engine_topology_dump
```

当多个 GPU 都优先使用同一张 NIC 时，可能导致该网卡负载过大，或者需要限定使用的 NIC 时，可以手动设定 Topo。
自定义拓扑JSON文件路径 MC_CUSTOM_TOPO_JSON

```json
//JSON格式如下
 {
   "<storage_type>": [
     ["preferred_hca..."],
     ["avail_hca..."]
   ]
 }

//也就是每个 key 的 value 都是长度为 2 的数组：
//第一项：优先使用的 RDMA 网卡列表  
//第二项：可回退使用的网卡列表
//示例(topo.json)：
{
   "cpu:0":  [["mlx5_2"], []],
   "cpu:1":  [["mlx5_3"], []],
   "cpu:2":  [["mlx5_4"], []],
   "cpu:3":  [["mlx5_5"], []],
   "cpu:4":  [["mlx5_6"], []],
   "cpu:5":  [["mlx5_7"], []],
   "cpu:6":  [["mlx5_8"], []],
   "cpu:7":  [["mlx5_9"], []],
   "hip:0":  [["mlx5_2"], []],
   "hip:1":  [["mlx5_3"], []],
   "hip:2":  [["mlx5_4"], []],
   "hip:3":  [["mlx5_5"], []],
   "hip:4":  [["mlx5_6"], []],
   "hip:5":  [["mlx5_7"], []],
   "hip:6":  [["mlx5_8"], []],
   "hip:7":  [["mlx5_9"], []]
 }
```

## Mooncake bench 测试

### Transfer engine bench(RDMA)

RDMA 一般用于节点间测试，节点内也支持。
`transfer_engine_bench` 在 `pip show mooncake-transfer-engine` 显示的安装路径下。

```bash
# Node 1（target）
# 请将 `<local_host_ip>` 替换为当前节点 IP
transfer_engine_bench --mode=target --auto_discovery --protocol=rdma \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --local_server_name=${local_host_ip}

# Node 2（initiator）
# 请将 `<local_host_ip>` 替换为当前节点 IP，`<remote_host_ip>` 替换为对端节点 IP
# port: 从 node1 log 里找到 ZZZ，Transfer Engine RPC using XXX, listening on YYY:ZZZ）
transfer_engine_bench --mode=initiator --auto_discovery --protocol=rdma \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --local_server_name=${local_host_ip} --segment_id=${remote_host_ip}:${port}

#测试最大 throughput 推荐参数
--buffer_size=4294967296 --threads=12 --batch_size=128 --block_size=2097152
#MC_SLICE_SIZE 可以调整为 131072，甚至可以更大。
```

### Transfer engine bench(HIP IPC)

HIP IPC 用于节点内测试。
transfer_engine_bench 路径：“pip show mooncake-transfer-engine” ，transfer_engine_bench 在安装路径下。

```bash
#node 1
export MC_FORCE_HIP=1
transfer_engine_bench --mode=target --protocol=hip \
    --metadata_server=P2PHANDSHAKE --gpu_id=0 \
    --local_server_name=127.0.0.1

#node 1
# port: 从 node1 log 里找到 ZZZ，Transfer Engine RPC using XXX, listening on YYY:ZZZ）
export MC_FORCE_HIP=1
transfer_engine_bench --mode=initiator --protocol=hip \
    --metadata_server=P2PHANDSHAKE --gpu_id=1 \
    --local_server_name=127.0.0.1 --segment_id=127.0.0.1:${port}
```

### Transfer engine bench(HIP RPC)

HIP RPC 用于节点间测试。
transfer_engine_bench 路径：“pip show mooncake-transfer-engine” ，transfer_engine_bench 在安装路径下。

```bash
#node 1
export MC_FORCE_HIP=1
export MC_USE_HIP_IPC=0
# local_host_ip: node 1 ip
transfer_engine_bench --mode=target --protocol=hip \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --duration 20 \
    --local_server_name=${local_host_ip}

#node 2
# port: 从 node1 log 里找到 ZZZ，Transfer Engine RPC using XXX, listening on YYY:ZZZ）
export MC_FORCE_HIP=1
export MC_USE_HIP_IPC=0
transfer_engine_bench --mode=initiator --protocol=hip \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --duration 20 \
    --local_server_name=${local_host_ip} --segment_id=${remote_host_ip}:${port}

#测试最大 throughput 推荐参数
--buffer_size=4294967296 --threads=12 --batch_size=2 --block_size=536870912
```

## SGLang PD 分离

### SGLang 单节点 1P1D 测试

```bash
# Prefill 节点
python -m sglang.launch_server \
    --model-path=/model/Qwen3-8B \
    --disaggregation-mode prefill \
    --port 30000 \
    --attention-backend=fa3 \
    --page-size=64

# Decode 节点（--base-gpu-id 用于指定 GPU 起始编号）
python -m sglang.launch_server \
    --model-path=/model/Qwen3-8B \
    --disaggregation-mode decode \
    --port 30001 \
    --base-gpu-id 1 \
    --attention-backend=fa3 \
    --page-size=64

# Router
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 \
    --decode http://127.0.0.1:30001 \
    --host 0.0.0.0 --port 8000
```

### SGLang 双节点 1P1D 测试

```bash
# Prefill 节点（Node 1）
python -m sglang.launch_server \
    --model-path=/model/Qwen3-8B \
    --disaggregation-mode prefill \
    --attention-backend=fa3 \
    --page-size=64 \
    --host 10.16.1.58 \
    --port 30002 \
    --dist-init-addr 10.16.1.58:5000 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# Decode 节点（Node 2）
python -m sglang.launch_server \
    --model-path=/model/Qwen3-8B \
    --disaggregation-mode decode \
    --base-gpu-id 1 \
    --attention-backend=fa3 \
    --page-size=64 \
    --host 10.16.1.60 \
    --port 30002 \
    --dist-init-addr 10.16.1.58:5000 \
    --nnodes 1 \
    --node-rank 0 \
    --disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9

# Router
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://10.16.1.58:30002 \
    --decode http://10.16.1.60:30002 \
    --host 0.0.0.0 --port 8000
```

## vLLM PD 分离

### P、D 单实例单节点(1P1D)

```bash
# KV Producer（Prefill）
vllm serve Qwen3/Qwen3-8B \
    --port 8010 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# KV Consumer（Decode）
HIP_VISIBLE_DEVICES=1 vllm serve Qwen3/Qwen3-8B \
    --port 8020 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Mooncake Connector Proxy
# mooncake_connector_proxy.py 在 vllm 源码中：
# https://developer.sourcefind.cn/codes/OpenDAS/vllm/-/tree/v0.18.1/examples/online_serving/disaggregated_serving/mooncake_connector
python3 vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://0.0.0.0:8010" "8998" \
    --decode "http://0.0.0.0:8020" \
    --port 8000

#测试
curl http://localhost:8000/v1/completions \  
    -H "Content-Type: application/json" \  
    -d '{    
        "model": "Qwen/Qwen3-8B",    
        "messages": [      
            {"role": "user", "content": "Tell me a long story about artificial intelligence."}    
        ]  
    }'
```

### P：TP8  D：DP8EP8 (1P1D)

```bash
# KV Producer（Prefill）
vllm serve /models/vllm-w8a8-models/GLM-5-W8A8 \
    -q slimquant_marlin \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    -tp 8 \
    --port 9348 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --block-size 64 \
    --kv-cache-dtype fp8_ds_mla \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# KV Consumer（Decode）
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_HCU_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_HCU_USE_FLASHMLA=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

vllm serve /models/vllm-w8a8-models/GLM-5-W8A8  \
    -q slimquant_marlin \
    --trust-remote-code \  
    --dtype bfloat16 \  
    --max-model-len 65536 \  
    --max-num-batched-tokens 128 \  
    -dp 8 \  
    --port 9349 \  
    --max-num-seqs 64 \  
    --gpu-memory-utilization 0.92 \  
    --block-size 64 \  
    --kv-cache-dtype fp8_ds_mla \  
    --enable-expert-parallel \  
    --all2all-backend deepep_low_latency \  
    --disable-custom-all-reduce \  
    --enable-chunked-prefill \  
    --enable-prefix-caching \  
    -cc '{"inductor_compile_config":{"combo_kernels": false}}' \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Mooncake Connector Proxy
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \  
    --prefill "http://10.16.1.15:9348" "8998" \  
    --decode "http://10.16.1.16:9349" \  
    --port 8000
```

### P：PP16  D：TP8 (1P1D)

```bash
# KV Producer（Prefill）
10.16.1.15:
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
ray start --head --node-ip-address=10.16.1.15  --port=1255 --num-gpus=8 --num-cpus=32
10.16.1.16:
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
ray start --address=10.16.1.15:1255  --num-gpus=8 --num-cpus=32
10.16.1.15:
vllm serve /models/v2_6/GLM-w4a8-V2_6_test \
    --trust-remote-code \
    -pp 16 \
    --dtype bfloat16 \  
    --max-model-len 65536 \  
    --max-num-batched-tokens 8192 \  
    --max-num-seqs 64 \  
    --kv-cache-dtype fp8_ds_mla \  
    --gpu-memory-utilization 0.9 \  
    --distributed-executor-backend ray \  
    --enforce-eager \  
    -cc '{"inductor_compile_config":{"combo_kernels": false}}' \  
    --port 9348 \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# KV Consumer（Decode）
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_DP_MASTER_IP=10.16.1.15
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
vllm serve /models/v2_6/GLM-w4a8-V2_6_test \
    --trust-remote-code \  
    -tp 8 \  
    --dtype bfloat16 \  
    --max-model-len 65536 \  
    --max-num-batched-tokens 8192 \  
    --max-num-seqs 64 \  
    --kv-cache-dtype fp8_ds_mla \  
    --gpu-memory-utilization 0.9 \  
    -cc '{"inductor_compile_config":{"combo_kernels": false}}' \  
    --port 9349 \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Mooncake Connector Proxy
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://10.16.1.15:9348" "8998" \  
    --decode "http://10.16.1.18:9349" \  
    --port 8000
```

### P：SP8  D：DP16EP16 (1P1D)

```bash
# KV Producer（Prefill）
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export HIPBLASLT_TUNING_OVERRIDE_FILE=/workspace/rocblas/hipblaslt.config
export ROCBLAS_TENSILE_LIBPATH=/workspace/rocblas/rocblas_hy3_fp8_zmy
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
LMSLIM_USE_FUSED_RMS_QUANT=1 \
VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=0 \
vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
    --speculative-config.method mtp \  
    --speculative-config.num_speculative_tokens 2 \  
    --max-model-len 65536 \  
    --max-num-batched-tokens 8192 \  
    --max-num-seqs 128 \  
    --dtype bfloat16 \  
    --tensor-parallel-size 8 \  
    --no-enable-prefix-caching \  
    --tool-call-parser hy_v3 \  
    --reasoning-parser hy_v3 \  
    --enable-auto-tool-choice \  
    --enable-custom-sp \  
    --enforce-eager \  
    --kv_cache_dtype fp8_e4m3 \  
    --port 8010 \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# KV Consumer（Decode）
export VLLM_HOST_IP=10.16.1.16
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export VLLM_TORCH_PROFILER_DIR=/data/vllm_profile
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=0
export USE_FUSED_RMS_QUANT=0
export USE_FUSED_SILU_MUL_QUANT=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_CUSTOM_CACHE=0
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0
export VLLM_USE_V32_ENCODE=1
export VLLM_HCU_USE_FLASHMLA=1
export VLLM_HCU_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export VLLM_MOE_DP_CHUNK_SIZE=128
export ROCSHMEM_IB_GID_INDEX=0
export VLLM_USE_LIGHTOP=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
    --trust-remote-code \  
    -dp 16 \  
    -tp 1 \  
    --enable-expert-parallel \  
    --disable-custom-all-reduce \  
    --dtype bfloat16 \  
    --enable-chunked-prefill \  
    --max-model-len 53000 \  
    --enable-prefix-caching \  
    --block-size 64 \  
    --gpu-memory-utilization 0.89 \  
    --data-parallel-size-local 8 \  
    --data-parallel-address 10.16.1.16 \  
    --data-parallel-rpc-port 1127 \  
    --data-parallel-start-rank 0 \  
    --kv-cache-dtype fp8_e4m3 \  
    -q slimquant_marlin \  
    --max-num-seqs 256 \  
    --all2all_backend=deepep_low_latency \  
    --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}' \  
    --port 8020 \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
10.16.1.18:
export VLLM_HOST_IP=10.16.1.18
export NCCL_SOCKET_IFNAME=ens14f0
export GLOO_SOCKET_IFNAME=ens14f0
export VLLM_TORCH_PROFILER_DIR=/data/vllm_profile
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=0
export USE_FUSED_RMS_QUANT=0
export USE_FUSED_SILU_MUL_QUANT=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_CUSTOM_CACHE=0
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0
export VLLM_USE_V32_ENCODE=1
export VLLM_HCU_USE_FLASHMLA=1
export VLLM_HCU_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/workspace/topo.config
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export VLLM_MOE_DP_CHUNK_SIZE=128
export ROCSHMEM_IB_GID_INDEX=0
export VLLM_USE_LIGHTOP=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
vllm serve /models/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
    --trust-remote-code \  
    -dp 16 \  
    -tp 1 \  
    --enable-expert-parallel \  
    --disable-custom-all-reduce \  
    --dtype bfloat16 \  
    --enable-chunked-prefill \  
    --max-model-len 53000 \  
    --enable-prefix-caching \  
    --block-size 64 \  
    --gpu-memory-utilization 0.89 \  
    --data-parallel-size-local 8 \  
    --data-parallel-address 10.16.1.16 \  
    --data-parallel-rpc-port 1127 \  
    --data-parallel-start-rank 8 \  
    --kv-cache-dtype fp8_e4m3 \  
    -q slimquant_marlin \  
    --max-num-seqs 256 \  
    --headless \  
    --all2all_backend=deepep_low_latency \  
    --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}' \  
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Mooncake Connector Proxy
python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://10.16.1.15:8010" "8998" \  
    --decode "http://10.16.1.16:8020" \  
    --port 8000
```

### P：SP8EP8  D：DP16EP16(2P1D)

```bash
# Prefill 节点 1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/hy3/rocblas/hipblaslt.config
export ROCBLAS_TENSILE_LIBPATH=/home/hy3/rocblas/rocblas_hy3_fp8_zmy
export GPU_MAX_HW_QUEUES=4

export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

export ROCSHMEM_HEAP_SIZE=4000000000
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024

export ROCSHMEM_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998
export ALLREDUCE_STREAM_WITH_COMPUTE=1

export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/usr/local/hyhal/lib64:/host-rdma:$LD_LIBRARY_PATH

MODEL_PATH="/home/hy3/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3"
MODEL_NAME="$(basename "$MODEL_PATH")"
LOG_TIME="$(date +%Y%m%d_%H%M%S)"
mkdir -p log
LOG_FILE="log/tmp-${MODEL_NAME}-sp8ep8-p-${LOG_TIME}.log"

VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=0 \
vllm serve $MODEL_PATH \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --enable-custom-sp \
  --enforce-eager \
  --kv_cache_dtype fp8_e4m3 \
  --port 8010 \
  --enable-expert-parallel \
  --all2all_backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
  2>&1 | tee "$LOG_FILE"

  #Prefill 节点 2
  export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export HIPBLASLT_TUNING_OVERRIDE_FILE=/home/hy3/rocblas/hipblaslt.config
export ROCBLAS_TENSILE_LIBPATH=/home/hy3/rocblas/rocblas_hy3_fp8_zmy
export GPU_MAX_HW_QUEUES=4

export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

export ROCSHMEM_HEAP_SIZE=4000000000
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024

export ROCSHMEM_IB_GID_INDEX=0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

export HIP_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8999
export ALLREDUCE_STREAM_WITH_COMPUTE=1

export LD_LIBRARY_PATH=/usr/local/hyhal/lib:/usr/local/hyhal/lib64:/host-rdma:$LD_LIBRARY_PATH

MODEL_PATH="/home/hy3/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3"
MODEL_NAME="$(basename "$MODEL_PATH")"
LOG_TIME="$(date +%Y%m%d_%H%M%S)"
mkdir -p log
LOG_FILE="log/tmp-${MODEL_NAME}-sp8ep8-p-${LOG_TIME}.log"

VLLM_HCU_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=0 \
vllm serve $MODEL_PATH \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --enable-custom-sp \
  --enforce-eager \
  --kv_cache_dtype fp8_e4m3 \
  --port 8011 \
  --enable-expert-parallel \
  --all2all_backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
  2>&1 | tee "$LOG_FILE"

  #D节点
  #参考 P：SP8  D：DP16EP16 (1P1D)

  #代理服务器
  python3 /workspace/vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://xxx:8010" "8998" \  
    --prefill "http://xxx" "8999" \
    --decode "http://xxx:8020" \  
    --port 8000
```

### vllm环境变量

```bash
#vllm报传输失败可以尝试增加超时。
export VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1200
#增加 worker number
--kv-transfer-config '{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "num_workers": 32
  }
}'
```

## SGLang HiCache with Mooncake Backend

官方文档：https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/index.html

在启动服务前，必须清晰掌握节点内的计算、网络与内存布局。

### 内存与拓扑配置

```bash
# 查看显存容量及状态
hy-smi --showmeminfo vram

# 查看 HCU 间拓扑连接
hy-smi --showtopo

# 查看系统内存与共享内存现状
free -h
df -h /dev/shm
cat /proc/meminfo

# 查看物理硬件层级拓扑
lstopo

# 查看 NUMA 节点状态
numactl -H
numastat -m
```

### 释放页缓存

当 NUMA 本地节点内存耗尽、RDMA pinned memory 分配失败或 OOM 被触发时，可手动释放缓存：

```bash
sync
echo 3 > /proc/sys/vm/drop_caches
```

注意：执行后短期 I/O 性能会下降。建议在启动超大型任务前或出现 OOM 时使用，不要频繁执行。

### Mooncake Master 配置

Mooncake 作为 L3 Cache 后端，依赖元数据服务器进行节点发现与 Cache 寻址，负责协调整个集群的逻辑存储空间池，管理 L3 KV 缓存空间分配与淘汰。

```bash
mooncake_master --enable_http_metadata_server=true \
    --http_metadata_server_host=<your_node_ip> \
    --http_metadata_server_port=8080
```

查看网卡绑定的 NUMA 节点：

```bash
cat /sys/class/infiniband/shca_0/device/numa_node
```

### 跨节点内存交错

当发现 numastat -m 中节点负载极度不均（如 Node 0 剩余 1G，Node 2 剩余 50G）时，必须打破局部性限制，允许内存分配散布到全节点。跨节点访问可能会略有性能下降。

```bash
numactl --interleave=all python3 -m sglang.launch_server <args...>
```

### 动态扩容共享内存

无需重启系统即可即时调整shmem容量限制。

```bash
sudo mount -o remount,size=400G /dev/shm
```

### HiCache CPU 内存使用

使用 HiCache 时，默认情况下，L2 主机 DRAM（CPU 内存）中用于 KV 缓存的大小是 L1 设备内存（GPU 内存）中 KV 缓存大小的 2 倍。
如果模型较小但 GPU 内存很大，特别是在多 TP（tensor parallel）场景下，这可能导致 L1 KV 缓存变得很大，继而消耗过多 CPU DRAM。
在这种情况下，应根据你的硬件手动配置合适的 L2 缓存大小。
可以通过设置 --hicache-ratio 或 --hicache-size 实现。

## LMCache with Mooncake Backend

官方文档：https://docs.lmcache.ai/kv_cache/storage_backends/mooncake.html

### 启动 Mooncake Store

```bash
# 警告：下面的 mkfs.ext4 会清空整块磁盘数据，请先用 lsblk/blkid 确认设备名，
# 不要误用系统盘或已有数据的磁盘；通常需要 sudo 权限。
lsblk

# 格式化并挂载本地盘（请将 /dev/<your-nvme-device> 替换为已确认无误的目标设备）
sudo mkfs.ext4 /dev/<your-nvme-device>
sudo mount /dev/<your-nvme-device> /mnt/mooncake

# 启动 mooncake master
nohup mooncake_master --enable_http_metadata_server=true \
    --rpc_port=50051 \
    --enable_offload=true > mooncake.log 2>&1 &
```

支持三种后端模式：
- `bucket_storage_backend`：多个对象文件一起写在文件桶里
- `file_per_key_storage_backend`：每个对象一个文件
- `offset_allocator_storage_backend`：所有对象写在一个大文件中，不支持进程重启后缓存复用

```bash
export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=/mnt/mooncake
export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=bucket_storage_backend

mooncake_client \
    --master_server_address=127.0.0.1:50051 \
    --host=10.17.95.253 \
    --protocol="tcp" \
    --device_names=enp97s0f0 \
    --port=50052 \
    --global_segment_size="5368709120" \
    --enable_offload=true \
    --metadata_server="P2PHANDSHAKE"
```

### LMCache 配置文件

```yaml
# lmcache_config.yaml
local_cpu: False
remote_url: "mooncakestore://127.0.0.1:50051/"
max_local_cpu_size: 10
numa_mode: "auto"
pre_caching_hash_algorithm: sha256_cbor_64bit

extra_config:
  use_exists_sync: true
  save_chunk_meta: False
  local_hostname: "10.17.95.253"
  metadata_server: "P2PHANDSHAKE"
  protocol: "tcp"
  device_name: ""
  global_segment_size: 5368709120
  master_server_address: "127.0.0.1:50051"
  local_buffer_size: 0
  mooncake_prefer_local_alloc: true
```

### 启动 vLLM

```bash
export VLLM_LOGGING_LEVEL=INFO
export LMCACHE_LOG_LEVEL=INFO
export PYTHONHASHSEED=0
export LMCACHE_CONFIG_FILE=lmcache_config.yaml

nohup vllm serve /llm/models/Qwen3-32B \
    --max-log-len 64 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --disable-log-requests \
    --pipeline-parallel-size 1 \
    --served-model-name "Qwen3-32B" \
    > vllm_running.log 2>&1 &
```

## TransferEngine API

### 创建与初始化

```python
from mooncake.engine import TransferEngine

engine = TransferEngine()
engine.initialize(
    "localhost",       # 本机地址（必须唯一）
    "P2PHANDSHAKE",    # metadata server
    "rdma",            # 协议：rdma / tcp
    ""                 # device（可选，空串表示使用所有设备）
)
```

### 注册内存

```python
import numpy as np

buf = np.ones(1024, dtype=np.uint8)
addr = buf.ctypes.data
size = buf.nbytes

engine.register_memory(addr, size)
```

### 数据传输（同步写）

```python
ret = engine.transfer_sync_write(
    target_hostname,   # 目标服务器主机名
    local_buffer_addr, # 本地缓冲区地址
    remote_buffer_addr,# 远程缓冲区地址
    length             # 传输字节数
)
```

### 注销内存

```python
ret = engine.unregister_memory(addr)
```

### 示例：跨节点传输

<details>
<summary>Server 端代码</summary>

```python
import zmq
import torch
from mooncake.engine import TransferEngine

def main():
    torch.cuda.set_device(0)
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)
    send_socket.bind("tcp://*:5555")
    recv_socket = context.socket(zmq.PULL)
    recv_socket.bind("tcp://*:5556")

    HOSTNAME = "localhost"
    METADATA_SERVER = "P2PHANDSHAKE"
    PROTOCOL = "rdma"
    DEVICE_NAME = ""

    server_engine = TransferEngine()
    server_engine.initialize(HOSTNAME, METADATA_SERVER, PROTOCOL, DEVICE_NAME)
    session_id = f"{HOSTNAME}:{server_engine.get_rpc_port()}"

    server_buffer = torch.full((1024 * 1024,), 77, dtype=torch.uint8, device="cuda:0")
    server_ptr = server_buffer.data_ptr()
    server_len = server_buffer.nbytes

    torch.cuda.synchronize(0)
    server_engine.register_memory(server_ptr, server_len)

    buffer_info = {"session_id": session_id, "ptr": server_ptr, "len": server_len}
    send_socket.send_json(buffer_info)

    transfer_status = recv_socket.recv_json()
    if transfer_status.get("status") == "transfer_complete":
        print("Data transfer completed.")

    expect_val = 92
    is_correct = torch.all(server_buffer == expect_val).item()
    print("Data verification successful!" if is_correct else "Data verification failed!")

    server_engine.unregister_memory(server_ptr)
    send_socket.close()
    recv_socket.close()
    context.term()

if __name__ == "__main__":
    main()
```
</details>

<details>
<summary>Client 端代码</summary>

```python
import torch
import zmq
from mooncake.engine import TransferEngine

def main():
    torch.cuda.set_device(1)
    context = zmq.Context()
    recv_socket = context.socket(zmq.PULL)
    recv_socket.connect("tcp://localhost:5555")
    send_socket = context.socket(zmq.PUSH)
    send_socket.connect("tcp://localhost:5556")

    buffer_info = recv_socket.recv_json()
    server_session_id = buffer_info["session_id"]
    server_ptr = buffer_info["ptr"]
    server_len = buffer_info["len"]

    HOSTNAME = "localhost"
    METADATA_SERVER = "P2PHANDSHAKE"
    PROTOCOL = "rdma"
    DEVICE_NAME = ""

    client_engine = TransferEngine()
    client_engine.initialize(HOSTNAME, METADATA_SERVER, PROTOCOL, DEVICE_NAME)

    client_buffer = torch.full((1024 * 1024,), 92, dtype=torch.uint8, device="cuda:1")
    client_ptr = client_buffer.data_ptr()
    client_len = client_buffer.nbytes

    torch.cuda.synchronize(1)
    client_engine.register_memory(client_ptr, client_len)

    client_engine.transfer_sync_write(
        server_session_id, client_ptr, server_ptr, min(client_len, server_len)
    )

    send_socket.send_json({"status": "transfer_complete"})
    client_engine.unregister_memory(client_ptr)
    send_socket.close()
    recv_socket.close()
    context.term()

if __name__ == "__main__":
    main()
```
</details>
