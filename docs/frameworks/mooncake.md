# Mooncake on DCU

> 本文档涵盖 Mooncake 在 DCU 环境下的安装、配置、环境变量说明以及各框架（SGLang / vLLM）的 PD 分离部署和 Mooncake Store 集成方案。

- [简介](#简介)
- [安装](#安装)
- [环境变量](#环境变量)
  - [LOG](#log)
  - [RDMA](#rdma)
  - [P2P](#p2p)
- [Mooncake bench 测试](#mooncake-bench-测试)
- [SGLang PD 分离](#sglang-pd-分离)
  - [SGLang 单节点 1P1D 测试](#SGLang-单节点-1P1D-测试)
  - [SGLang 双节点 1P1D 测试](#SGLang-双节点-1P1D-测试)
- [vLLM PD 分离](#vllm-pd-分离)
  - [vLLM 单节点 1P1D 测试](#vLLM-单节点-1P1D-测试)
  - [vLLM 双节点 1P1D 测试](#vLLM-双节点-1P1D-测试)
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

- **多协议传输**：支持 RDMA、TCP 等多种传输协议，在 DCU 环境下还可启用 HIP IPC（节点内）和 HIP RPC（跨节点）传输，灵活适配不同网络硬件
- **多级缓存池**：在 GPU 显存（L1）、主机 DRAM（L2）、SSD（L3）上构建分层缓存，平衡推理速度与容量
- **即插即用**：已深度集成至 SGLang 和 vLLM，通过 `--kv-transfer-config` 或 `--disaggregation-mode` 即可启用 PD 分离
- **缓存后端**：可作为 LMCache 的远端存储后端（Mooncake Store），支持内存卸载与 SSD 卸载

官方文档：<https://kvcache-ai.github.io/Mooncake/>

## 安装

```bash
# pip 安装 whl 包
# 普通网卡版本，不带 hylink 支持
http://pypi.sourcefind.cn:666/das_nightly/dtk2604-rc4-mooncake/+f/79c/add379d74452d/mooncake_transfer_engine-0.3.10.post1+das.opt1.dtk2604.2605131137.gd34f6f-cp310-cp310-manylinux_2_35_x86_64.whl
pip install mooncake_transfer_engine*.whl

# 普通网卡版本，带 hylink 支持的版本
http://pypi.sourcefind.cn:666/das_nightly/dtk2604-rc4-mooncake/+f/4aa/b1d2c2d1653e9/mooncake_transfer_engine_rpc-0.3.10.post1+das.opt1.dtk2604.2605131408.gd34f6f-cp310-cp310-manylinux_2_35_x86_64.whl
pip install mooncake_transfer_engine_rpc*.whl

# 天龙网卡版本，不带 hylink 支持
http://pypi.sourcefind.cn:666/das_nightly/dtk2604-rc4-mooncake/+f/2e2/14988dbb22475/mooncake_transfer_engine_shca-0.3.10.post1+das.opt1.dtk2604.2605131044.gd34f6f-cp310-cp310-manylinux_2_35_x86_64.whl
pip install mooncake_transfer_engine_shca*.whl
```

## 环境变量

### LOG

```bash
# 可选：TRACE / INFO / WARNING / ERROR，默认 INFO
export MC_LOG_LEVEL=TRACE
```

### RDMA

```bash
# 握手失败时可通过设置 host ip 解决
export SGLANG_HOST_IP=${HOST_IP}
export VLLM_HOST_IP=${HOST_IP}

# 存在跨 SM IB NIC transfer 问题时，启用设备亲和性
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

# 同一交换机内通信可以尝试切换 GID（默认 3=global，0=link-local）
export MC_IB_GID_INDEX=0
```

### P2P

```bash
# 启用 HIP IPC Transport（仅用于节点内通信）
export MC_FOCRE_MNNVL=1

# 启用 HIP RPC Transport（用于节点内和节点间通信，需安装 mooncake_transfer_engine_rpc）
export MC_FOCRE_MNNVL=1
export MC_USE_HIP_IPC=0

# 修改 copy kernel cu number
export MC_HIP_COPY_BLOCKS=xxx
```

## Mooncake bench 测试

`transfer_engine_bench` 在 `pip show mooncake-transfer-engine` 显示的安装路径下。

```bash
# Node 1（target）
# 请将 `<local_host_ip>` 替换为当前节点 IP
transfer_engine_bench --mode=target --auto_discovery --protocol=rdma \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --local_server_name=<local_host_ip>

# Node 2（initiator）
# 请将 `<local_host_ip>` 替换为当前节点 IP，`<remote_host_ip>` 替换为对端节点 IP
# `<port>` 替换为从 Node 1 日志 `Transfer Engine RPC using XXX, listening on YYY:ZZZ` 中获取的端口
transfer_engine_bench --mode=initiator --auto_discovery --protocol=rdma \
    --metadata_server=P2PHANDSHAKE --gpu_id=-1 \
    --local_server_name=<local_host_ip> --segment_id=<remote_host_ip>:<port>
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

### vLLM 单节点 1P1D 测试

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
python3 vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://0.0.0.0:8010" "8998" \
    --decode "http://0.0.0.0:8020" \
    --port 8000
```

### vLLM 双节点 1P1D 测试

```bash
# KV Producer（Prefill）
vllm serve Qwen3/Qwen3-8B \
    --port 8010 \
    --data-parallel-size 2 --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'

# KV Consumer（Decode）
vllm serve Qwen3/Qwen3-8B \
    --port 8020 \
    --data-parallel-size 2 --tensor-parallel-size 4 \
    --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'

# Mooncake Connector Proxy
python3 vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py \
    --prefill "http://10.63.60.113:8010" "8998" \
    --decode "http://10.63.60.114:8020" \
    --port 8000
```

## SGLang HiCache with Mooncake Backend

官方文档：https://kvcache-ai.github.io/Mooncake/getting_started/examples/sglang-integration/index.html

在启动服务前，必须清晰掌握节点内的计算、网络与内存布局。

### 内存与拓扑配置

```bash
# 查看显存容量及状态
hy-smi --showmeminfo vram

# 查看 DCU 间拓扑连接
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
