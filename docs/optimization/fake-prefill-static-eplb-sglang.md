# Fake Prefill + Static EPLB 测试方法

> **场景**:在 SGLang 环境下,用 fake prefill 采集 expert 分布,离线生成 static EPLB placement,然后用同一份 fake prefill workload 测试 decode 性能的完整流程。
>
> **模型**:Hy3-CHANNEL-FP8(HunyuanVideo v3 系,MoE + NEXTN spec decode + PD disaggregation)
> **拓扑**:2 节点 × 8 卡 = 16 卡 EP,DP=16

---

## 一、目标与适用范围

**目标**:在完全对齐测试场景下(dump / test 同一份 fake prefill 分布),测试 Fake Prefill + static EPLB 的 decode 性能。

**适用范围**:
- ✅ 测试 Fake Prefill 开启 static EPLB 的 decode 吞吐 / TPOT / ITL 表现
- ✅ 验证 dump → load → apply 全流程在当前环境下可复现
- ⚠️ 数据仅反映 fake prefill 对齐测试场景下的性能,不直接等同于真实业务

---

## 二、完整流程

### Step 1:启动 dump 阶段的 server(采集 expert 分布)

**Node 0 :**
```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/eplb
export SGLANG_SIMULATE_ACC_LEN=2.7 #根据mtp接受率大小和mtp数灵活调整
export SGLANG_SIMULATE_ACC_METHOD=match-expected
# ... 其余环境变量见下方"通用环境变量"

python -m sglang.launch_server \
  --model-path /path/to/model \
  --dp-size 16 --tp-size 16 --ep-size 16 \
  --deepep-mode low_latency \
  --moe-a2a-backend deepep \
  --enable-dp-attention \
  --moe-dense-tp-size=1 \
  --enable-dp-lm-head \
  --trust-remote-code \
  --dist-init-addr <NODE0_IP>:30009 \
  --host <NODE0_IP> \
  --nnodes 2 --node-rank 0 \
  --dtype bfloat16 \
  --port 3009 \
  --attention-backend fa3 \
  --page-size 64 \
  --mem-fraction-static 0.85 \
  --context-length 8192 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 64 \
  --expert-distribution-recorder-mode stat \
  --enable-expert-distribution-metrics \
  --ep-num-redundant-experts 16 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 2 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 3 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend fake \
  --watchdog-timeout 36000
```

**Node 1 (<NODE1_IP>):** 改 `--host <NODE1_IP> --node-rank 1`,`SGLANG_HOST_IP` 和 `NCCL_SOCKET_IFNAME` 换成 node 1 的网卡。

### Step 2:客户端触发 dump 采样

Server 起来后,先启动 recorder,再发bench,最后 stop + dump:

```bash
# 开始记录
curl -X POST http://<NODE0_IP>:3009/start_expert_distribution_record

# 发送 fake prefill bench(参数需与后续 test 完全一致)
python3 -m sglang.bench_serving \
    --backend sglang \
    --model /path/to/model \
    --host <NODE0_IP> --port 3009 \
    --dataset-name random \
    --random-input-len 3155 \
    --random-output-len 1024 \
    --random-range-ratio 1.0  \      

# 停止并 dump
curl -X POST http://<NODE0_IP>:3009/stop_expert_distribution_record
curl -X POST http://<NODE0_IP>:3009/dump_expert_distribution_record
```

Dump 完成后,`/path/to/eplb` 下会生成 `expert_distribution_recorder_<timestamp>.pt` 文件。

### Step 3:启动 test server

**关键改动**(相对 dump 脚本):

```bash
# 去掉
- --expert-distribution-recorder-mode stat

# 加上
+ --init-expert-location /path/to/expert_distribution_recorder_XXX.pt
+ --ep-dispatch-algorithm static
+ --eplb-algorithm auto
```

**保留**:
- `--ep-num-redundant-experts 16`(冗余专家数量，根据实际ep并行策略灵活调整)
- `--enable-expert-distribution-metrics`(可选,用来验证 EPLB 生效)
- 其他所有参数与 dump 完全一致

---

## 三、关键参数说明

### Dump 阶段必须固定的项(dump / test 双侧一致)

| 项 | 值 | 为什么必须一致 |
|---|---|---|
| `--dataset-name` | `random` | dump/test保持一致 |
| `--random-range-ratio` | `1.0` | 否则 sglang 会在区间内随机采样长度 |
| `--random-input-len / output-len` | 固定 | 长度变了 routing 分布就变 |
| `--num-prompts / max-concurrency` | 固定 | batch 变化会改变每步的 token 数分布 |

---

## 四、通用环境变量(两个 node 都需要)

```bash
# NCCL / ROCm
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_SOCKET_IFNAME=<node_specific_ifname>  # node 0: enp10s0f4u2c2, node 1: enp10s0f4u1
export GLOO_SOCKET_IFNAME=<same_as_above>
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPU_MAX_HW_QUEUES=3

# HIP 内存拷贝调优
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512

# SGLang MoE / kernel
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_FUSED_RMS_ROTARY=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=512

# ROCSHMEM
export ROCSHMEM_HEAP_SIZE=5368709120
export ROCSHMEM_MAX_NUM_CONTEXTS=64
export ROCSHMEM_GDR_DISABLE_XDP=1
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_ALLOWED_IBV_DEVICES=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_0,mlx5_1
export ROCSHMEM_TOPO_FILE_FORCE=/path/to/topo.config

# EPLB
export SGLANG_SIMULATE_ACC_LEN=2.7
export SGLANG_SIMULATE_ACC_METHOD=match-expected

# NUMA
sysctl -w kernel.numa_balancing=0
```
