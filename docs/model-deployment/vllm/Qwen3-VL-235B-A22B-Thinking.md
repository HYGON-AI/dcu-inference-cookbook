# NMZ
```
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RPC_TIMEOUT=1800000
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0 #根据实际情况绑定
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

vllm serve /model/Qwen3-VL-235BB-A22B-Thinking  \
        --dtype float16 \
        --disable-cascade-attn \
        --tensor-parallel-size 8
```
# BW1000
```
export VLLM_HOST_IP=XX.XX.XX.XX
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000

vllm serve /model/Qwen3-VL-235BB-A22B-Thinking \
        --dtype float16 \
        --disable-cascade-attn \
        --kv-cache-dtype fp8_e5m2 
        --distributed-executor-backend ray \
        --tensor-parallel-size 16

```
# K100AI
```
export VLLM_HOST_IP=12.12.12.74
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_IB_HCA=mlx5_0:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000

vllm serve /model/Qwen3-VL-235BB-A22B-Thinking \
        --dtype float16 \
        --disable-cascade-attn \
        --distributed-executor-backend ray \
        --tensor-parallel-size 16
```
