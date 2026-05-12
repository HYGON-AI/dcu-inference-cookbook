# NMZ
```
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_RANK0_NUMA=0  #按照实际的
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3

model_path=/model/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.9

vllm serve ${model_path} \
 --dtype ${data_type} \
 --host 0.0.0.0 \
 --port ${port} \
 --trust-remote-code \
 -tp ${tp} \
 --distributed-executor-backend mp \
 --gpu-memory-utilization $gpu_memory \
 --disable-cascade-attn \
 --max-model-len 61440 \
--max-num-batched-tokens 204800 \
--kv-cache-dtype fp8_e4m3 \
```


# BW1000
```
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1 #按照实际
export VLLM_USE_PD_SPLIT=1
export VLLM_RANK0_NUMA=3
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=0
export VLLM_RANK4_NUMA=7
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=5
export VLLM_RANK7_NUMA=4

model_path=/model/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.9


vllm serve ${model_path} \
 --dtype ${data_type} \
 --host 0.0.0.0 \
 --port ${port} \
 --trust-remote-code \
 -tp ${tp} \
 --distributed-executor-backend mp \
 --gpu-memory-utilization $gpu_memory \
 --max-model-len 61440 \
 --kv-cache-dtype fp8_e5m2 \
 --disable-cascade-attn \
```

# K100
```
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_RANK0_NUMA=0 #按照实际
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=0
export VLLM_RANK3_NUMA=0
export VLLM_RANK4_NUMA=1
export VLLM_RANK5_NUMA=1
export VLLM_RANK6_NUMA=1
export VLLM_RANK7_NUMA=1 
export ALLREDUCE_STREAM_WITH_COMPUTE=1

model_path=/model/Qwen3-235B-A22B-Instruct-2507
data_type="float16"
tp=8
port=8507
gpu_memory=0.95

vllm serve ${model_path} \
 --dtype ${data_type} \
 --host 0.0.0.0 \
 --port ${port} \
 --trust-remote-code \
 -tp ${tp} \
 --distributed-executor-backend mp \
 --gpu-memory-utilization $gpu_memory \
 --disable-cascade-attn \
 --max-model-len 40960 \
```
