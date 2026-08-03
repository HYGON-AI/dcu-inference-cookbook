# DeepSeek-V3.2 on SGLang

## 模型简介

DeepSeek-V3.2 是 DeepSeek V3 系列的 MoE 大模型版本，面向高吞吐对话、复杂推理与代码生成场景。基于 SGLang 在 HCU 平台可通过 IFB 或 PD 分离模式部署，并兼容 OpenAI API。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/DeepSeek-V3.2-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3.2-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.12 | BW1100 | 8x | IFB | [**\`>_\`**](#deepseek-v32-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512) |
| [hygon/DeepSeek-V3.2-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-V3.2-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.12 | BW1100 | 8x | IFB | [**`>_`**](#deepseek-v32-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512) |

## 启动命令

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER_AR=0
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1

export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1

export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
sysctl -w kernel.numa_balancing=0

export SGLANG_ENABLE_SPEC_V2=1
#triton改写算子
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

#export HIP_GRAPH_ACCUMULATE_DISPATCH=0 #torchprof需要
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
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

#int8
export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=0 #该环境变量导致精度异常
export SGLANG_USE_FUSED_RMS_QUANT=1
export SGLANG_USE_FUSED_SILU_MUL_QUANT=1 #开启的精度正常


model_path=/mnt/vllm-w8a8-models/DeepSeek-V3.2-Channel-INT8

model=${model_path##*/}
tp=8
pp=1
nodes=1
rank=0

host_ip=$(hostname -I | awk '{print $1}')
master_ip=$host_ip
max_model_len=40960
gpu_mem=0.9
port=8328
port1=5328



python3 -m sglang.launch_server \
 --model-path $model_path  \
 --context-length $max_model_len \
 --dtype bfloat16 \
 --host $host_ip  --port $port --dist-init-addr $master_ip:$port1 \
 --mem-fraction-static $gpu_mem \
 --trust-remote-code \
 --nnodes $nodes --node-rank $rank \
 --tp-size $tp --pp-size $pp \
 --page-size 64 \
 --cuda-graph-max-bs 32 \
 --moe-runner-backend triton \
 --quantization slimquant_marlin \
 --kv-cache-dtype fp8_e4m3 \
 --attention-backend nsa \
 --nsa-prefill-backend flashmla_auto \
 --nsa-decode-backend flashmla_kv\
 --tool-call-parser deepseekv32\
 --reasoning-parser deepseek-v3

```

### DeepSeek-V3.2-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12
```bash

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1

export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
sysctl -w kernel.numa_balancing=0

export SGLANG_ENABLE_SPEC_V2=1
#triton改写算子
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
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

# #int8
# export SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD=1 #w8a8
# export SGLANG_USE_FUSED_RMS_QUANT=1
# export SGLANG_USE_FUSED_SILU_MUL_QUANT=1


model_path=/mnt/deepseek-v3.2/DeepSeek-V3.2-channel-fp8


model=${model_path##*/}
tp=8
pp=1
nodes=1
rank=0

host_ip=$(hostname -I | awk '{print $1}')
master_ip=$host_ip
max_model_len=40960
gpu_mem=0.9
port=8328
port1=5328

#--quantization slimquant_marlin \
# --disable-cuda-graph \

python3 -m sglang.launch_server \
 --model-path $model_path  \
 --context-length $max_model_len \
 --dtype bfloat16 \
 --host $host_ip  --port $port --dist-init-addr $master_ip:$port1 \
 --mem-fraction-static $gpu_mem \
 --trust-remote-code \
 --nnodes $nodes --node-rank $rank \
 --tp-size $tp --pp-size $pp \
 --page-size 64 \
 --cuda-graph-max-bs 32 \
 --quantization w8a8_fp8 \
 --kv-cache-dtype fp8_e4m3 \
 --attention-backend nsa \
 --nsa-prefill-backend flashmla_auto \
 --nsa-decode-backend flashmla_kv\
 --tool-call-parser deepseekv32\
 --reasoning-parser deepseek-v3

 ```

## API 调用

### DeepSeek-V3.2-Channel-INT8-w8a8 IFB

```bash
#!/usr/bin/env bash
set -euo pipefail

curl http://localhost:8328/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2-Channel-INT8",
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请给出一个高并发服务限流方案。"}
    ]
  }'

```

### DeepSeek-V3.2-Channel-FP8-w8a8 IFB

```bash
#!/usr/bin/env bash
set -euo pipefail

curl http://localhost:8328/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3.2-channel-fp8",
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请给出一个高并发服务限流方案。"}
    ]
  }'

```