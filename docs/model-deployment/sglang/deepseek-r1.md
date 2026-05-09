# DeepSeek-R1 on SGLang

## 模型简介

DeepSeek-R1 是 DeepSeek 推出的推理强化模型系列，面向复杂推理、数学与代码场景。SGLang 在 DCU 平台可通过张量并行部署 R1 系列模型，并兼容 OpenAI API 接口。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| DeepSeek-R1 | 671B (MoE) | 128K | 8x DCU 144GB TP |

## 启动命令

### DeepSeek-R1-Channel-FP8

#### IFB【tp8】
```bash
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_OPT_CAT=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_RMS_QUANT_PATH=1
export USE_FUSED_RMS_QUANT_PATH=1
export SGLANG_USE_FUSED_RMSNORM_ROPE=1
export SGLANG_TORCH_PROFILER_DIR=/workspace/prof
export SGLANG_SET_CPU_AFFINITY=1
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=4
export SGLANG_ENABLE_SPEC_V2=1

export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1

export HIP_H2D_DISABLE_COPY_BUFFER=0 # 同步异步强制走WriteBuffer
export HIP_D2H_DISABLE_COPY_BUFFER=0 # 同步异步强制走ReadBuffer
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768 # 小于此值走CPUCopy
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768 # 大于此值走HSACOPY（CopyBuffer）
export HIP_D2H_DIRECT_COPY_THRESHOLD=512 # 小于此值走CPUCopy
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512 # 大于此值走HSACOPY（CopyBuffer）

export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072

export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export USE_SPE_MQP=1
export MC_ALLOWED_IBV_DEVICES=mlx5_6,mlx5_7,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

python3 -m sglang.launch_server \
    --model-path hygon/DeepSeek-R1-Channel-FP8-w8a8 \
    --numa-node 0 0 0 0 1 1 1 1 \
    --disable-radix-cache \
    --chunked-prefill-size -1 \
    --max-running-requests 256 \
    --speculative-algorithm EAGLE --speculative-num-steps 2  --speculative-eagle-topk 1  --speculative-num-draft-tokens 1 \
    --context-length 65536 \
    --quantization w8a8_fp8 \
    --kv-cache-dtype fp8_e4m3 \
    --trust-remote-code \
    --nnodes 1 \
    --node-rank 0 \
    --dtype bfloat16 \
    --tp-size 8 \
    --pp-size 1 \
    --mem-fraction-static 0.9 \
    --attention-backend dcu_mla

```


## curl 调用

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Channel-FP8",
    "max_tokens": 1024,
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "请给出一个高并发服务限流方案。"}
    ]
  }'
```
