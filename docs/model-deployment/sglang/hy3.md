# Hy3

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/Hy3-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-INT8-w8a8) | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB(tp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB (tp8dp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1100-8x-sglang-0512-tp8dp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB(tp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1000-8x-sglang-0512-tp8) |
|                                                                                                 | INT8 W8A8 | [0.5.12](../docker_images.md) | BW1000 | 8 | IFB(tp8dp8) | [**`>_`**](#hy3-channel-int8-w8a8-ifb-bw1000-8x-sglang-0512-tp8dp8) |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | [0.5.12](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#hy3-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0512-tp8) |

## 启动命令

### Hy3-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
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
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8dp8)

~~~bash
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
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.12 (tp8)

~~~bash
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
~~~

### Hy3-Channel-INT8-w8a8 IFB BW1000 8x SGLang 0.5.12 (tp8dp8)

~~~bash
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
~~~

### Hy3-Channel-FP8-w8a8 IFB BW1100 8x SGLang 0.5.12 (tp8)

~~~bash
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
~~~

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-INT8-w8a8",
    messages=[{"role": "user", "content": "中国的首都是哪里？"}],
    max_tokens=128,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/Hy3-Channel-INT8-w8a8", "messages": [{"role": "user", "content": "中国的首都是哪里？"}], "max_tokens": 128}'
```
