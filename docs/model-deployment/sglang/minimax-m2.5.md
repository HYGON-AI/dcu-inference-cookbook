# MiniMax-2.5 on SGLang

## 模型简介

MiniMax-M2.5-Channel-FP8-w8a8 是 MiniMax 推出的大规模 MoE（混合专家）语言模型系列，总参数量 230B，激活参数约 10B，在长文本理解和生成方面表现突出。



## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [MiniMax-M2-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-int8-w8a8-ifb-bw1100-8x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-int8-w8a8-1p1d-bw1100-16x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1000 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-int8-w8a8-ifb-bw1000-8x) |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1000 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-int8-w8a8-1p1d-bw1000-16x) |
| [MiniMax-M2-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-FP8-w8a8) | FP8 W8A8 | 0.5.10 | BW1100 | 8 | IFB | [**`>_`**](#minimax-m2-5-channel-fp8-w8a8-ifb-bw1100-8x) |
|                                                                                                 | FP8 W8A8 | 0.5.10 | BW1100 | 16 | 1P1D| [**`>_`**](#minimax-m2-5-channel-fp8-w8a8-1p1d-bw1100-16x) |


## 启动命令

### MiniMax-M2-5-Channel-INT8-w8a8 IFB BW1100 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --dp-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2-5-Channel-INT8-w8a8 IFB BW1000 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.95  \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2-5-Channel-FP8-w8a8 IFB BW1100 8x

网卡配置参考：[mlx]

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \
  --quantization w8a8_fp8 \
  --page-size 64 \
  --dtype bfloat16 \
  --tp-size 4 --pp-size 1 --dp-size 2 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.92 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 8192 \
  --max-running-requests 512 \
  --context-length 131072 \
  --port 30000
```

### MiniMax-M2-5-Channel-INT8-w8a8 1P1D BW1100 16x

网卡配置参考：[mlx]

#### P node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \ 
  --dtype bfloat16 \ 
  --tp-size 2 --pp-size 4 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --chunked-prefill-size 4096 \
  --max-running-requests 512 \ 
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 10.63.60.114 --port 30000 
```

#### D node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
 
sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think\
  --mem-fraction-static 0.9 --attention-backend fa3 \ 
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.63.60.113 --port 30001 
```

### MiniMax-M2-5-Channel-INT8-w8a8 1P1D BW1000 16x

网卡配置参考：[shca]

#### P node 0

```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export NCCL_TOPO_FILE="/home/built-in-508-topo-input-tj-default.xml"
export LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \ 
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \ 
  --host 10.212.16.171 --port 30000
```

#### D node 0
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export MC_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export NCCL_TOPO_FILE="/home/topo_lib/built-in-508-topo-input-tj-default.xml"
export LD_LIBRARY_PATH=/home/topo_lib/lib:$LD_LIBRARY_PATH
export ROCSHMEM_DISABLE_HDP_FLUSH=1
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_HEAP_SIZE=3173741824
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export LMSLIM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-INT8-w8a8 \
  --quantization slimquant_marlin \ 
  --kv-cache-dtype bfloat16 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \
  --chunked-prefill-size 16384 \
  --max-running-requests 512 \
  --context-length 131027 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.212.16.172 --port 30001 \
  --disaggregation-ib-device shca_0,shca_1,shca_2,shca_3
```

### MiniMax-M2-5-Channel-FP8-w8a8 1P1D BW1100 16x

网卡配置参考：[mlx]

#### P节点 
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1

sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \ 
  --dtype bfloat16 \ 
  --tp-size 2 --pp-size 4 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think \
  --mem-fraction-static 0.9 \ 
  --attention-backend fa3 \
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --chunked-prefill-size 4096 \
  --max-running-requests 512 \ 
  --context-length 131072 \
  --disaggregation-mode prefill \
  --load-balance-method round_robin \
  --host 10.63.60.114 --port 30000 
```

#### D节点
```bash
export SGLANG_USE_MODELSCOPE=1
export USE_DCU_CUSTOM_ALLREDUCE=1
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000
export SGLANG_USE_LIGHTOP=1 
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export LMSLIM_USE_LIGHTOP=1
export SGLANG_KVALLOC_KERNEL=1
export SGLANG_CREATE_EXTEND_AFTER_DECODE_SPEC_INFO=1
export SGLANG_ASSIGN_EXTEND_CACHE_LOCS=1
export SGLANG_ASSIGN_REQ_TO_TOKEN_POOL=1
export SGLANG_GET_LAST_LOC=1
export SGLANG_CREATE_FLASHMLA_KV_INDICES_TRITON=1
export SGLANG_CREATE_CHUNKED_PREFIX_CACHE_KV_INDICES=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export MC_GID_INDEX=0
export ALLREDUCE_STREAM_WITH_COMPUTE=1
 
sglang serve \
  --model-path hygon/MiniMax-M2.5-Channel-FP8-w8a8/ \
  --quantization w8a8_fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --trust-remote-code \ 
  --page-size 64 \
  --dtype bfloat16 \ 
  --tp-size 8 --pp-size 1 --dp-size 1 \
  --tool-call-parser minimax-m2 \
  --reasoning-parser minimax-append-think\
  --mem-fraction-static 0.9 --attention-backend fa3 \ 
  --numa-node 0 0 0 0 1 1 1 1 \ 
  --max-running-requests 512 \
  --context-length 131072 \
  --disaggregation-mode decode \
  --prefill-round-robin-balance \ 
  --host 10.63.60.113 --port 30001 
```

## Router
```python
python3 -m sglang_router.launch_router \
--pd-disaggregation \
--prefill http://10.63.60.114:30000 \
--decode http://10.63.60.113:30001 \
--policy cache_aware --port 30002
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiniMax-M2.5-Channel-FP8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/MiniMax-M2.5-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 默认端口为 `30002`，若与 P node 0 部署在同一机器上需指定其他端口。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<router_ip>:30002/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/MiniMax-M2.5-Channel-FP8-w8a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://<router_ip>:30002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hygon/MiniMax-M2.5-Channel-FP8-w8a8",
    "messages": [
      {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
      {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 128
  }'
```

