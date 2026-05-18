# MiniMax-2.x on vLLM

## 模型简介

MiniMax-2.x 是 MiniMax 推出的大规模 MoE（混合专家）语言模型系列，总参数量 229B，激活参数约 10B，在长文本理解和生成方面表现突出。
支持模型MiniMax-2.5和MiniMax-2.7，MiniMax-2.7复用MiniMax-2.5的运行命令。

## 模型列表

|                                               模型权重                                               |       量化方式       | 总参数 | 激活参数 | vLLM版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令                                                      |
| :--------------------------------------------------------------------------------------------------: | :------------------: | ------ | -------- | -------- | :------: | :--: | :------: | ------------------------------------------------------------- |
| [MiniMax-M2.5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-INT8-w8a8) | INT8  CHANNEL_W8A8 | 229B   | ~10B     | 0.15.1   |  BW1100  |  8  |   IFB   | [对应命令](#MiniMax-2.x-w8a8-channel_wise-int8(8x_BW1100_144GB)) |
| [MiniMax-M2.5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-INT8-w8a8) | INT8  CHANNEL_W8A8 | 229B   | ~10B     | 0.15.1   |  BW1000  |  8  |   IFB   | [对应命令](#MiniMax-2.x-w8a8-channel_wise-int8(8x_BW1000_64GB))  |
|  [MiniMax-M2.5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-Channel-FP8-w8a8)  | FP8   CHANNEL_W8A8 | 229B   | ~10B     | 0.15.1   |  BW1100  |  8  |   IFB   | [对应命令](#MiniMax-2.x-w8a8-channel_wise-fp8(8x_BW1100_144GB))  |
|              [MiniMax-M2.5-bf16](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-bf16)              |         BF16         | 229B   | ~10B     | 0.15.1   |  BW1100  |  8  |   IFB   | [对应命令](#MiniMax-2.x-bf16(8x_BW1100_144GB))                   |
|              [MiniMax-M2.5-bf16](https://www.modelscope.cn/models/hygon/MiniMax-M2.5-bf16)              |         BF16         | 229B   | ~10B     | 0.15.1   |  BW1000  |  8  |   IFB   | [对应命令](#MiniMax-2.x-bf16(8x_BW1000_64GB))                    |

## 启动命令

### MiniMax-2.x-w8a8-channel_wise-int8(8x_BW1100_144GB)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1


export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1
export USE_FUSED_SILU_MUL_QUANT=1
export USE_FUSED_RMS_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1 


export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_OPT_RESHAPE_AND_CACHE=1 
export VLLM_USE_GLOBAL_CACHE13=1 
export VLLM_FUSED_MOE_CHUNK_SIZE=16384  
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_USE_OPT_OP=1 
export VLLM_USE_AITER_MOE_W8A8=0


vllm serve /hygon/MiniMax-M2.5-W8A8 \
 --host 0.0.0.0 \
 --trust-remote-code \
 -tp 4 -pp 2 \
 --gpu-memory-utilization 0.92 \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 -q slimquant_marlin \
 --kv-cache-dtype fp8_e4m3 \
 --enable-prefix-caching \
 --disable-cascade-attn
```

### MiniMax-2.x-w8a8-channel_wise-int8(8x_BW1000_64GB)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1

export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1
export USE_FUSED_SILU_MUL_QUANT=1
export USE_FUSED_RMS_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1

export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_OPT_RESHAPE_AND_CACHE=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_USE_OPT_OP=1
export VLLM_USE_AITER_MOE_W8A8=0

vllm serve /hygon/MiniMax-M2.5-W8A8 \
 --host 0.0.0.0 \
 --trust-remote-code \
 -tp 8 \
 --gpu-memory-utilization 0.92 \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 -q slimquant_marlin \
 --enable-prefix-caching \
 --disable-cascade-attn 
```
### MiniMax-2.x-w8a8-channel_wise-fp8(8x_BW1100_144GB)
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1

export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_OPT_RESHAPE_AND_CACHE=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_USE_OPT_OP=1
export VLLM_USE_AITER_MOE_W8A8=0


vllm serve /hygon/MiniMax-M2.5-Channel-FP8-w8a8 \
 --host 0.0.0.0 \
 --trust-remote-code \
 -tp 4 -pp 2 \
 --gpu-memory-utilization 0.92 \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 --kv-cache-dtype fp8_e4m3 \
 --enable-prefix-caching \
 --disable-cascade-attn \
 -q slimquant_marlin 

```
### MiniMax-2.x-bf16(8x_BW1100_144GB)
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1

export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_USE_OPT_OP=1
export VLLM_USE_AITER_MOE_W8A8=0


vllm serve /hygon/MiniMax-M2.5-bf16 \
 --host 0.0.0.0 \
 --trust-remote-code \
 -tp 8 \
 --gpu-memory-utilization 0.92 \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 --enable-prefix-caching \
 --kv-cache-dtype fp8_e4m3 \
 --disable-cascade-attn
```
### MiniMax-2.x-bf16(8x_BW1000_64GB)
```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1
export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1

export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_USE_OPT_OP=1
export VLLM_USE_AITER_MOE_W8A8=0


vllm serve /hygon/MiniMax-M2.5-bf16 \
 --host 0.0.0.0 \
 --trust-remote-code \
 -tp 8 \
 --gpu-memory-utilization 0.92 \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 --enable-prefix-caching \
 --disable-cascade-attn
```


### 显存不足时

```bash
# 降低上下文长度
--max-model-len 8192

# 启用 KV Cache 量化
--kv-cache-dtype fp8_e4m3

# 降低显存利用率
--gpu-memory-utilization 0.9
```

## API 调用

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="/hygon/MiniMax-M2.5-W8A8",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "请详细分析大模型在金融领域的应用前景"},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

## curl 调用

```python

curl -X POST http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "/hygon/MiniMax-M2.5-W8A8",
    "messages": [
        {"role": "user", "content": "中国的首都是哪里？"}
    ],
    "max_tokens": 1024
}'

```

## DCU 适配注意

- MoE 架构：总参数 229B，但每次推理只激活约 10B，实际显存需求低于 dense 模型
- 需要 `--trust-remote-code`
- 建议使用 8x BW1100 144GB（1024GB 总显存）
- 长上下文场景 KV Cache 占用大，MoE 模型尤为明显
- 如果遇到 OOM，优先降低 `--max-model-len` 或启用 `--kv-cache-dtype fp8_e4m3`或降低显存利用率`--gpu-memory-utilization`
