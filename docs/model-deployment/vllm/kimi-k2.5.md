# Kimi-K2.5 on vLLM

## 模型简介

Kimi-K2.5 是月之暗面（Moonshot AI）推出的新一代大语言模型，以超长上下文和强大的信息处理能力著称。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K2.5](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5) | INT4 W4A16 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k25-ifb-bw1100-8x-vllm-021) |
| [moonshotai/Kimi-K2.5](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5) | INT4 W4A16 | 0.18 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k25-ifb-bw1100-8x-vllm-015) |
|  | INT4 W4A16 | [0.18](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#kimi-k25-ifb-bw1000-16x-vllm-018) |
|  | INT4 W4A16 | [0.18](../docker_images.md) | K100_AI | 16 | IFB | [**`>_`**](#kimi-k25-ifb-k100_ai-16x-vllm-018) |
| [moonshotai/Kimi-K2.5](https://www.modelscope.cn/models/moonshotai/Kimi-K2.5) | INT4 W4A16 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#kimi-k25-ifb-bw1100-8x-vllm-015) |

## 启动命令

### Kimi-K2.5 IFB BW1100 8x vLLM 0.21

```bash
vllm serve moonshotai/Kimi-K2.5 \
    -tp 8 \
    --trust-remote-code   \
    --dtype bfloat16  \
    --max-model-len 65536  \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-backend aiter
```

### Kimi-K2.5 IFB BW1100 8x vLLM 0.18

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export VLLM_HCU_USE_AITER_W4A16_MOE=1

vllm serve moonshotai/Kimi-K2.5 \
    -tp 8 \
    --trust-remote-code   \
    --dtype bfloat16  \
    --max-model-len 65536  \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 16384 \
    --kv-cache-dtype fp8_e4m3
```

### Kimi-K2.5 IFB BW1000 16x vLLM 0.18

```bash
export VLLM_HOST_IP=$(hostname -I | awk '{print $2}')
export GLOO_SOCKET_IFNAME=ens61f0np0
export NCCL_SOCKET_IFNAME=ens61f0np0
export VLLM_NUMA_BIND=1
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_8:1,mlx5_9:1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_HCU_USE_AITER_W4A16_MOE=1

vllm serve moonshotai/Kimi-K2.5 \
  --trust-remote-code \
  --kv-cache-dtype fp8_e5m2 \
  -pp 2 \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384
```

### Kimi-K2.5 IFB K100_AI 16x vLLM 0.18

```bash
export VLLM_HOST_IP=$(hostname -I | awk '{print $2}')
export GLOO_SOCKET_IFNAME=ens61f0np0
export NCCL_SOCKET_IFNAME=ens61f0np0
export VLLM_NUMA_BIND=1
export NCCL_IB_HCA=mlx5_0:1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_HCU_USE_AITER_W4A16_MOE=1

vllm serve moonshotai/Kimi-K2.5 \
  -pp 2 \
  -tp 8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384
```

### Kimi-K2.5 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_USE_LIGHTOP=1
export VLLM_USE_PIECEWISE=1
export VLLM_1D_MROPE=1
export USE_FUSED_RMS_QUANT=0 
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1
export VLLM_W8A8_BACKEND=3
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_CAT_MLA=1 
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 
export VLLM_ROCM_USE_AITER_MOE=1

vllm serve moonshotai/Kimi-K2.5 \
  -tp 8 \
  --trust-remote-code   \
  --dtype bfloat16  \
  --max-model-len 65536  \
  --disable-log-requests  \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8_e4m3
```


## API 调用

### IFB

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json"  \
  -d '{
      "model": "moonshotai/Kimi-K2.5", 
      "messages": [{"role": "user", "content": "中国的首都是什么？"}], 
      "temperature": 0, 
      "max_tokens": 400
  }'
```
