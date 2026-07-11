# GLM-5.1 on vLLM

## 模型简介

GLM-5.1 是 Z.ai 推出的 GLM-5 系列模型版本，适用于长上下文、对话生成和工具调用等场景。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5.1-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5.1-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-51-w8a8-1p2d-bw1100-24x-vllm-018) |

## 启动命令

### GLM-5.1-Channel-INT8-w8a8 1P2D BW1100 24x vLLM 0.18

以下示例中 `13.13.3.21` 为 P 节点，`13.13.3.25` 为 D node 0，`13.13.3.28` 为 D node 1，实际部署时请根据实际情况修改。

#### P node

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_HOST_IP=13.13.3.21
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export NCCL_SOCKET_IFNAME=ens61f0np0
export GLOO_SOCKET_IFNAME=ens61f0np0

vllm serve hygon/GLM-5.1-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --port 30000 \
  --enforce-eager \
  --dtype bfloat16 \
  --max-num-batched-tokens 16384 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 64 \
  --block-size 64 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
  --kv-cache-dtype fp8_ds_mla \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --enable-lightly-cp \
  --enable-lightly-cplb
```

#### D node 0

```bash
export VLLM_HOST_IP=13.13.3.25
export NCCL_SOCKET_IFNAME=ens61f0np0
export GLOO_SOCKET_IFNAME=ens61f0np0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ROCSHMEM_IB_GID_INDEX=0
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/GLM-5.1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --port 30000 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.89 \
  --data-parallel-size-local 8 \
  --data-parallel-address 13.13.3.25 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 0 \
  --kv-cache-dtype fp8_ds_mla \
  -q slimquant_marlin \
  --max-num-seqs 128 \
  --max-num-batched-tokens 128 \
  --all2all_backend=deepep_low_latency \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

#### D node 1

```bash
export VLLM_HOST_IP=13.13.3.28
export NCCL_SOCKET_IFNAME=ens61f0np0
export GLOO_SOCKET_IFNAME=ens61f0np0
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ROCSHMEM_IB_GID_INDEX=0
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=./topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_USE_MODELSCOPE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/GLM-5.1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  --port 30000 \
  --enable-expert-parallel \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.89 \
  --data-parallel-size-local 8 \
  --data-parallel-address 13.13.3.25 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 8 \
  --kv-cache-dtype fp8_ds_mla \
  -q slimquant_marlin \
  --max-num-seqs 128 \
  --max-num-batched-tokens 128 \
  --headless \
  --all2all_backend=deepep_low_latency \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2,"quantization":"slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}' \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'
```

## API 调用

### PD 分离

vLLM PD 分离模式下，客户端请求发送到 P 节点服务端口，示例中为 `13.13.3.21:9351`。

```python
from openai import OpenAI

client = OpenAI(base_url="http://13.13.3.21:9351/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5.1-Channel-INT8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请简单介绍 GLM-5.1。"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://13.13.3.21:9351/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/GLM-5.1-Channel-INT8-w8a8",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "请简单介绍 GLM-5.1。"}
  ],
  "max_tokens": 128
  }'
```
