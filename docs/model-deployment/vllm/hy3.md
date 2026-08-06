# Hy3 on vLLM

## 模型简介

Hy3 是由腾讯混元团队开发的一款拥有 2950 亿参数的混合专家（MoE）模型，其中激活参数为 210 亿，MTP 层参数为 38 亿。基于 Channel FP8 量化优化，在 HCU 硬件上提供高效推理性能。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#hy3-channel-fp8-w8a8-ifb-bw1100-8x-vllm-021) |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | 0.21 | BW1100(超节点) | 16 | IFB | [**`>_`**](#hy3-channel-fp8-w8a8-ifb-bw1100-16x-vllm-021) |
| [hygon/Hy3-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/Hy3-Channel-FP8-w8a8) | FP8 W8A8 | 0.21 | BW1100(超节点) | 32 | 2P1D | [**`>_`**](#hy3-channel-fp8-w8a8-2p1d-bw1100-32x-vllm-021) |

## 启动命令

### Hy3-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.21

以下示例为单节点部署。

```bash
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --kv_cache_dtype fp8_e4m3 \
  --gpu-memory-utilization 0.92 \
  --enable-auto-tool-choice \
  --served-model-name hy3-fp8
```

### Hy3-Channel-FP8-w8a8 IFB BW1100 16x vLLM 0.21

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ROCSHMEM_HEAP_SIZE=5368709120
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  -q slimquant_marlin \
  --enable-expert-parallel \
  --all2all_backend=deepep_low_latency \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 700 \
  --max-num-batched-tokens 700 \
  --no-enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.90 \
  --data-parallel-size-local 16 \
  --data-parallel-rpc-port 1127 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}'
```

### Hy3-Channel-FP8-w8a8 2P1D BW1100 32x vLLM 0.21

P node 0 和 P node 1 分别使用服务端口 `8010` 和 `8011`。

#### P node 0

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export ROCSHMEM_HEAP_SIZE=5368709120
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --enable-custom-sp \
  --enforce-eager \
  --kv_cache_dtype fp8_e4m3 \
  --port 8010 \
  --enable-expert-parallel \
  --all2all_backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

#### P node 1

```bash
export HIP_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export GPU_MAX_HW_QUEUES=4
export ROCSHMEM_HEAP_SIZE=5368709120
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export VLLM_MOONCAKE_BOOTSTRAP_PORT=8998

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  --trust-remote-code \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2 \
  --speculative-config.quantization "slimquant_marlin" \
  -q slimquant_marlin \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 128 \
  --dtype bfloat16 \
  --tensor-parallel-size 8 \
  --enable-prefix-caching \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --enable-custom-sp \
  --enforce-eager \
  --kv_cache_dtype fp8_e4m3 \
  --port 8011 \
  --enable-expert-parallel \
  --all2all_backend deepep_high_throughput \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

#### D node

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export ROCSHMEM_HEAP_SIZE=5368709120
export ROCSHMEM_IPC_MNVL=1
export VLLM_DEEPEP_BUFFER_SIZE_MB=0
export GPU_MAX_HW_QUEUES=4
export VLLM_HCU_USE_CUSTOM_FLASH_ATTN=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1

vllm serve hygon/Hy3-Channel-FP8-w8a8 \
  --trust-remote-code \
  -dp 16 \
  -tp 1 \
  -q slimquant_marlin \
  --enable-expert-parallel \
  --all2all_backend=deepep_low_latency \
  --disable-custom-all-reduce \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-model-len 16384 \
  --max-num-seqs 700 \
  --max-num-batched-tokens 700 \
  --no-enable-prefix-caching \
  --block-size 64 \
  --gpu-memory-utilization 0.90 \
  --data-parallel-size-local 16 \
  --data-parallel-rpc-port 1127 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative_config '{"method":"mtp","num_speculative_tokens":2, "quantization": "slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

#### + Static EPLB

详细配置和使用流程请参考：[Static EPLB：离线专家布局的记录与加载](../../optimization/static-eplb-vllm.md)。

```bash
#record阶段 参数加:
--enable-eplb \
  --eplb-config '{"expert_map_record_path":"/path/to/eplb.json", "num_redundant_experts":16, "step_interval":3000, "log_balancedness":true, "log_balancedness_interval":100, "use_async":false}'

#replay阶段 参数加:
--enable-eplb \
  --eplb-config '{"expert_map_path":"/path/to/eplb.json", "num_redundant_experts":16, "step_interval":100000000000, "log_balancedness":true, "log_balancedness_interval":100, "use_async":false}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hy3-fp8",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

```bash
curl http://0.0.0.0:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hy3-fp8",
  "messages": [
    {"role": "user", "content": "你好，请用Python写一个贪吃蛇的游戏脚本"}
  ],
  "max_tokens": 1500,
  "temperature": 0.0
  }'
```

### PD 分离

PD 分离模式下，客户端请求发送到 P 节点服务端口。下面以 P node 0 的 `8010` 端口为例。

```python
from openai import OpenAI

client = OpenAI(base_url="http://<P_node0_ip>:8010/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/Hy3-Channel-FP8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手。"},
        {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://<P_node0_ip>:8010/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/Hy3-Channel-FP8-w8a8",
  "messages": [
    {"role": "user", "content": "你好，请用 Python 写一个贪吃蛇游戏脚本"}
  ],
  "max_tokens": 1500,
  "temperature": 0.0
  }'
```
