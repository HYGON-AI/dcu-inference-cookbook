# DeepSeek-R1 on vLLM

## 模型简介

DeepSeek-R1 是 DeepSeek 推出的推理强化模型，面向复杂推理、数学与代码场景，通过强化学习提升模型的思维链推理能力。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [deepseek-ai/DeepSeek-R1-Distill-Qwen-32B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) | FP16 | 0.21 | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1100-2x-vllm-021) |
|                                                                                                                   | FP16 | 0.21 | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1000-4x-vllm-021) |
|                                                                                                                   | FP16 | 0.21 | K100_AI | 4 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-k100_ai-4x-vllm-021) |
|                                                                                                                   | FP16 | [0.18](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1100-2x-vllm-018) |
|                                                                                                                   | FP16 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-bw1000-4x-vllm-018) |
|                                                                                                                   | FP16 | [0.18](../docker_images.md) | K100_AI | 4 | IFB | [**`>_`**](#deepseek-r1-distill-qwen-32b-ifb-k100ai-4x-vllm-018) |
| [deepseek-ai/DeepSeek-R1-Distill-Llama-70B](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) | BF16 | 0.21 | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1100-2x-vllm-021) |
|                                                                                                                     | BF16 | 0.21 | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1000-4x-vllm-021) |
|                                                                                                                     | BF16 | 0.21 | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-k100_ai-8x-vllm-021) |
|                                                                                                                     | BF16 | [0.18](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1100-2x-vllm-018) |
|                                                                                                                     | BF16 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-bw1000-4x-vllm-018) |
|                                                                                                                     | BF16 | [0.18](../docker_images.md) | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-ifb-k100ai-8x-vllm-018) |
| [hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) | INT8 W8A8 | 0.21 | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-bw1100-2x-vllm-021) |
|                                                                                                                                             | INT8 W8A8 | 0.21 | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-bw1000-4x-vllm-021) |
|                                                                                                                                             | INT8 W8A8 | 0.21 | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-k100_ai-8x-vllm-021) |
|                                                                                                                                             | INT8 W8A8 | [0.18](../docker_images.md) | BW1100 | 2 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-bw1100-2x-vllm-018) |
|                                                                                                                                             | INT8 W8A8 | [0.18](../docker_images.md) | BW1000 | 4 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-bw1000-4x-vllm-018) |
|                                                                                                                                             | INT8 W8A8 | [0.18](../docker_images.md) | K100_AI | 8 | IFB | [**`>_`**](#deepseek-r1-distill-llama-70b-quantizedw8a8-ifb-k100ai-8x-vllm-018) |
| [hygon/DeepSeek-R1-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Channel-INT8-w8a8) | INT8 W8A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1100-8x-vllm-021) |
|                                                                                                              | INT8 W8A8 | 0.21 | BW1000 | 16 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1000-16x-vllm-021) |
|                                                                                                              | INT8 W8A8 | [0.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1100-8x-vllm-018) |
|                                                                                                              | INT8 W8A8 | [0.18](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1000-16x-vllm-018) |
|                                                                                                              | INT8 W8A8 | [0.15](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1100-8x-vllm-015) |
|                                                                                                              | INT8 W8A8 | [0.15](../docker_images.md) | BW1000 | 16 | IFB | [**`>_`**](#deepseek-r1-channel-int8-w8a8-ifb-bw1000-16x-vllm-015) |
| [hygon/DeepSeek-R1-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/DeepSeek-R1-Channel-FP8-w8a8) | FP8 W8A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-vllm-021) |
|                                                                                                             | FP8 W8A8 | [0.18](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-vllm-018) |
|                                                                                                             | FP8 W8A8 | [0.15](../docker_images.md) | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-channel-fp8-w8a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-R1-W4A8-V2_6](https://www.modelscope.cn/models/hygon/DeepSeek-R1-W4A8-V2_6) | INT4 W4A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-v2_6-ifb-bw1100-8x-vllm-021) |
|                                                                                              | INT4 W4A8 | 0.21 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-v2_6-ifb-bw1000-8x-vllm-021) |
|                                                                                              | INT4 W4A8 | [0.18] | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-v2_6-ifb-bw1100-8x-vllm-018) |
|                                                                                              | INT4 W4A8 | [0.18] | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-v2_6-ifb-bw1000-8x-vllm-018) |
| [hygon/DeepSeek-R1-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-R1-0528-W4A8-V2) | INT4 W4A8 | 0.21 | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-0528-w4a8-v2-ifb-bw1100-8x-vllm-021) |
|                                      | INT4 W4A8 | 0.21 | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-0528-w4a8-v2-ifb-bw1000-8x-vllm-021) |
|                                      | INT4 W4A8 | [0.18](../docker_images.md) | BW1100 | 4 | IFB | [**`>_`**](#deepseek-r1-0528-w4a8-v2-ifb-bw1100-4x-vllm-018) |
|                                      | INT4 W4A8 | [0.18](../docker_images.md) | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-0528-w4a8-v2-ifb-bw1000-8x-vllm-018) |
| [hygon/DeepSeek-R1-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-R1-0528-W4A8-V2) | W4A8 | [0.15] | BW1100 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-ifb-bw1100-8x-vllm-015) |
| [hygon/DeepSeek-R1-0528-W4A8-V2](https://www.modelscope.cn/models/hygon/DeepSeek-R1-0528-W4A8-V2) | W4A8 | [0.15] | BW1000 | 8 | IFB | [**`>_`**](#deepseek-r1-w4a8-ifb-bw1000-8x-vllm-015) |

## 启动命令

### DeepSeek-R1-Distill-Qwen-32B IFB BW1100 2x vLLM 0.21

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1000 4x vLLM 0.21

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Qwen-32B IFB K100_AI 4x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1100 2x vLLM 0.21

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1000 4x vLLM 0.21

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Llama-70B IFB K100_AI 8x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dtype float16 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB BW1100 2x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB BW1000 4x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --attention-backend FLASH_ATTN_CUSTOM
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB K100_AI 8x vLLM 0.21

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --dtype float16 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1100 2x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Qwen-32B IFB BW1000 4x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --kv-cache-dtype fp8_e5m2 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Qwen-32B IFB K100_AI 4x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1100 2x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B IFB BW1000 4x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --kv-cache-dtype fp8_e5m2 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B IFB K100_AI 8x vLLM 0.18

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
  --dtype float16 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB BW1100 2x vLLM 0.18

```bash
vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --kv-cache-dtype fp8_e4m3 \
  --dtype float16 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB BW1000 4x vLLM 0.18

```bash
vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --kv-cache-dtype fp8_e5m2 \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 IFB K100_AI 8x vLLM 0.18

```bash
export VLLM_HCU_USE_CUSTOM_QUANTIZATION_GEMM=0
export VLLM_HCU_USE_CUSTOM_OPS=0

vllm serve hygon/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8 \
  --dtype float16 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --max-num-seqs 1024 \
  --distributed-executor-backend=mp \
  --no-enable-prefix-caching \
  --disable-cascade-attn
```

### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}' \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHMLA
```

### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.21

#### Master node

```bash
vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
  --attention-backend FLASHMLA \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr <master_node_ip>
```

#### Worker node

```bash
vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
  --attention-backend FLASHMLA \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr <master_node_ip> \
  --headless
```
### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export VLLM_USE_OPT_CAT=1
export VLLM_HCU_USE_CAT_MLA=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_FLASHMLA=1

vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
  --kv-cache-dtype fp8_e4m3
```

### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.18

#### 主节点

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export VLLM_USE_OPT_CAT=1
export VLLM_HCU_USE_CAT_MLA=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_FLASHMLA=1

vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 12.5.6.42 \
  --master-port 30000
```

#### 从节点

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export VLLM_USE_OPT_CAT=1
export VLLM_HCU_USE_CAT_MLA=1
export VLLM_HCU_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_HCU_USE_FLASHMLA=1

vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --max-model-len 65536 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 3,"quantization": "slimquant_marlin"}' \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 12.5.6.42 \
  --headless \
  --master-port 30000
```

### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.15

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export VLLM_SPEC_DECODE_EAGER=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=1
export VLLM_USE_FLASH_ATTN_FP8=0
export VLLM_USE_CAT_MLA=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1

vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 35000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}' \
  --kv-cache-dtype fp8_e4m3
```

### DeepSeek-R1-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.15
ray方式启动，示例如下：

ray start --head --node-ip-address=xx.xx.xx.xx  --port=6570 --num-gpus=8 --num-cpus=16

ray start --address='xx.xx.xx.xx:6570' --num-gpus=8 --num-cpus=16

```bash
export VLLM_HOST_IP=12.12.12.41
export NCCL_SOCKET_IFNAME=ens47f0np0
export GLOO_SOCKET_IFNAME=ens47f0np0
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx6_1:1,mlx5_8:1,mlx5_9:1
export NCCL_NET_GDR_READ=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export VLLM_SPEC_DECODE_EAGER=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=1
export VLLM_USE_FLASH_ATTN_FP8=0
export VLLM_USE_CAT_MLA=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1

vllm serve hygon/DeepSeek-R1-Channel-INT8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 16 \
  --dtype bfloat16 \
  --max-model-len 35000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}'
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.21

```bash
vllm serve hygon/DeepSeek-R1-Channel-FP8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}' \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHMLA
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.18

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export VLLM_USE_OPT_CAT=0
export VLLM_HCU_USE_CAT_MLA=0
export VLLM_HCU_USE_FLASHMLA=1

vllm serve hygon/DeepSeek-R1-Channel-FP8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}' \
  --no-enable-prefix-caching \
  --kv-cache-dtype fp8_e4m3
```

### DeepSeek-R1-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.15

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export VLLM_SPEC_DECODE_EAGER=0
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=16384
export USE_FUSED_RMS_QUANT=0
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=1
export VLLM_USE_FLASH_ATTN_FP8=1
export VLLM_USE_CAT_MLA=1
export VLLM_USE_LIGHTOP=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1

vllm serve hygon/DeepSeek-R1-Channel-FP8-w8a8 \
  --trust-remote-code \
  -q slimquant_marlin \
  -tp 8 \
  --dtype bfloat16 \
  --max-model-len 35000 \
  --disable-log-requests \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --compilation-config '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "deepseek_mtp", "num_speculative_tokens": 2,"quantization": "slimquant_marlin"}' \
  --kv-cache-dtype fp8_e4m3
```

### DeepSeek-R1-W4A8-V2_6 IFB BW1100 8x vLLM 0.21

```bash
export VLLM_USE_MODELSCOPE=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-W4A8-V2_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --attention-backend FLASHMLA \
  --moe-backend aiter
```

### DeepSeek-R1-W4A8-V2_6 IFB BW1000 8x vLLM 0.21

```bash
export VLLM_USE_MODELSCOPE=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-W4A8-V2_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --attention-backend FLASHMLA \
  --moe-backend aiter
```

### DeepSeek-R1-W4A8-V2_6 IFB BW1100 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-W4A8-V2_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
```

### DeepSeek-R1-W4A8-V2_6 IFB BW1000 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-W4A8-V2_6 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
```

### DeepSeek-R1-0528-W4A8-V2 IFB BW1100 8x vLLM 0.21

```bash
export VLLM_USE_MODELSCOPE=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --attention-backend FLASHMLA \
  --moe-backend aiter
```

### DeepSeek-R1-0528-W4A8-V2 IFB BW1000 8x vLLM 0.21

```bash
export VLLM_USE_MODELSCOPE=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}' \
  --attention-backend FLASHMLA \
  --moe-backend aiter
```

### DeepSeek-R1-0528-W4A8-V2 IFB BW1100 4x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --max-model-len 65536 \
  --max-num-batched-tokens 8192 \
  -tp 4 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
```

### DeepSeek-R1-0528-W4A8-V2 IFB BW1000 8x vLLM 0.18

```bash
export VLLM_USE_MODELSCOPE=1
export VLLM_ROCM_USE_AITER_MOE=1
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_HCU_USE_CAT_MLA=0

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --max-num-batched-tokens 8192 \
  -tp 8 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 256 \
  --speculative_config '{"method":"deepseek_mtp","num_speculative_tokens":2}'
```

### DeepSeek-R1-W4A8 IFB BW1100 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_CUSTOM_CACHE=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_FLASH_MLA=1
export VLLM_USE_CAT_MLA=1

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  -q slimquant_w4a8_marlin \
  -tp 8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_w4a8_marlin"}'
```

### DeepSeek-R1-W4A8 IFB BW1000 8x vLLM 0.15

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0
export VLLM_USE_PIECEWISE=1
export USE_FUSED_RMS_QUANT=1
export USE_FUSED_SILU_MUL_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_CUSTOM_CACHE=1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_FLASH_MLA=1

vllm serve hygon/DeepSeek-R1-0528-W4A8-V2 \
  --trust-remote-code \
  --dtype bfloat16 \
  -q slimquant_w4a8_marlin \
  -tp 8 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --block-size 64 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --kv-cache-dtype fp8_e5m2 \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_w4a8_marlin"}'
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/DeepSeek-R1-0528-W4A8-V2",
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
  "model": "hygon/DeepSeek-R1-0528-W4A8-V2",
  "messages": [
    {"role": "user", "content": "你好，请用Python写一个贪吃蛇的游戏脚本"}
  ],
  "max_tokens": 1500,
  "temperature": 0.0
  }'
```
