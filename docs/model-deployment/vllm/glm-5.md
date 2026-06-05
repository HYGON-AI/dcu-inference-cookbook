# GLM-5 on vLLM

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1100-8x-vllm-015) |
| [hygon/GLM-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-FP8-w8a8) | FP8 W8A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#glm-5-channel-fp8-w8a8-ifb-bw1100-8x-vllm-015) |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100 | 8 | IFB | [**``>_``**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-vllm-018) |
|  | INT8 W8A8 | 0.18 | BW1000 | 16 | IFB | [**``>_``**](#glm-5-channel-int8-w8a8-ifb-bw1000-16x-vllm-018) |
|  | INT8 W8A8 | 0.15 | BW1100 | 8 | IFB | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-vllm-015) |
|  | INT8 W8A8 | 0.15 | BW1100 | 24 | 1P2D | [**``>_``**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-vllm-015) |

## 启动命令

### GLM-5-Channel-INT4-w4a8 IFB BW1100 8x vLLM 0.15

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0 # 宽松采样，提高mtp接受率从而提高tpot性能，略微影响精度 default 1

#===============融合算子=================
export USE_FUSED_RMS_QUANT=1 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和kvfp8一起开
export VLLM_USE_V32_ENCODE=1
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

vllm serve hygon/GLM-5-Channel-INT4-w4a8 \
  -q slimquant_w4a8_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 8 \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.92 \
  --disable-log-requests \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_w4a8_marlin"}'
```

### GLM-5-Channel-FP8-w8a8 IFB BW1100 8x vLLM 0.15

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0 # 宽松采样，提高mtp接受率从而提高tpot性能，略微影响精度 default 1

#===============融合算子=================
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=0 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和kvfp8一起开
export VLLM_USE_V32_ENCODE=1
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

vllm serve hygon/GLM-5-Channel-FP8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 8 \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.92 \
  --disable-log-requests \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}'
```

### GLM-5-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.18

以下示例为单节点部署。

```bash
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    -tp 8 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --block-size 64 \
    --speculative_config '{
        "method":"deepseek_mtp",
        "num_speculative_tokens":2,
        "quantization":"slimquant_marlin"
    }' \
    --kv-cache-dtype fp8_ds_mla
```

### GLM-5-Channel-INT8-w8a8 IFB BW1000 16x vLLM 0.18

以下示例为双节点部署，请根据实际情况修改 `<node1_ip>`。

#### Node 1

```bash
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    -tp 16 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --block-size 64 \
    --speculative_config '{
        "method":"deepseek_mtp",
        "num_speculative_tokens":2,
        "quantization":"slimquant_marlin"
    }' \
    --kv-cache-dtype fp8_ds_mla \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr <node1_ip>
```

#### Node 2

```bash
export VLLM_HCU_USE_FLASHMLA=1
export LMSLIM_USE_GLOBAL_MOE_CACHE=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 65536 \
    --max-num-batched-tokens 8192 \
    -tp 16 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --block-size 64 \
    --speculative_config '{
        "method":"deepseek_mtp",
        "num_speculative_tokens":2,
        "quantization":"slimquant_marlin"
    }' \
    --kv-cache-dtype fp8_ds_mla \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr <node1_ip> \
    --headless
```

### GLM-5-Channel-INT8-w8a8 IFB BW1100 8x vLLM 0.15

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0 # 宽松采样，提高mtp接受率从而提高tpot性能，略微影响精度 default 1

#===============融合算子=================
export USE_FUSED_RMS_QUANT=1 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和kvfp8一起开
export VLLM_USE_V32_ENCODE=1
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 8 \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.92 \
  --disable-log-requests \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}'
```

### GLM-5-Channel-INT8-w8a8 1P2D BW1100 24x vLLM 0.15

以下示例中 `10.16.1.36` 为 P 节点，也是代理节点，`10.16.1.42` 是 D node 0 的主节点，实际部署时请根据实际情况修改。

#### P node

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0

#=====融合算子=====
export USE_FUSED_RMS_QUANT=1 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTTOP_MOE_SUM_MUL_ADD=1 #w4a16/awq # 需要关闭
export VLLM_USE_LIGHTTOP_RMS_ROPE_CONCAT=0 # 不能和 kvfp8 一起开
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_DISABLE_DSA=0
export USE_LIGHTTOP_TOPK=1
export USE_LIGHTTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_USE_DP_CONNECTOR=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo_400g.conf

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -tp 8 \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}' \
  --enable-lightly-cp --enable-lightly-cplb \
  --enforce-eager \
  --kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_producer","kv_buffer_size":"1e4","kv_port":"21002","kv_connector_extra_config":{"proxy_ip":"10.16.1.36","proxy_port":"30001","http_port":"8000","send_type":"PUT_ASYNC","instance_ip":"10.16.1.36"}}'
```

#### D node 0

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0

#=====融合算子=====
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0 #w4a16/awq # 需要关闭
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和 kvfp8 一起开
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

# DEEPEP / NCCL
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

#deep_ep
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256

#EP
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DP_CONNECTOR=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -dp 16 \
  -tp 1 \
  --block-size 64 \
  --enable-expert-parallel \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.88 \
  --disable-log-requests \
  --max-num-batched-tokens 128 \
  --kv-cache-dtype fp8_ds_mla \
  --cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector": "DuSwiftConnectorDp", "kv_role": "kv_consumer", "kv_buffer_size": "1e9", "kv_port": "21003", "kv_connector_extra_config": {"proxy_ip": "10.16.1.36", "proxy_port": "30001", "http_port": "8000", "send_type": "PUT_ASYNC", "instance_ip": "10.16.1.42"}}' \
  --data-parallel-size-local 8 \
  --data-parallel-address 10.16.1.42 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 0 \
  --disable-custom-all-reduce
```

#### D node 1

```bash

export VLLM_USE_MODELSCOPE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_REJECT_SAMPLE_OPT=0

#=====融合算子=====
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

#注意： --compilation-config 中需要添加"custom_ops": ["all", "+rms_norm"]
export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_USE_FUSED_FILL_RMS_CAT=1
export VLLM_USE_LIGHTTOP_MOE_SUM_MUL_ADD=0 #w4a16/awq # 需要关闭
export VLLM_USE_LIGHTTOP_RMS_ROPE_CONCAT=0 # 不能和 kvfp8 一起开
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_DISABLE_DSA=0
export USE_LIGHTTOP_TOPK=1
export USE_LIGHTTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

# DEEPEP / NCCL
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

#deep_ep
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256

#EP
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DP_CONNECTOR=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
  -q slimquant_marlin \
  --trust-remote-code \
  --dtype bfloat16 \
  -dp 16 \
  -tp 1 \
  --block-size 64 \
  --enable-expert-parallel \
  --max-model-len 72000 \
  --gpu-memory-utilization 0.88 \
  --disable-log-requests \
  --max-num-batched-tokens 128 \
  --kv-cache-dtype fp8_ds_mla \
  -cc '{"pass_config": {"fuse_act_quant": false}}' \
  --speculative_config '{"method": "mtp", "num_speculative_tokens": 2, "quantization": "slimquant_marlin"}' \
  --kv-transfer-config '{"kv_connector": "DuSwiftConnectorDp", "kv_role": "kv_consumer", "kv_buffer_size": "1e9", "kv_port": "21003", "kv_connector_extra_config": {"proxy_ip": "10.16.1.36", "proxy_port": "30001", "http_port": "8000", "send_type": "PUT_ASYNC", "instance_ip": "10.16.1.42"}}' \
  --data-parallel-size-local 8 \
  --data-parallel-address 10.16.1.42 \
  --data-parallel-rpc-port 1127 \
  --data-parallel-start-rank 8 \
  --disable-custom-all-reduce \
  --headless
```

## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT4-w4a8",  # 替换为实际使用的模型名
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/GLM-5-Channel-INT4-w4a8",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "中国的首都是哪里？"}
  ],
  "max_tokens": 128
  }'
```

### PD 分离

vLLM PD 分离模式下，客户端请求直接发送到 P 节点的代理端口（示例中为 `10.16.1.36:30001`）。

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.16.1.36:30001/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT8-w8a8",
    messages=[
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": "请分析一下当前中国 AI 芯片产业的发展现状"},
    ],
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

```bash
curl http://10.16.1.36:30001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "hygon/GLM-5-Channel-INT8-w8a8",
  "messages": [
    {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
    {"role": "user", "content": "中国的首都是哪里？"}
  ],
  "max_tokens": 128
  }'
```
