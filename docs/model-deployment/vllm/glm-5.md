# GLM-5 on vLLM

## 模型简介

GLM-5 是智谱 AI 推出的新一代大语言模型，在中文理解、长文本处理、工具调用等方面表现优异。

## 模型列表

| 模型 | 参数量 | 上下文 | 推荐硬件 |
|------|--------|--------|---------|
| GLM-5 | 754B | 198K | IFB:8x BW1100 144GB && PD分离:1P2D 24x BW1100 144GB |

## 启动命令

### IFB【TP8+MTP2】

```
rm -rf ~/.cache
rm -rf ~/.triton

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7     
export ALLREDUCE_STREAM_WITH_COMPUTE=1           
export NCCL_MIN_NCHANNELS=16                      
export NCCL_MAX_NCHANNELS=16                   
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1   
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PD_SPLIT=1
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=1 # 宽松采样，提高mtp接受率从而提高tpot性能，略微影响精度 default 1

#===============融合算子=================
export USE_FUSED_RMS_QUANT=1 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0

#===============共享专家融合 已知精度异常=============
export VLLM_ROCM_USE_AITER=0 #default 0
export VLLM_ROCM_USE_AITER_MOE=0 #default 0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 #default 0

export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_CUSTOM_CACHE=1 #default 1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 #default 1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和kvfp8一起开
export VLLM_USE_V32_ENCODE=1
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

MODEL_PATH=/module/GLM-5-W8A8

vllm serve "$MODEL_PATH" \
    -q slimquant_marlin \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    -tp 8 \
    --max-model-len 72000 \
    --gpu-memory-utilization 0.92 \
    --disable-log-requests \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8_ds_mla \
    -cc '{"pass_config": {"fuse_act_quant": false}}' \
    --speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}'
```

### PD分离

### 以下示例中10.16.1.36为P节点，也是代理节点，10.16.1.42是D节点中的主节点，实际部署时请根据实际情况修改

### P节点(CP8)

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PD_SPLIT=1
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=1
#=====融合算子=====
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0
#=====共享专家融合 已知精度异常=====
export VLLM_ROCM_USE_AITER=0 #default 0
export VLLM_ROCM_USE_AITER_MOE=0 #default 0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 #default 0
#注意：--compilation-config 中需要添加"custom_ops": ["all", "+rms_norm"]
export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_CUSTOM_CACHE=1 #default 1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 #default 1
export VLLM_USE_LIGHTTOP_MOE_SUM_MUL_ADD=0 #w4a16/awq # 需要关闭
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
export
NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo_400g.conf

rm -rf ~/.cache
rm -rf ~/.triton
MODEL_PATH=/module/GLM-5-W8A8

vllm serve "$MODEL_PATH" \
-q slimquant_marlin \
--port 20012 \
--trust-remote-code \
--dtype bfloat16 \
-tp 8 \
--max-model-len 72000 \
--gpu-memory-utilization 0.9 \
--disable-log-requests \
--enable-chunked-prefill \
--max-num-batched-tokens 16384 \
--enable-prefix-caching \
--kv-cache-dtype fp8_ds_mla \
-cc '{"pass_config": {"fuse_act_quant": false}}' \
--speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}' \
--enable-lightly-cp --enable-lightly-cplb \
--enforce-eager \
--kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_producer","kv_buffer_size":"1e4","kv_port":"21002","kv_connector_extra_config":{"proxy_ip":"10.16.1.36","proxy_port":"30001","http_port":"20012","send_type":"PUT_ASYNC","instance_ip":"10.16.1.36"}}'
```

### D0节点(DP16EP16+MTP3)

```bash
rm -rf ~/.cache
rm -rf ~/.triton
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PD_SPLIT=1
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=1
#=====融合算子=====
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0
#=====共享专家融合 已知精度异常=====
export VLLM_ROCM_USE_AITER=0 #default 0
export VLLM_ROCM_USE_AITER_MOE=0 #default 0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 #default 0
#注意：--compilation-config 中需要添加"custom_ops": ["all", "+rms_norm"]
export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_CUSTOM_CACHE=1 #default 1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 #default 1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=0 #w4a16/awq # 需要关闭
export VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT=0 # 不能和 kvfp8 一起开
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_DISABLE_DSA=0
export USE_LIGHTOP_TOPK=1
export USE_LIGHTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1
#多机需要额外增加：通信部分设置，根据实际环境调整
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

# DEEPEP / NCCL
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

#deep_ep new
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo_400g.config
export USE_SPE MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256

#EP
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DP_CONNECTOR=1

rm -rf ~/.cache
rm -rf ~/.triton
MODEL_PATH=/module/GLM-5-W8A8

vllm serve "$MODEL_PATH" \
-q slimquant_marlin \
--port 20013 \
--trust-remote-code \
--dtype bfloat16 \
-dp 16 \
-tp 1 \
--block-size 64 \
--enable-expert-parallel \
--max-model-len 72000 \
--gpu-memory-utilization 0.88 \
--disable-log-requests \
--enable-chunked-prefill \
--max-num-batched-tokens 128 \
--enable-prefix-caching \
--kv-cache-dtype fp8_ds_mla \
--cc '{"pass_config": {"fuse_act_quant": false}}' \
--speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}' \
--kv-transfer-config '{"kv_connector": "DuSwiftConnectorDp", "kv_role": "kv_consumer", "kv_buffer_size": "1e9", "kv_port": "21003", "kv_connector_extra_config": {"proxy_ip": "10.16.1.36", "proxy_port": "30001", "http_port": "20013", "send_type": "PUT_ASYNC", "instance_ip": "10.16.1.42"}}' \
--data-parallel-size-local 8 \
--data-parallel-address 10.16.1.42 \
--data-parallel-rpc-port 1127 \
--data-parallel-start-rank 0 \
--disable-custom-all-reduce
```

### D1节点(DP16EP16+MTP3)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export Allgather_Base_STREAM_WITH_COMPUTE=1
export SENDRECV_STREAM_WITH_COMPUTE=1
export HIP_KERNEL_EVENT_SYSTEMFENCE=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_PD_SPLIT=1
export VLLM_USE_PIECEWISE=0
export VLLM_REJECT_SAMPLE_OPT=1
#=====融合算子=====
export USE_FUSED_RMS_QUANT=0 #default 0
export USE_FUSED_SILU_MUL_QUANT=1 #default 0
#=====共享专家融合 已知精度异常=====
export VLLM_ROCM_USE_AITER=0 #default 0
export VLLM_ROCM_USE_AITER_MOE=0 #default 0
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0 #default 0
#注意： --compilation-config 中需要添加"custom_ops": ["all", "+rms_norm"]
export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 default 0
export VLLM_FUSED_MOE_CHUNK_SIZE=16384 #default 32768
export VLLM_CUSTOM_CACHE=1 #default 1
export VLLM_USE_OPT_CAT=1
export VLLM_USE_FUSED_FILL_RMS_CAT=1 #default 1(mtp 可能存在精度问题
export VLLM_USE_LIGHTTOP_MOE_SUM_MUL_ADD=0 #w4a16/awq # 需要关闭
export VLLM_USE_LIGHTTOP_RMS_ROPE_CONCAT=0 # 不能和 kvfp8 一起开
export VLLM_USE_V32_ENCODE=1
export VLLM_USE_FLASH_MLA=1
export VLLM_DISABLE_DSA=0
export USE_LIGHTTOP_TOPK=1
export USE_LIGHTTOP_PER_TOKEN_GROUP_QUANT_FP8=1
export USE_LIGHTTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX=1

#多机需要额外增加：根据实际测试环境调整
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
export NCCL_SOCKET_IFNAME=enp33s0f3u1
export GLOO_SOCKET_IFNAME=enp33s0f3u1
export NCCL_IB_HCA=mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

# DEEPEP / NCCL
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0

#deep_ep new
export ROCSHMEM_HEAP_SIZE=4000000000
export ROCSHMEM_TOPO_FILE_FORCE=/mnt/glm5_pd/topo_400g.config
export USE_SPE_MQP=1
export ROCSHMEM_SQ_SIZE=1024
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=256

#EP
export VLLM_MOE_DP_CHUNK_SIZE=128
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DP_CONNECTOR=1

rm -rf ~/.cache
rm -rf ~/.triton
MODEL_PATH=/module/GLM-5-W8A8
vllm serve "$MODEL_PATH" \
-q slimquant_marlin \
--port 20013 \
--trust-remote-code \
--dtype bfloat16 \
-dp 16 \
-tp 1 \
--block-size 64 \
--enable-expert-parallel \
--max-model-len 72000 \
--gpu-memory-utilization 0.88 \
--disable-log-requests \
--enable-chunked-prefill \
--max-num-batched-tokens 128 \
--enable-prefix-caching \
--kv-cache-dtype fp8_ds_mla \
-cc '{"pass_config": {"fuse_act_quant": false}}' \
--speculative_config '{"method": "mtp", "num_speculative_tokens": 3, "quantization": "slimquant_marlin"}' \
--kv-transfer-config '{"kv_connector": "DuSwiftConnectorDp", "kv_role": "kv_consumer", "kv_buffer_size": "1e9", "kv_port": "21003", "kv_connector_extra_config": {"proxy_ip": "10.16.1.36", "proxy_port": "30001", "http_port": "20013", "send_type": "PUT_ASYNC", "instance_ip": "10.16.1.42"}}' \
--data-parallel-size-local 8 \
--data-parallel-address 10.16.1.42 \
--data-parallel-rpc-port 1127 \
--data-parallel-start-rank 8 \
--disable-custom-all-reduce \
--headless
```
