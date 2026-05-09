# MiniMax-2.x on vLLM

## 模型简介

MiniMax-2.5 是 MiniMax 推出的大规模 MoE（混合专家）语言模型系列，总参数量 229B，激活参数约 10B，在长文本理解和生成方面表现突出。

## 模型列表

| 模型 | 总参数 | 激活参数 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|---------|--------|---------|---------|
| MiniMax-2.5 | 229B | ~10B | 1M | INT8 W8A8 | 8x BW1100 144GB|

## 启动命令

### MiniMax-Text-01（八卡 128GB）

```bash
rm -rf ~/.cache
rm -rf ~/.triton
rm -rf /tmp/torchinductor_root/

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_P2P_LEVEL=SYS
export NCCL_LAUNCH_MODE=GROUP
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=0
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=1
export VLLM_RANK4_NUMA=2
export VLLM_RANK5_NUMA=2
export VLLM_RANK6_NUMA=3
export VLLM_RANK7_NUMA=3
export NCCL_NET_GDR_READ=1
export VLLM_RPC_TIMEOUT=1800000
export NCCL_NET_GDR_LEVEL=7
export NCCL_SDMA_COPY_ENABLE=0
export VLLM_USE_OPT_ZEROS=1
export VLLM_USE_PD_SPLIT=1

# export VLLM_TORCH_PROFILER_DIR=/home/work/prof/
export VLLM_TORCH_PROFILER_DIR=/mnt/claw/torchprof
export VLLM_V1_USE_FUSED_QKV_SPLIT_RMS_ROPE_KVSTORE=1
export VLLM_USE_LIGHTOP=1
export LMSLIM_USE_LIGHTOP=1
export USE_FUSED_SILU_MUL_QUANT=1
export USE_FUSED_RMS_QUANT=1
export VLLM_USE_LIGHTOP_MOE_SUM_MUL_ADD=1 #moe_sum融合

# 替换lightop接口
export VLLM_USE_LIGHTOP_MOE_ALIGN=1
export VLLM_USE_LIGHTOP_FILL_MOE_ALIGN=1
export VLLM_USE_OPT_RESHAPE_AND_CACHE=1 
# chunksize设置为16384
export VLLM_USE_GLOBAL_CACHE13=1 #减少显存碎片化 
export VLLM_FUSED_MOE_CHUNK_SIZE=16384  # --max-num-batched-tokens 16384
export VLLM_USE_PIECEWISE=1
export VLLM_USE_LIGHTOP_FUSED_TOPP_TOPK=1

export VLLM_USE_OPT_OP=1 #新加变量
export VLLM_MLA_CP=1
export VLLM_MLA_CPLB=1

model_path=/mnt1/metax-tech/MiniMax-M2.5-W8A8/
model=${model_path##*/}
time=$(date "+%Y-%m-%d-%H-%M-%S")
data_type="bfloat16"
port=8000
gpu_memory=0.92

bwtype="nmz1100"
# bwtype="bw1000"
log_date=$(date "+%Y-%m-%d")
log_dir="${bwtype}_${model}/${log_date}"
mkdir -p "${log_dir}"



vllm serve ${model_path} \
 --dtype ${data_type} \
 --host 0.0.0.0 \
 --port ${port} \
 --trust-remote-code \
 -tp 4 -pp 2 \
 --gpu-memory-utilization $gpu_memory \
 --disable-log-requests \
 --max-model-len 73216 \
 --max-num-batched-tokens 16384 \
 -cc '{"pass_config": {"fuse_act_quant": false}, "cudagraph_mode": "full", "custom_ops": ["all"]}' \
 -q slimquant_marlin \
 --kv-cache-dtype fp8_e4m3 \
 --enable-prefix-caching \
 --disable-cascade-attn \
 2>&1 | tee "${log_dir}/serve_${time}.log"

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
    model="MiniMaxAI/MiniMax-Text-01",
    messages=[
        {"role": "system", "content": "你是一个专业的 AI 助手。"},
        {"role": "user", "content": "请详细分析大模型在金融领域的应用前景"},
    ],
    max_tokens=4096,
)
print(response.choices[0].message.content)
```

## DCU 适配注意

- MoE 架构：总参数 229B，但每次推理只激活约 10B，实际显存需求低于 dense 模型
- 需要 `--trust-remote-code`
- 建议使用 8x BW1100 144GB（1024GB 总显存）
- 长上下文场景 KV Cache 占用大，MoE 模型尤为明显
- 如果遇到 OOM，优先降低 `--max-model-len` 或启用 `--kv-cache-dtype fp8_e4m3`或降低显存利用率`--gpu-memory-utilization`

