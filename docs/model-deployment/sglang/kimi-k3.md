# Kimi-K3 on SGLang

## 模型简介

Kimi-K3 是 Moonshot AI 推出的 Kimi 系列模型，面向长上下文、工具调用和推理场景。本页提供基于 SGLang 的 BW1100 部署命令。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K3](https://www.modelscope.cn/models/moonshotai/Kimi-K3) | BF16 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-ifb-bw1100-16x-sglang-0518) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.5.18 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-sglang-0518) |

## 启动命令

### Kimi-K3 IFB BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### Node 0

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1

sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 0 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```

#### Node 1

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1

sglang serve \
  --trust-remote-code \
  --model-path moonshotai/Kimi-K3 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 1 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```

### kimi-k3-INT4 IFB BW1100 16x SGLang 0.5.18

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### Node 0

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 0 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```

#### Node 1

```bash
export SGLANG_K3_ATTN_RESIDUAL_HCU=1
export SGLANG_KDA_USE_HCU_OP=1
export SGLANG_USE_INT4_W4A8=1
export SGLANG_USE_LIGHTOP_W4A8_MARLIN_MOE=0

sglang serve \
  --trust-remote-code \
  --model-path hygon/kimi-k3-INT4 \
  --tp-size 16 \
  --port 11010 \
  --dist-init-addr <ip:端口> \
  --nnodes 2 \
  --node-rank 1 \
  --attention-backend hcu_mla \
  --linear-attn-backend triton \
  --page-size 64 \
  --linear-attn-prefill-backend flashkda \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3 \
  --mem-fraction-static 0.93 \
  --host 0.0.0.0 \
  --mm-attention-backend fa3 \
  --kv-cache-dtype fp8_e4m3 \
  --max-running-requests 128 \
  --mamba-full-memory-ratio 0.1 \
  --mamba-ssm-dtype bfloat16 \
  --load-format fastsafetensors \
  --disable-radix-cache
```
