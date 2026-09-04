# Kimi-K3 on vLLM

## 模型简介

Kimi-K3 是 Moonshot AI 推出的 Kimi 系列模型，面向长上下文、工具调用和推理场景。本页提供基于 vLLM 的 BW1100 部署命令。

## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [moonshotai/Kimi-K3](https://www.modelscope.cn/models/moonshotai/Kimi-K3) | BF16 | 0.26 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-ifb-bw1100-16x-vllm-026) |
| [hygon/kimi-k3-INT4](https://www.modelscope.cn/models/hygon/kimi-k3-INT4) | INT4 W4A8 | 0.26 | BW1100 | 16 | IFB | [**`>_`**](#kimi-k3-int4-ifb-bw1100-16x-vllm-026) |

## 启动命令

### Kimi-K3 IFB BW1100 16x vLLM 0.26

#### Node 0

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ip \
  --master-port 端口 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

#### Node 1

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve moonshotai/Kimi-K3 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ip \
  --master-port 端口 \
  --headless \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

### kimi-k3-INT4 IFB BW1100 16x vLLM 0.26

#### Node 0

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ip \
  --master-port 端口 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```

#### Node 1

```bash
export VLLM_KIMI_CONV1D_BACKEND=1
export VLLM_KIMI_FLASHKDA_BACKEND=1
export VLLM_USE_FLASHMLA=1

vllm serve hygon/kimi-k3-INT4 \
  --trust-remote-code \
  --moe-backend auto \
  -cc.pass_config.fuse_allreduce_rms=False \
  --tensor-parallel-size 16 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ip \
  --master-port 端口 \
  --headless \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.95 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"image": 0}' \
  --max-num-seqs 128 \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser kimi_k3 \
  --reasoning-parser kimi_k3
```
