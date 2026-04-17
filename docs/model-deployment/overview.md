# 模型部署方案概览

## 部署架构

```
┌─────────────────────────────────────────────┐
│                  客户端请求                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            API Gateway / Load Balancer       │
└──────────────────┬──────────────────────────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│ Node 1  │  │ Node 2  │  │ Node N  │
│ DCU x8  │  │ DCU x8  │  │ DCU x8  │
└─────────┘  └─────────┘  └─────────┘
```

## 目录导航

### LLM / VLM 推理

| 框架 | 说明 | 文档 |
|------|------|------|
| vLLM | PagedAttention、高吞吐、Omni 全模态 | [vllm/](vllm/overview.md) |
| SGLang | RadixAttention、灵活调度、结构化生成 | [sglang/](sglang/overview.md) |

### 图像 / 视频生成

| 框架 | 说明 | 文档 |
|------|------|------|
| Diffusers | HuggingFace 生态，代码驱动 | [diffusion/](diffusion/overview.md) |
| ComfyUI | 节点式可视化工作流 | [diffusion/comfyui-dcu.md](diffusion/comfyui-dcu.md) |

## 推理框架对比

| 框架 | 特点 | LLM | VLM | 图像生成 | 视频生成 | DCU 适配 |
|------|------|-----|-----|---------|---------|---------|
| vLLM | PagedAttention、高吞吐、Omni 全模态 | ✅ | ✅ | ❌ | ❌ | ✅ 良好 |
| vLLM Omni | 统一多模态 API（文本+图像+音频） | ✅ | ✅ | ❌ | ❌ | ✅ 良好 |
| SGLang | RadixAttention、灵活调度 | ✅ | ✅ | ❌ | ❌ | ✅ 良好 |
| Transformers | 原生支持、灵活 | ✅ | ✅ | ❌ | ❌ | ✅ 良好 |
| Diffusers | 图像/视频生成生态 | ❌ | ❌ | ✅ | ✅ | ✅ 良好 |
| ComfyUI | 节点式工作流、可视化 | ❌ | ❌ | ✅ | ✅ | ✅ 良好 |
| llama.cpp | CPU+GPU 混合推理 | ✅ | ❌ | ❌ | ❌ | 🔄 适配中 |

## 支持的模型

### vLLM / SGLang（LLM & VLM）

| 模型 | 参数量 | 类型 | vLLM | SGLang |
|------|--------|------|------|--------|
| Qwen3 | 0.6B ~ 235B | LLM | ✅ | ✅ |
| Qwen3.5 | 7B ~ 72B+ | LLM | ✅ | ✅ |
| GLM-5 | 9B ~ 130B+ | LLM | ✅ | ✅ |
| Kimi-K2 | 1.5B ~ 72B | LLM | ✅ | ✅ |
| MiniMax-2.x | 456B (MoE) | LLM | ✅ | ✅ |
| Qwen2.5-VL | 3B ~ 72B | VLM | ✅ | ✅ |

详见各框架目录下的模型文档。

### Diffusion（图像 & 视频）

| 模型 | 类型 | Diffusers | ComfyUI |
|------|------|-----------|---------|
| Wan2.1 | 视频生成 | ✅ | ✅ |
| FLUX.1 | 图像生成 | ✅ | ✅ |
| SD3 / SDXL | 图像生成 | ✅ | ✅ |
| ControlNet | 条件生成 | ✅ | ✅ |

详见 → [diffusion/](diffusion/overview.md)

## 模型规模与硬件匹配

### 大语言模型

| 模型规模 | 推荐显存 | 单卡方案 | 多卡方案 |
|---------|---------|---------|---------|
| 1.5B | 4GB+ | 1x DCU 64GB | - |
| 7B | 16GB+ | 1x DCU 64GB | - |
| 14B | 32GB+ | 1x DCU 64GB | - |
| 32B | 64GB+ | 1x DCU 128GB | 2x DCU TP |
| 72B | 128GB+ | - | 2x DCU 128GB TP |
| 72B | 128GB+ | - | 4x DCU 64GB TP |
| 110B+ | 256GB+ | - | 4x DCU 128GB TP |
| 456B (MoE) | 200GB+ | - | 4x DCU 128GB TP |

### 多模态模型

| 模型 | 类型 | 推荐显存 | 方案 |
|------|------|---------|------|
| Qwen2.5-VL-7B | 视觉语言 | 20GB+ | 1x DCU 64GB |
| InternVL2-26B | 视觉语言 | 56GB+ | 1x DCU 64GB |
| Wan2.1-14B | 视频生成 | 32GB+ | 1x DCU 64GB |
| Wan2.1-1.3B | 视频生成 | 8GB+ | 1x DCU 64GB |
| FLUX.1-dev | 图像生成 | 24GB+ | 1x DCU 64GB |

## 部署模式

### 在线推理 (Online Serving)
- 低延迟、高并发
- 推荐: vLLM / SGLang
- 详见 → [vllm/](vllm/overview.md) / [sglang/](sglang/overview.md)

### 离线推理 (Offline Batch)
- 高吞吐、批处理
- 推荐: Transformers / Diffusers + 批处理脚本

### 多模态推理 (Multimodal)
- 图像理解、图像生成、语音处理、视频生成
- 推荐: vLLM Omni (全模态) / Diffusers (图像/视频) / ComfyUI (可视化工作流)
- 详见 → [diffusion/](diffusion/overview.md)
