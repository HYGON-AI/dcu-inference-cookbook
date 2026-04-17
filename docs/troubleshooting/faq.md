# FAQ

## 基础问题

### DCU 和 GPU 有什么区别？

DCU（Deep Computing Unit）是 AMD 推出的加速处理器，基于 CDNA 架构。与 NVIDIA GPU 的主要区别：

- **编程模型**: 使用 HIP（Heterogeneous-compute Interface for Portability），兼容 CUDA API
- **软件栈**: ROCm 替代 CUDA Toolkit
- **生态**: 部分框架需要适配，主流框架（PyTorch、vLLM、Diffusers 等）已支持

### DCU 能直接运行 CUDA 代码吗？

大部分 CUDA 代码可以通过 HIP 兼容层运行，但：
- 部分 CUDA 专属算子需要手动适配
- 第三方 CUDA 库（如 cuDNN）需要替换为 ROCm 对应库（MIOpen）
- 复杂的自定义 CUDA kernel 可能需要修改

### 如何选择推理框架？

| 场景 | 推荐 |
|------|------|
| LLM 高并发在线服务 | vLLM / SGLang |
| VLM 图像理解服务 | vLLM / SGLang |
| 图像生成 | Diffusers |
| 语音识别/合成 | Transformers + FastAPI |
| 开发调试 | Transformers / Diffusers |
| 极致显存优化 | llama.cpp (GGUF) |

## 性能问题

### DCU 推理性能比 GPU 差多少？

取决于具体场景和模型：
- **LLM BF16 推理**: 通常为同级别 GPU 的 70-90%
- **VLM 推理**: 视觉编码器部分差距较小，整体约 70-85%
- **图像生成**: Diffusers 在 DCU 上运行良好，约 75-90%
- **INT8/INT4 量化**: 差距可能更大，部分算子优化仍在进行

### 如何最大化 DCU 利用率？

1. 使用 bf16 / fp16 数据类型
2. 合理配置 tensor-parallel
3. 启用 continuous batching（LLM/VLM）
4. 使用 PagedAttention
5. 预热 MIOpen kernel 缓存
6. 图像生成启用 VAE tiling/slicing

## 模型支持

### 支持哪些大语言模型？

主流开源大模型均已验证：
- Qwen 系列（Qwen2、Qwen2.5）
- LLaMA 系列（LLaMA 2/3）
- ChatGLM 系列
- Baichuan 系列
- DeepSeek 系列

### 支持哪些多模态模型？

- **视觉语言**: Qwen2.5-VL、InternVL2、LLaVA、GLM-4V
- **图像生成**: Stable Diffusion 3、FLUX、SDXL、ControlNet
- **语音识别**: Whisper、SenseVoice、Paraformer
- **语音合成**: ChatTTS、CosyVoice
- **视频生成**: CogVideoX

### 如何处理超长上下文？

```bash
# vLLM
--max-model-len 32768

# 注意：长上下文会显著增加显存占用
# 72B 模型 32K 上下文约需 400GB+ 显存
```

### VLM 能处理多大的图片？

- 取决于模型和显存
- Qwen2.5-VL 支持动态分辨率，最大约 1344x1344（单卡 64GB）
- 更高分辨率需要更多显存或多卡 TP
- 建议不超过 2048x2048
