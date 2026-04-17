# DCU AI Best Practices

> 在 AMD DCU（海光 DCU / Hygon DCU）上运行大语言模型与多模态模型的最优实践指南。

## 📖 简介

本仓库整理了在 DCU 硬件上部署、调优和运行 AI 模型的经验与最佳实践，涵盖：

- **大语言模型 (LLM)** — 文本生成、对话、代码补全等
- **全模态模型 (Omni)** — 文本+图像+音频统一理解与生成
- **多模态模型 (VLM)** — 视觉语言模型、图像生成、语音识别等
- **环境搭建** — ROCm 工具链、驱动安装、Python 环境配置
- **模型部署** — 推理服务、分布式部署、多卡方案
- **性能优化** — 显存优化、算子调优、量化、KV Cache 策略
- **框架适配** — vLLM (含 Omni)、SGLang、Transformers、ComfyUI 等
- **故障排查** — 常见问题、错误码、FAQ
- **性能基准** — 各模型在 DCU 上的实测数据

## 🗂 目录结构

```
dcu-llm-best-practices/
├── docs/
│   ├── getting-started.md              # 快速开始
│   ├── environment-setup.md            # 环境搭建
│   ├── model-deployment/               # 模型部署
│   │   ├── overview.md                 # 部署方案概览
│   │   ├── vllm/                       # vLLM 部署
│   │   │   ├── overview.md             # vLLM 总览
│   │   │   ├── qwen3.md                # Qwen3
│   │   │   ├── qwen3.5.md              # Qwen3.5
│   │   │   ├── glm-5.md                # GLM-5
│   │   │   ├── kimi-k2.md              # Kimi-K2
│   │   │   └── minimax-2.x.md            # MiniMax-2.x
│   │   ├── sglang/                     # SGLang 部署
│   │   │   ├── overview.md             # SGLang 总览
│   │   │   ├── qwen3.md                # Qwen3
│   │   │   ├── qwen3.5.md              # Qwen3.5
│   │   │   ├── glm-5.md                # GLM-5
│   │   │   ├── kimi-k2.md              # Kimi-K2
│   │   │   └── minimax-2.x.md            # MiniMax-2.x
│   │   └── diffusion/                  # Diffusion 模型部署
│   │       ├── overview.md             # Diffusion 总览
│   │       ├── wan2.1.md               # Wan2.1 视频生成
│   │       ├── sd3-flux.md             # SD3/FLUX/SDXL
│   │       ├── cogvideox.md            # CogVideoX
│   │       └── comfyui-dcu.md          # ComfyUI
│   ├── optimization/                   # 性能优化
│   │   ├── performance-tuning.md       # 性能调优
│   │   ├── memory-optimization.md      # 显存优化
│   │   ├── kernel-optimization.md      # 算子优化
│   │   └── quantization.md             # 量化方案
│   ├── frameworks/                     # 框架适配
│   │   ├── vllm-dcu.md                 # vLLM on DCU
│   │   ├── sglang-dcu.md               # SGLang on DCU
│   │   └── transformers-dcu.md         # Transformers on DCU
│   ├── troubleshooting/                # 故障排查
│   │   ├── common-issues.md            # 常见问题
│   │   ├── error-codes.md              # 错误码参考
│   │   └── faq.md                      # 常见问答
│   └── benchmarks/                     # 性能基准
│       ├── overview.md                 # 基准测试说明
│       └── results/                    # 测试结果
├── scripts/
│   ├── setup/                          # 环境安装脚本
│   └── examples/                       # 示例脚本
├── CONTRIBUTING.md                     # 贡献指南
└── LICENSE                            # 开源协议
```

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone http://112.11.119.99:10068/zhangqha/dcu-llm-best-practices.git
cd dcu-llm-best-practices

# 2. 查看环境搭建指南
# 详见 docs/environment-setup.md

# 3. 启动推理服务（示例）
# 详见 docs/model-deployment/vllm/overview.md
```

## 📋 支持的模型类型

### 大语言模型 (LLM)（备注：后续根据实际情况修改，AI生成)

| 模型系列 | 代表模型 | 推荐框架 |
|---------|---------|---------|
| Qwen3 | Qwen3-8B/32B/235B | vLLM / SGLang |
| Qwen3.5 | Qwen3.5-7B/14B/72B | vLLM / SGLang |
| GLM-5 | GLM-5-9B/25B/72B | vLLM / SGLang |
| Kimi-K2 | Kimi-K2-7B/13B/72B | vLLM / SGLang |
| MiniMax | MiniMax-2.x (456B MoE) | vLLM / SGLang |

### 多模态模型 (VLM)（备注：后续根据实际情况修改，AI生成)

| 模型类型 | 代表模型 | 推荐框架 |
|---------|---------|---------|
| 全模态 | Qwen2.5-Omni、UltraVox | vLLM Omni |
| 视觉语言 | Qwen2.5-VL、LLaVA、InternVL | vLLM / Transformers |
| 图像生成 | Stable Diffusion、FLUX | Diffusers / ComfyUI |
| 语音识别 | Whisper、SenseVoice | Transformers |
| 语音合成 | ChatTTS、CosyVoice | Transformers |
| 视频生成 | Wan2.1、CogVideoX | Diffusers / ComfyUI |

## 📋 硬件兼容性（备注：后续根据实际情况修改，AI生成）

| 硬件型号 | 架构 | 显存 | 支持状态 |
|---------|------|------|---------|
| K100 AI | CDNA 3 | 64GB | ✅ 已验证 |
| K200 AI | CDNA 3 | 128GB | ✅ 已验证 |
| Z100 AI | CDNA 3 | 128GB | 🔄 测试中 |

## 🤝 贡献

欢迎提交 Issue 和 PR！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 📄 License

[MIT License](LICENSE)

## ⚠️ 免责声明

本仓库内容基于实际使用经验整理，可能随 ROCm 版本更新而变化。建议结合官方文档使用。
