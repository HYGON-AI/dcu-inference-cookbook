# DCU Inference Cookbook

## 📖 简介

本仓库整理了在 DCU 硬件上部署、调优和运行 AI 模型的经验与最佳实践，涵盖：

- **大语言模型 (LLM)** — 文本生成、对话、代码补全等
- **全模态模型 (Omni)** — 文本+图像+音频统一理解与生成
- **多模态模型 (VLM)** — 视觉语言模型、图像生成、语音识别等
- **环境搭建** — DTK 工具链、驱动安装、Python 环境配置
- **模型部署** — 推理服务、分布式部署、多卡方案
- **性能优化** — 显存优化、算子调优、量化、KV Cache 策略
- **框架适配** — vLLM (含 Omni)、SGLang、Transformers、ComfyUI 等
- **故障排查** — 常见问题、错误码、FAQ
- **性能基准** — 各模型在 DCU 上的实测数据

## 📋 模型列表

<table>
  <thead>
    <tr>
      <th>模型</th>
      <th>框架</th>
      <th align="center">K100_AI</th>
      <th align="center">BW1000</th>
      <th align="center">BW1100</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Qwen3</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3.5</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-V3</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/deepseek-v3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-5</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/glm-5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/glm-5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Kimi-K2</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/kimi-k2.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/kimi-k2.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Kimi-K2.5</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/kimi-k2.5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/kimi-k2.5.md">✅</a></td>
    </tr>
    <tr>
      <td>MiMo-V2-Flash</td>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/mimo-v2-flash.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">MiniMax-M2.5</td>
      <td>vLLM</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td>
      <td align="center"><a href="docs/model-deployment/sglang/minimax-m2.5.md">✅</a></td>
      <td align="center"><a href="docs/model-deployment/sglang/minimax-m2.5.md">✅</a></td>
    </tr>
  </tbody>
</table>

## 🤝 贡献

欢迎提交 Issue 和 PR！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
