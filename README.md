<div align="center">

![DCU Inference Cookbook Logo](./assets/logo.svg)

</div>


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
  <tr>
    <td align="center"><b>厂商</b></td>
    <td><b>模型</b></td>
    <td><b>框架</b></td>
    <td align="center"><b>K100_AI</b></td>
    <td align="center"><b>BW1000</b></td>
    <td align="center"><b>BW1100</b></td>
  </tr>
  <tbody>
    <tr>
      <td rowspan="18" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png" height="40"/><br/>Qwen</td>
      <td rowspan="2">Qwen3.6</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3.5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3-VL</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center"><a href="docs/model-deployment/vllm/qwen3-vl.md">✅</a></td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3-Next</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3-Coder-Next</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3-Coder</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Qwen2.5-VL</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">QwQ</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="10" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6538815d1bdb3c40db94fbfa/xMBly9PUMphrFVMxLX4kq.png" height="40"/><br/>DeepSeek</td>
      <td rowspan="2">DeepSeek-V4</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-V3.2</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/deepseek-v3.2.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/deepseek-v3.2.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-V3.1</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-V3</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/deepseek-v3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">DeepSeek-R1</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/deepseek-r1.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="4" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/646cf8084eefb026fb8fd8bc/oCTqufkdTkjyGodsx1vo1.png" height="40"/><br/>Meta Llama</td>
      <td rowspan="2">Llama-3.3</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Llama-3.1</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="14" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62dc173789b4cf157d36ebee/i_pxzM2ZDo3Ub-BEgIkE9.png" height="40"/><br/>Z.ai</td>
      <td rowspan="2">GLM-5.1</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/glm-5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/glm-5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.7</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.6V</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.6</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.5V</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/641c1e77c3983aa9490f8121/X1yT2rsaIbR9cdYGEVu0X.jpeg" height="40"/><br/>Moonshot AI</td>
      <td rowspan="2">Kimi-K2.6</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">Kimi-K2.5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/kimi-k2.5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/kimi-k2.5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Kimi-K2</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/kimi-k2.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/kimi-k2.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/676e38ad04af5bec20bc9faf/dUd-LsZEX0H_d4qefO_g6.jpeg" height="40"/><br/>MiniMax</td>
      <td rowspan="2">MiniMax-M2.7</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">MiniMax-M2.5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/minimax-m2.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/minimax-m2.5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">MiniMax-M2</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/644f7e6233ac8f46fa0b9e26/CmF2ocXhkr2UtHXgmwq7-.png" height="40"/><br/>StepFun</td>
      <td rowspan="2">Step-3.5</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/662e1f9da266499277937d33/fyKuazRifqiaIO34xrhhm.jpeg" height="40"/><br/>inclusionAI</td>
      <td rowspan="2">Ling-1T</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="4" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680cb7d1233834890a64acee/5w_4aLfF-7MAyaIPOV498.jpeg" height="40"/><br/>Xiaomi MiMo</td>
      <td rowspan="2">MiMo-V2.5</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">MiMo-V2-Flash</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/mimo-v2-flash.md">✅</a></td>
    </tr>
  </tbody>
</table>

## 🤝 贡献

欢迎提交 Issue 和 PR！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
