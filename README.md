<div align="center">

![inference-cookbook-das Logo](./assets/logo.svg)

</div>


## 📖 简介

本仓库整理了在 HCU 硬件上部署、调优和运行 AI 模型的经验与最佳实践，涵盖：

- **大语言模型 (LLM)** — 文本生成、对话、代码补全等
- **多模态模型 (VLM)** — 视觉语言模型、图像生成、语音识别等
- **全模态模型 (Omni)** — 文本+图像+音频统一理解与生成

## 📋 模型列表

✅ 已验证 &nbsp;|&nbsp; 🚧 开发中 &nbsp;|&nbsp; `-` 暂未验证

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
      <td rowspan="20" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/620760a26e3b7210c2ff1943/-s1gyJfvbE1RgO5iBeNOi.png" height="40"/><br/>Qwen</td>
      <td rowspan="2">Qwen3.8</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/qwen3.8.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.8.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3.6</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.6.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.6.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.6.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.6.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.6.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.6.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3.5</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.5.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.5.md">✅</a></td>
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
      <td rowspan="2">Qwen3-Coder</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen3</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/qwen3.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">Qwen2.5-VL</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center"><a href="docs/model-deployment/vllm/qwen2.5-vl.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwen2.5-vl.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">QwQ</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/qwq.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwq.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/qwq.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen2-VL</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/qwen2-vl.md">✅</a></td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="2">Qwen2-72B</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/qwen2.md">✅</a></td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td rowspan="10" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/6538815d1bdb3c40db94fbfa/xMBly9PUMphrFVMxLX4kq.png" height="40"/><br/>DeepSeek</td>
      <td rowspan="2">DeepSeek-V4</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center"><a href="docs/model-deployment/sglang/deepseek-v4.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/deepseek-v4.md">✅</a></td>
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
      <td align="center"><a href="docs/model-deployment/vllm/deepseek-r1.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/deepseek-r1.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/deepseek-r1.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center"><a href="docs/model-deployment/sglang/deepseek-r1.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/deepseek-r1.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/deepseek-r1.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="8" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/62dc173789b4cf157d36ebee/i_pxzM2ZDo3Ub-BEgIkE9.png" height="40"/><br/>Z.ai</td>
      <td rowspan="2">GLM-5.2</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center"><a href="docs/model-deployment/vllm/glm-5.2.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/glm-5.2.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2">GLM-5.1</td>
      <td>vLLM</td>
      <td align="center">🚧</td><td align="center"><a href="docs/model-deployment/vllm/glm-5.1.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/glm-5.1.md">✅</a></td>
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
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/glm-5.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/glm-5.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">GLM-4.7</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/glm4.7.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/glm4.7.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/glm4.7.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">🚧</td><td align="center">🚧</td><td align="center">🚧</td>
    </tr>
    <tr>
      <td rowspan="2" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/5dd96eb166059660ed1ee413/Lp3m-XLpjQGwBItlvn69q.png" height="40"/><br/>Tencent</td>
      <td rowspan="2">Hy3</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/hy3.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/hunyuan-v3.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="6" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/641c1e77c3983aa9490f8121/X1yT2rsaIbR9cdYGEVu0X.jpeg" height="40"/><br/>Moonshot AI</td>
      <td rowspan="2">Kimi-K2.6</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/kimi-k2.6.md">✅</a></td>
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
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/minimax-m2.7.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">MiniMax-M2.5</td>
      <td>vLLM</td>
      <td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/minimax-2.x.md">✅</a></td>
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
      <td rowspan="4" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/680cb7d1233834890a64acee/5w_4aLfF-7MAyaIPOV498.jpeg" height="40"/><br/>Xiaomi MiMo</td>
      <td rowspan="2">MiMo-V2.5-Pro</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/mimo-v2.5-pro.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2">MiMo-V2-Flash</td>
      <td>vLLM</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/mimo-v2-flash.md">✅</a></td><td align="center"><a href="docs/model-deployment/sglang/mimo-v2-flash.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="12" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/67b610677ea7952def8b29c6/N6jQbbeaa_FcUY-wI1dgG.png" height="40"/><br/>Wan</td>
      <td rowspan="3">Wan2.1-I2V</td>
      <td>vLLM-Omni</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>Wan-DAS</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/wan-das/wan2.1-i2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/wan-das/wan2.1-i2v.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="3">Wan2.2-I2V</td>
      <td>vLLM-Omni</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/wan2.2-i2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/wan2.2-i2v.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>Wan-DAS</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-i2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-i2v.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="3">Wan2.2-T2V</td>
      <td>vLLM-Omni</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/wan2.2-t2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/vllm/wan2.2-t2v.md">✅</a></td>
    </tr>
    <tr>
      <td>SGLang Diffusion</td>
      <td align="center">-</td><td align="center">-</td><td align="center"><a href="docs/model-deployment/sglang/wan2.2-t2v.md">✅</a></td>
    </tr>
    <tr>
      <td>Wan-DAS</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-t2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-t2v.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="3">Wan2.2-TI2V</td>
      <td>vLLM-Omni</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>SGLang Diffusion</td>
      <td align="center">-</td><td align="center">-</td><td align="center">-</td>
    </tr>
    <tr>
      <td>Wan-DAS</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-ti2v.md">✅</a></td><td align="center"><a href="docs/model-deployment/wan-das/wan2.2-ti2v.md">✅</a></td>
    </tr>
    <tr>
      <td rowspan="2" align="center"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/1664511063789-632c234f42c386ebd2710434.png" height="40"/><br/>BAAI</td>
      <td rowspan="2">BGE</td>
      <td>Infinity</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/infinity/bge-reranker-v2-minicpm-layerwise.md">✅</a></td><td align="center">-</td>
    </tr>
    <tr>
      <td>vLLM</td>
      <td align="center">-</td><td align="center"><a href="docs/model-deployment/vllm/bge-reranker.md">✅</a></td><td align="center">-</td>
    </tr>
  </tbody>
</table>

## 📄 许可证与第三方来源

本项目采用 [MIT License](LICENSE)。

仓库不直接内嵌第三方源码。文档中引用的模型、推理框架、工具和服务仍由各自项目的许可证约束，具体说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 🤝 贡献

欢迎提交 Issue 和 PR！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

