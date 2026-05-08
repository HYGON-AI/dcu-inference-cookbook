# 贡献指南

感谢你对本仓库的关注！以下是贡献指南。

## 贡献方式

### 📝 文档贡献(PR)

最简单的贡献方式：

1. Fork 本仓库
2. 创建分支：`git checkout -b docs/your-topic`
3. 编写或修改文档
4. 提交 PR

### 🐛 问题报告(Issue)

发现错误或有改进建议？请提交 Issue，包含：

- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（DTK 版本、硬件型号等）

## 文档规范

> 参考示例：[docs/model-deployment/sglang/kimi-k2.5.md](docs/model-deployment/sglang/kimi-k2.5.md)

**核心标准：别人复制你的命令，不需要问任何问题就能跑起来，并得到接近的结果。**

**❌ 不要这样做：**

- 不要定义额外的 shell 变量（除了必要的 vllm / sglang 环境变量），否则读者需要理解变量定义才能使用命令
- 不要保留注释掉的代码，会干扰读者判断哪些命令需要执行

**✅ 应该这样做：**

- 使用官方模型名称，例如 `meta-llama/Llama-3-8B-Instruct`，因为每个人的本地模型路径各不相同
- 量化模型使用 ModelScope 上 Hygon 官方量化的 channelwise 模型，例如 `hygon/GLM-5-Channel-INT8-w8a8`
- 使用框架默认端口，不要自定义 `--port`：vLLM 默认 `8000`，SGLang 默认 `30000`
- SGLang 使用 `sglang serve` 启动服务，不要使用已废弃的 `python -m sglang.launch_server`
- 仅针对以下三种 DCU 型号撰写适配文档：`K100_AI`、`BW1000`、`BW1100`，不要添加其他型号
- 提供 `curl` 调用示例，让读者可以立即验证服务是否正常，例如：

  ```bash
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3-8B-Instruct", "prompt": "Hello", "max_tokens": 32}'
  ```
