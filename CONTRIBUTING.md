# 贡献指南

感谢你对本仓库的关注！以下是贡献指南。

## 贡献方式

### 📝 文档贡献

最简单的贡献方式：

1. Fork 本仓库
2. 创建分支：`git checkout -b docs/your-topic`
3. 编写或修改文档
4. 提交 PR

### 🐛 问题报告

发现错误或有改进建议？请提交 Issue，包含：

- 问题描述
- 复现步骤
- 期望行为
- 实际行为
- 环境信息（ROCm 版本、硬件型号等）

### 📊 基准测试

提交你的测试数据：

1. 在 `docs/benchmarks/results/` 下创建结果文件
2. 包含完整测试环境信息
3. 使用统一的表格格式

### 🔧 脚本贡献

提交实用脚本到 `scripts/` 目录：

- 环境安装脚本 → `scripts/setup/`
- 示例脚本 → `scripts/examples/`

## 文档规范

### Markdown 格式

- 使用中文撰写
- 代码块标注语言类型
- 表格对齐整齐
- 链接使用相对路径

### 目录结构

- 新增文档放在对应子目录下
- 更新 README.md 中的目录结构
- 添加必要的交叉引用

### "最佳实践" 文档撰写规范

> 参考示例：[docs/model-deployment/sglang/kimi-k2.5.md](docs/model-deployment/sglang/kimi-k2.5.md)

**核心标准：别人复制你的命令，不需要问任何问题就能跑起来，并得到接近的结果。**

**❌ 不要这样做：**

- 不要定义额外的 shell 变量（除了必要的 vllm / sglang 环境变量），否则读者需要理解变量定义才能使用命令
- 不要保留注释掉的代码，会干扰读者判断哪些命令需要执行

**✅ 应该这样做：**

- 使用官方模型名称，例如 `meta-llama/Llama-3-8B-Instruct`，因为每个人的本地模型路径各不相同
- 使用框架默认端口，不要自定义 `--port`：vLLM 默认 `8000`，SGLang 默认 `30000`
- 提供 `curl` 调用示例，让读者可以立即验证服务是否正常，例如：

  ```bash
  curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3-8B-Instruct", "prompt": "Hello", "max_tokens": 32}'
  ```

## PR 规范

- 标题清晰描述改动内容
- 单个 PR 聚焦一个主题
- 保持最小化改动
- 通过预览检查格式

## License

贡献的内容遵循 [MIT License](../LICENSE)。
