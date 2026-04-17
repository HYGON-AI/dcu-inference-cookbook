# 环境搭建

本指南介绍在 DCU 上搭建大模型运行环境的完整流程。

## 系统要求

- **操作系统**: Kylin V10 / UOS 20 / CentOS 7.9+
- **内核版本**: 5.10+
- **DCU 驱动**: ROCm 5.x / 6.x（视硬件型号而定）
- **Python**: 3.10 - 3.12
- **CUDA 兼容**: 通过 HIPI 工具链提供 CUDA 兼容层

## 安装步骤

### 1. 安装 DCU 驱动

```bash
# 检查 DCU 设备是否识别
lspci | grep -i AMD
# 或
hipconfig
```

### 2. 安装 ROCm 工具链

```bash
# 参考 ROCm 官方文档安装对应版本
# https://rocm.docs.amd.com/
```

### 3. 配置 Python 环境

```bash
# 推荐使用 conda
conda create -n llm-dcu python=3.10
conda activate llm-dcu

# 安装 PyTorch (DCU 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### 4. 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
python -c "import torch; print(torch.cuda.device_count())"
# 应输出 DCU 卡数
```

## 常见问题

详见 [troubleshooting/common-issues.md](troubleshooting/common-issues.md)
