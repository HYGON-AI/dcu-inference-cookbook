#!/bin/bash
# DCU 环境检查脚本
# 用于验证 DCU 驱动、ROCm、PyTorch 是否正确安装

set -e

echo "================================"
echo "  DCU Environment Check"
echo "================================"
echo ""

# 1. 检查 DCU 设备
echo "[1/6] Checking DCU devices..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi
    echo "✅ rocm-smi found"
else
    echo "❌ rocm-smi not found - DCU driver may not be installed"
fi
echo ""

# 2. 检查 HIP 配置
echo "[2/6] Checking HIP config..."
if command -v hipconfig &> /dev/null; then
    hipconfig
    echo "✅ hipconfig found"
else
    echo "❌ hipconfig not found"
fi
echo ""

# 3. 检查 Python
echo "[3/6] Checking Python..."
python3 --version
echo ""

# 4. 检查 PyTorch
echo "[4/6] Checking PyTorch..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'HIP available: {torch.cuda.is_available()}')
print(f'HIP version: {torch.version.hip}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  Device {i}: {props.name}, Memory: {props.total_mem / 1024**3:.1f} GB')
" 2>&1 || echo "❌ PyTorch check failed"
echo ""

# 5. 检查 Transformers
echo "[5/6] Checking Transformers..."
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')" 2>&1 || echo "❌ Transformers not installed"
echo ""

# 6. 检查 vLLM
echo "[6/6] Checking vLLM..."
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>&1 || echo "⚠️  vLLM not installed (optional)"
echo ""

echo "================================"
echo "  Check Complete"
echo "================================"
