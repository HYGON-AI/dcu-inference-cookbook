# 常见问题

## 环境相关

### IB网卡

配置的网卡需要满足：
1.ibstat查询状态为Active
2.未用于存储

举例环境网卡状态：
1.存储网卡：mlx5_0,mlx5_1
2.状态Down网卡：mlx5_6,mlx5_7
3.状态Active网卡：mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

有效设备：mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

最终参数：--disaggregation-ib-device mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_8,mlx5_9

### Q: `torch.cuda.is_available()` 返回 False

**排查步骤：**

```bash
# 1. 检查 DCU 设备是否被识别
rocm-smi
# 或
hipconfig

# 2. 检查驱动版本
cat /opt/rocm/.info/version

# 3. 检查 HIP 可见设备
echo $HIP_VISIBLE_DEVICES

# 4. 检查 PyTorch 是否为 ROCm 版本
python -c "import torch; print(torch.version.hip)"
```

**常见原因：**
- 驱动未正确安装
- HIP_VISIBLE_DEVICES 设置错误
- PyTorch 安装了 CUDA 版本而非 ROCm 版本

### Q: ImportError: libamdhip64.so

```bash
# 添加 ROCm 库路径
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
```

## LLM 推理相关

### Q: OOM (Out of Memory)

**解决方案（按优先级）：**

1. 减小 `--max-model-len`
2. 降低 `--gpu-memory-utilization`（或提高以充分利用）
3. 使用量化模型（INT4/INT8）
4. 增加 `--tensor-parallel-size`
5. 启用 KV Cache 量化

### Q: 推理速度异常慢

```bash
# 1. 确认数据类型
# 应使用 bf16，而非 fp32

# 2. 检查 DCU 利用率
rocm-smi

# 3. 确认 tensor-parallel 配置合理

# 4. 检查是否有 CPU 瓶颈
htop
```

### Q: 生成结果质量下降

- 确认使用 bf16 而非 INT4（除非显存不足）
- 检查 `temperature` 和 `top_p` 参数
- 确认模型权重完整加载

## VLM 推理相关

### Q: VLM 处理图片时报错

```bash
# 1. 检查图片 URL 是否可访问
curl -I https://example.com/photo.jpg

# 2. 检查图片大小限制
# vLLM 默认限制单请求图片数，通过 --limit-mm-per-prompt 调整

# 3. 检查图片格式
# 支持 JPEG、PNG、WebP，不支持 GIF 动图
```

### Q: VLM 显存不足

- 降低输入图片分辨率
- 减少 `--limit-mm-per-prompt` 数值
- 使用量化 VLM 模型
- 增加 tensor-parallel 卡数

## 图像生成相关

### Q: Diffusers 报 NotImplementedError

- 部分自定义 CUDA kernel 未适配 ROCm
- 尝试禁用特定优化选项（如 xformers）
- 检查 Diffusers 版本是否支持 ROCm

### Q: 图像生成 OOM

```python
# 启用 VAE tiling
pipe.vae.enable_tiling()

# 启用 VAE slicing
pipe.vae.enable_slicing()

# 降低分辨率
# 或使用 CPU offload
pipe.enable_model_cpu_offload()
```

### Q: 图像生成速度慢

```python
# 降低推理步数
image = pipe(prompt, num_inference_steps=20)  # 默认 50

# 使用 torch.compile
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead")
```
