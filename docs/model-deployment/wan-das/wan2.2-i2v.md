# Wan2.2-I2V on wan-das

## 模型简介

Wan2.2-I2V-A14B 是阿里通义实验室推出的图生视频（Image-to-Video）模型，基于 DiT 架构，可根据图像提示词生成高质量视频。

本文档给出 BW1000/BW1100 8 卡离线推理最佳实践，适用于 1280x720、1280x704、832x480、768x432的 Wan2.2-I2V-A14B 推理场景。



## 模型列表

| 模型权重                                                     | 量化方式 | 推荐硬件      | 卡数 | 部署方式 | 启动命令                                                     |
| ------------------------------------------------------------ | -------- | ------------- | ---- | -------- | ------------------------------------------------------------ |
| [Wan-AI/Wan2.2-I2V-A14B](https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B) | BF16     | BW1000/BW1100 | 8x   | Offline  | [启动命令](#Wan22-I2V-A14B-BF16) |
| [hygon/Wan2.2-I2V-A14B-INT8](https://www.modelscope.cn/models/hygon/Wan2.2-I2V-A14B-INT8-w8a8) | INT8     | BW1000/BW1100 | 8x   | Offline  | [启动命令](#Wan22-I2V-A14B-INT8) |



## 代码获取

```bash
git clone https://developer.sourcefind.cn/codes/OpenDAS/wan-das.git
```



## 启动命令



### Wan2.2-I2V-A14B BF16

```bash
export model_base=./models/Wan2.2-I2V-A14B/

export OMP_NUM_THREADS=32
export HIPBLASLT_ALLOW_TF32=1
export AllTOAll_STREAM_WITH_COMPUTE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=True
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1

export GLOG_minloglevel=3

torchrun --nproc_per_node=8 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 61 \
--sample_steps 4 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--sample_solver euler \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0
```



### Wan2.2-I2V-A14B INT8

```bash
export model_base=./models/Wan2.2-I2V-A14B-INT8-w8a8/

export OMP_NUM_THREADS=32
export HIPBLASLT_ALLOW_TF32=1
export AllTOAll_STREAM_WITH_COMPUTE=1
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=True
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1

export GLOG_minloglevel=3

torchrun --nproc_per_node=8 generate.py \
--task i2v-A14B \
--ckpt_dir ${model_base} \
--size 1280*720 \
--frame_num 61 \
--sample_steps 4 \
--dit_fsdp \
--t5_fsdp \
--cfg_size 1 \
--ulysses_size 8 \
--sample_solver euler \
--image examples/i2v_input.JPG \
--prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." \
--base_seed 0 \
--use_int8 1 \
```



## 性能优化

### GEMM INT8 量化

```
export model_base=./models/Wan2.2-I2V-A14B-INT8-w8a8/
--use_int8 1
```



### attn qkv投影矩阵合并、crossattn kv投影矩阵合并优化

```
--apply_attn_proj_fusion 1
```



### 使用SLA稀疏化

```
--use_sla 1 \
--sparse_attn_topk 0.4
```



### 使用SageAttn，仅在BW1100上支持

```
--use_sageattn 1
```