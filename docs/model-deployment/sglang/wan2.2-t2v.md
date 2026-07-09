# Wan2.2-T2V on SGLang Diffusion

## 模型简介

Wan2.2-T2V-A14B 是阿里通义实验室推出的文生视频（Text-to-Video）模型，基于 DiT 架构，可根据文本提示词生成高质量视频。SGLang Diffusion 在 DCU 平台上支持通过序列并行、SLA 注意力后端和 Cache-DiT 对 Wan2.2-T2V 推理进行加速。

本文档给出 BW1100（NMZ）4 卡在线推理最佳实践，适用于 1280x720、81 帧、40 步的 Wan2.2-T2V-A14B-Diffusers 推理场景。若使用 BW1000 或显存较小的节点，建议开启 FSDP 或 text encoder CPU offload 作为显存兜底，但性能会低于本文推荐配置。

## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [Wan-AI/Wan2.2-T2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers) | BF16 | 0.5.10 | BW1100 | 4x | Online | [启动命令](#wan22-t2v-a14b-diffusers-online-bw1100-4x-sglang-0510) |

## 启动命令

### Wan2.2-T2V-A14B-Diffusers Online BW1100 4x SGLang 0.5.10

以下命令保留 BW1100 硬件队列性能配置，并启用 Cache-DiT，可降低 Wan2.2 双 transformer denoise 阶段耗时。

```bash
export GPU_MAX_HW_QUEUES=3
export SGLANG_CACHE_DIT_ENABLED=true
export SGLANG_CACHE_DIT_RDT=0.24
export SGLANG_CACHE_DIT_MC=3

sglang serve \
  --model-type diffusion \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --host 0.0.0.0 \
  --pin-cpu-memory \
  --num-gpus 4 \
  --attention-backend sla_attn \
  --attention-backend-config sla_topk=0.25 \
  --dit-cpu-offload false \
  --text-encoder-cpu-offload false \
  --vae-cpu-offload false \
  --dit-layerwise-offload false \
  --use-fsdp-inference false \
  --ulysses-degree 4 \
  --ring-degree 1 \
  --warmup \
  --warmup-resolutions 1280x720 \
  --warmup-steps 1 \
  --output-path outputs
```

## API 调用

### Online Inference

```bash
curl -sS -X POST "http://localhost:30000/v1/videos" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "prompt": "A curious raccoon",
    "size": "1280x720",
    "num_frames": 81,
    "fps": 16,
    "num_inference_steps": 40,
    "seed": 42,
    "guidance_scale": 4.0,
    "embedded_guidance_scale": 6.0,
    "save_output": true,
    "output_path": "outputs",
    "output_file_name": "wan22_t2v_sglang.mp4"
  }'
```

返回结果中会包含 video id。可以通过如下接口轮询任务状态：

```bash
curl -sS "http://localhost:30000/v1/videos/<video-id>"
```

当状态为 `completed` 时，返回结果中的文件路径字段指向生成的视频文件。

## 性能参考

以下数据在 BW1100（NMZ）4 卡节点上验证，输入为 `1280x720`、`81` 帧、`16` FPS、`40` inference steps，提示词为 `A curious raccoon`。

| 配置 | Pixel 总耗时 | Denoise 耗时 | 单步耗时 | Decode 耗时 | API 推理耗时 |
| ---- | ------------ | ------------ | -------- | ----------- | -------------------- |
| SLA + Cache-DiT，FSDP off，text encoder offload off | 214.66s | 199.35s | 4.98s/step | 13.46s | 213.04s |

历史同类配置在 BW1100（NMZ）节点上可达到约 `201s` Pixel 总耗时。不同镜像版本、模型挂载存储、节点负载和首次加载状态会带来小幅波动，建议以 warmup 后的第二次及后续请求作为稳定性能参考。

## 关键参数说明

- SLA 注意力后端：Wan2.2-T2V 在 DCU 上加速的核心开关，对应启动命令中的 attention backend。
- SLA topk 0.25：推荐的质量与性能折中配置；更低 topk 可能继续提速，但需要额外做视频质量验证。
- Cache-DiT：Wan2.2 双 transformer 会分别启用缓存优化，可降低 denoise 阶段耗时。
- `--use-fsdp-inference false`：BW1100（NMZ）显存充足时推荐关闭 FSDP，可获得更低延迟。
- `--text-encoder-cpu-offload false`：将 text encoder 保持在卡上，减少在线请求时的 CPU/GPU 搬运开销。
- `--ulysses-degree 4 --ring-degree 1`：4 卡序列并行推荐配置。

## 显存不足时的兜底配置

若在 BW1000 或其他显存较小节点上出现 OOM，可优先尝试以下两项：

```bash
--use-fsdp-inference true \
--text-encoder-cpu-offload true
```

该配置能降低显存压力，但 text encoder offload 和 FSDP 会引入额外开销。实测在显存较小的 4 卡节点上，兜底配置可完成推理，但 Pixel 总耗时会高于 BW1100 推荐配置。

## 日志检查

启动和请求完成后，建议确认日志中包含以下关键行：

```text
attention backend: SLA
Using Wan SLA topk 0.25
Using Wan SLA skip linear
cache-dit enabled on dual transformers
[DenoisingStage] average time per step
Pixel data generated successfully
```

如果日志中没有 `cache-dit enabled on dual transformers`，说明 Cache-DiT 未生效；如果 text encoder CPU offload 或 FSDP inference 与预期不一致，需要重新检查启动参数。
