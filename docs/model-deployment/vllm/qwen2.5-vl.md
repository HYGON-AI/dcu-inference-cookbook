# Qwen2.5-VL on vLLM

## 模型简介

Qwen2.5-VL 是 Qwen 系列的多模态视觉语言模型，包含3B 7B、32B、72B 等多个参数规模。模型原生支持图像、文本、视频等多种输入，在视觉理解、文档解析、视频问答、目标定位等任务上表现优异，结合 vLLM 可实现高吞吐、低延迟的在线推理服务。


## 模型列表

| 模型 | 参数量 | 上下文 | 量化方式 | 推荐硬件 |
|------|--------|--------|--------|---------|
| Qwen2.5-VL-3B-Instruct | 3B | 128K | 未量化(BF16) | 1x BW1000 64GB |
| Qwen2.5-VL-7B-Instruct | 7B | 128K | 未量化(BF16)  | 1x BW1000 64GB |
| Qwen2.5-VL-32B-Instruct | 32B | 128K | 未量化(BF16) | 4x BW1000 64GB |
| Qwen2.5-VL-72B-Instruct | 72B | 128K | 未量化(BF16) | 8x BW1000 64GB |


## 启动命令

### Qwen2.5-VL-3B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    -tp 1 \
    --trust-remote-code 
```
### Qwen2.5-VL-7B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    -tp 1 \
    --trust-remote-code 
```
### Qwen2.5-VL-32B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen2.5-VL-32B-Instruct \
    -tp 4 \
    --trust-remote-code 
```
### Qwen2.5-VL-72B
```bash
export VLLM_HCU_USE_FLASH_ATTN=1
export VLLM_HCU_USE_FLASH_ATTN_UNIFIED=1
export VLLM_HCU_USE_CUSTOM_TOPK_TOPP_SAMPLER=1
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
    -tp 8 \
    --trust-remote-code 
```

## API调用
```python
from openai import OpenAI
import httpx  

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-VL-32B-Instruct",  # 修正为实际存在的模型名，并确保与本地一致
        messages=[
            {"role": "system", "content": "你是一个专业的编程助手。"},
            {"role": "user", "content": "用 Python 实现一个高效的 LRU Cache"},
        ],
        max_tokens=2048,
        temperature=0.7,
    )
    print(response.choices[0].message.content)
    
except httpx.ConnectError:
    print("连接失败：请确认本地大模型服务是否已启动（localhost:8000）？")
except Exception as e:
    print(f"发生错误：{e}")

```

