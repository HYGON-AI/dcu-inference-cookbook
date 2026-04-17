# 量化方案

## 量化方法对比

| 方法 | 精度损失 | 速度提升 | 显存节省 | 部署复杂度 |
|------|---------|---------|---------|-----------|
| GPTQ | 低 | 高 | 3-4x | 中 |
| AWQ | 低 | 高 | 3-4x | 中 |
| GGUF | 中 | 高 | 3-4x | 低 |
| SmoothQuant | 低 | 中 | 2x | 中 |
| KV Cache INT8 | 极低 | 中 | 1.5-2x | 低 |

## GPTQ 量化

### 量化模型

```bash
# 使用 auto-gptq
pip install auto-gptq

python quantize.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --bits 4 \
    --group_size 128 \
    --output_dir ./qwen2.5-7b-gptq-4bit
```

### 推理使用

```python
from transformers import AutoModelForCausalLM, GPTQConfig

model = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-7b-gptq-4bit",
    device_map="auto",
    trust_remote_code=True,
)
```

## AWQ 量化

```bash
pip install autoawq

# 量化
python -m awq.entrypoints.main \
    quantize \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --output_path ./qwen2.5-7b-awq-4bit \
    --w_bit 4 \
    --q_group_size 128
```

## GGUF 格式

GGUF 适合 llama.cpp 生态，支持 CPU + GPU 混合推理：

```bash
# 转换为 GGUF
pip install gguf
python convert_hf_to_gguf.py ./model --outfile model.gguf

# 量化
./llama-quantize model.gguf model-q4_k_m.gguf q4_k_m
```

## 量化精度评估

建议量化后进行以下评估：

1. **Perplexity**: 与原始模型对比，差距应 < 5%
2. **下游任务**: 在关键任务上验证准确率
3. **人工评估**: 抽样检查生成质量

```python
# Perplexity 评估
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_ppl(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings["input_ids"])
    return torch.exp(outputs.loss).item()
```
