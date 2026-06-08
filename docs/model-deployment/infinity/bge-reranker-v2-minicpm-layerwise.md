# bge-reranker-v2-minicpm-layerwise on infinity_emb

## 模型简介

bge-reranker-v2-minicpm-layerwise 是 BAAI 发布的层序 reranker 模型，基于 MiniCPM 架构，支持通过 `cutoff_layer` 参数动态截断推理层数，平衡精度和性能。

## 模型列表

| 模型权重 | 量化方式 | 框架 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ---- | -------- | ---- | -------- | -------- |
| [BAAI/bge-reranker-v2-minicpm-layerwise](https://www.modelscope.cn/models/BAAI/bge-reranker-v2-minicpm-layerwise) | BF16 | infinity_emb | BW1000 | 1 | IFB | [**`>_`**](#bw1000-1x-infinity_emb) |

## 启动命令

### BW1000 1x infinity_emb

```bash
export INFINITY_QUEUE_SIZE=320000

infinity_emb v2 \
    --model-id /path/to/bge-reranker-v2-minicpm-layerwise \
    --trust-remote-code \
    --device cuda \
    --engine torch \
    --no-bettertransformer
```

## API 调用

### Rerank 请求

```python
import requests

response = requests.post(
    "http://localhost:8000/rerank",
    json={
        "query": "什么是深度学习？",
        "documents": [
            "深度学习是机器学习的一个分支",
            "今天天气很好",
        ],
        "raw_scores": True,
    },
)
print(response.json())
```

```bash
curl http://localhost:8000/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "什么是深度学习？",
    "documents": [
      "深度学习是机器学习的一个分支",
      "今天天气很好"
    ],
    "raw_scores": true
  }'
```
