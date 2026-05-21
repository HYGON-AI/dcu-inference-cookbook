---
name: add-model
description: Guide for adding a new model deployment doc to dcu-inference-cookbook. Use this when asked to add a new model or create a new model deployment page.
---

# 新增模型部署文档规范

## 信息收集

在生成任何文档前，**必须先依次向用户询问以下信息**，每次只问一个问题，等待回答后再问下一个：

1. **模型卡**：模型在 ModelScope 或 HuggingFace 上的完整路径或 URL。
   例如：`hygon/GLM-5-Channel-INT4-w4a8`、`LLM-Research/Meta-Llama-3.1-70B-Instruct`

2. **框架**：`vLLM` 还是 `SGLang`（二选一）

3. **框架版本**：
   - 选择 vLLM 时只接受：`0.15.1` 或 `0.18.1`（其他版本需要用户重新输入）
   - 选择 SGLang 时只接受：`0.5.10`（其他版本需要用户重新输入）

4. **硬件平台**：从 `K100_AI`、`BW1000`、`BW1100` 中选择，可多选（其他值需要用户重新输入）

5. **启动命令**：
   - 若用户提供了完整的 `vllm serve` 或 `sglang serve` 命令，**原样采用，不做任何修改**
   - 若用户未提供，根据模型信息和硬件平台**生成模板命令**，告知用户可按需修改

收集完以上全部信息后，再按照下方规范生成文档。

## 模型列表表格格式

模型部署文档的 `## 模型列表` 章节必须包含以下列，且顺序固定：

| 模型权重 | 量化方式 | 框架版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | -------- | -------- | ---- | -------- | -------- |

> 列名 `框架版本` 对应 vLLM 文档写 `vLLM 版本`，SGLang 文档写 `SGLang 版本`。

### 各列说明

- **框架版本**：使用信息收集阶段用户指定的框架版本（如 `0.18.1`、`0.5.10`）。

- **模型权重**：模型在 ModelScope 上的完整路径，带链接。
  - 有 HYGON 量化版本时，优先使用 `hygon/` 前缀的 channelwise 模型：
    `[hygon/<MODEL-NAME>](https://www.modelscope.cn/models/hygon/<MODEL-NAME>)`
    例如：`[hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8)`
  - 无 HYGON 量化版本时（如 BF16 部署），使用上游原始模型 ID：
    例如：`[Qwen/Qwen3-235B-A22B](https://www.modelscope.cn/models/Qwen/Qwen3-235B-A22B)`
    或：`[moonshotai/Kimi-K2-Instruct](https://www.modelscope.cn/models/moonshotai/Kimi-K2-Instruct)`
  - 同一模型的多个行，后续行的模型权重列留空（用空格对齐）。

- **量化方式**：使用标准格式，例如：`INT4 W4A8`、`INT8 W8A8`、`FP8 W8A8`、`BF16`。

- **推荐硬件**：只使用以下官方产品名称（不带 GB 规格前缀外的其他字样）：
  - `K100_AI`
  - `BW1000 64GB`（简写 `BW1000`）
  - `BW1100 144GB`（简写 `BW1100`）

- **卡数**：整数，表示所需 DCU 数量。

- **部署方式**：
  - `IFB`：单机批量推理
  - `xPyD`：PD 分离，例如 `2P2D`（2 个 prefill 节点 + 2 个 decode 节点）、`1P2D`

- **启动命令**：加粗的 `` >_ `` 图标作为锚点链接，跳转到文档内对应的启动命令章节。格式：
  `[**\`>_\`**](#<anchor>)`
  例如：`[**\`>_\`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x)`

  **锚点生成规则**（GitHub Markdown）：章节标题全部小写，空格转 `-`，非字母数字非 `-` 字符直接删除（`/` 不转换为 `-`，直接删除）。

## 启动命令章节格式

**每一个表格行对应一个 `###` 章节**，章节标题格式固定为：

```
### <MODEL> <MODE> <HW> <Nx>
```

- `<MODEL>`：模型权重名（不含 `hygon/` 前缀，因为 `/` 会破坏锚点生成）
- `<MODE>`：`IFB` 或 `xPyD`（如 `2P2D`、`1P2D`）
- `<HW>`：推荐硬件简写（如 `BW1000`、`BW1100`）
- `<Nx>`：总卡数加 `x`（如 `8x`、`32x`、`24x`）

例如：
- `### GLM-5-Channel-INT4-w4a8 IFB BW1000 8x` → anchor `#glm-5-channel-int4-w4a8-ifb-bw1000-8x`
- `### GLM-5-Channel-INT4-w4a8 2P2D BW1000 32x` → anchor `#glm-5-channel-int4-w4a8-2p2d-bw1000-32x`

### SGLang IFB 章节结构

SGLang IFB 章节只有一个 bash 代码块，无需子标题：

````markdown
### GLM-5-Channel-INT4-w4a8 IFB BW1000 8x

```bash
export ...

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 8 \
  ...
```
````

### SGLang PD 分离章节结构

SGLang PD 分离章节开头加一行 IB 网卡配置说明，然后用 `####` 划分各节点：

````markdown
### GLM-5-Channel-INT4-w4a8 2P2D BW1000 32x

网卡配置参考：[IB 网卡](../../troubleshooting/common-issues.md#ib网卡)。

#### P node 0

```bash
export ...

sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --host "$(ip route get 1.1.1.1 | awk '/src/{print $7}')" \
  --port 30000 \
  --dist-init-addr "$(ip route get 1.1.1.1 | awk '/src/{print $7}'):5000" \
  --nnodes <P节点数> \
  --node-rank 0 \
  --tp-size <tp> \
  --disaggregation-mode prefill \
  ...
```

#### P node 1

说明：`--dist-init-addr` 填写当前分组 node0 的 IP，下面示例使用 `10.x.x.x`。

```bash
...（同 P node 0，node-rank 改为 1，dist-init-addr 改为固定 IP）
```

#### D node 0

```bash
...（decode 节点，disaggregation-mode decode）
```

#### D node 1

说明：`--dist-init-addr` 填写当前分组 D node0 的 IP，下面示例使用 `10.x.x.x`。

```bash
...（同 D node 0，node-rank 改为 1，dist-init-addr 改为固定 IP）
```

#### Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://<P_node0_ip>:30000 \
  --decode http://<D_node0_ip>:30000 \
  --policy cache_aware \
  --port 30001
```
````

### vLLM IFB 章节结构

vLLM IFB 章节只有一个 bash 代码块，无需子标题：

````markdown
### GLM-5-Channel-INT4-w4a8 IFB BW1100 8x

```bash
export VLLM_USE_MODELSCOPE=1
export ...

vllm serve hygon/GLM-5-Channel-INT4-w4a8 \
    -q <quantization> \
    --trust-remote-code \
    --dtype bfloat16 \
    -tp 8 \
    ...
```
````

### vLLM PD 分离章节结构

vLLM PD 分离的代理（proxy）内置于 P 节点进程中，通过 `--kv-transfer-config` 的 `proxy_port` 对外暴露，无需独立 Router 进程。章节开头加一行说明 P 节点和 D node 0 的示例 IP，然后用 `####` 划分各节点：

````markdown
### GLM-5-Channel-INT8-w8a8 1P2D BW1100 24x

以下示例中 `10.16.1.36` 为 P 节点（也是代理节点），`10.16.1.42` 是 D node 0 的主节点，实际部署时请根据实际情况修改。

#### P node

```bash
export VLLM_USE_MODELSCOPE=1
export ...
export VLLM_USE_DP_CONNECTOR=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    ...
    --enable-lightly-cp --enable-lightly-cplb \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_producer","kv_buffer_size":"1e4","kv_port":"21002","kv_connector_extra_config":{"proxy_ip":"<P_node_ip>","proxy_port":"30001","http_port":"8000","send_type":"PUT_ASYNC","instance_ip":"<P_node_ip>"}}'
```

#### D node 0

```bash
export VLLM_USE_MODELSCOPE=1
export ...
export VLLM_USE_DP_CONNECTOR=1

vllm serve hygon/GLM-5-Channel-INT8-w8a8 \
    -q slimquant_marlin \
    --trust-remote-code \
    ...
    --kv-transfer-config '{"kv_connector":"DuSwiftConnectorDp","kv_role":"kv_consumer","kv_buffer_size":"1e9","kv_port":"21003","kv_connector_extra_config":{"proxy_ip":"<P_node_ip>","proxy_port":"30001","http_port":"8000","send_type":"PUT_ASYNC","instance_ip":"<D_node0_ip>"}}' \
    --data-parallel-size-local 8 \
    --data-parallel-address <D_node0_ip> \
    --data-parallel-rpc-port 1127 \
    --data-parallel-start-rank 0 \
    --disable-custom-all-reduce
```

#### D node 1

```bash
...（同 D node 0，--data-parallel-start-rank 改为上一个 D node 的起始 rank + data-parallel-size-local；最后一个 D node 加 --headless）
```
````

## 模型相关说明

### SGLang

- **模型 ID**：使用 ModelScope 上的完整路径，例如 `hygon/GLM-5-Channel-INT4-w4a8`
- **启动命令**：使用 `sglang serve`（不用 `python -m sglang.launch_server`）
- **必须加**：`--trust-remote-code`
- **`--host`**：PD 分离模式必须指定，绑定节点对外的 IP；IFB 不需要。推荐用 `$(ip route get 1.1.1.1 | awk '/src/{print $7}')` 自动获取
- **`--dist-init-addr`**：多节点必须指定。格式 `<IP>:5000`。node 0 用自身 IP，其余节点填 node 0 的 IP
- **`--served-model-name`**：PD 分离时所有 P/D 节点必须设置相同的值；API 调用的 `model` 字段须与此一致
- **端口**：SGLang 默认 `30000`。Router 与 P node 0 同机时需换端口（推荐 `30001`）；不在文档中显式指定端口除非有特殊原因

### vLLM

- **模型 ID**：使用 ModelScope 上的完整路径，例如 `hygon/GLM-5-Channel-INT8-w8a8`
- **启动命令**：使用 `vllm serve`
- **必须加**：`--trust-remote-code`、`-q <quantization>`
- **`--served-model-name`**：PD 分离时所有 P/D 节点必须设置相同的值；API 调用的 `model` 字段须与此一致
- **端口**：vLLM 默认 `8000`（HTTP）；不在文档中显式指定 `--port` 除非有特殊原因
- **`--kv-transfer-config`**：PD 分离必须指定，P 节点为 `kv_producer`，D 节点为 `kv_consumer`
  - `proxy_port`：P 节点对外暴露的代理端口，客户端通过此端口发送请求（推荐 `30001`）
  - `http_port`：节点自身的 HTTP 端口，须与 `--port` 一致（默认 `8000`）
  - `kv_port`：KV 张量传输端口，P 节点与 D 节点须不同（如 P 用 `21002`，D 用 `21003`）
  - `instance_ip`：当前节点实际 IP
  - `proxy_ip`：P 节点 IP（P/D 节点均填 P 节点 IP）
- **D 节点多机**：通过 `--data-parallel-*` 参数配置；最后一个 D 节点加 `--headless`
- **P 节点特有**：`--enable-lightly-cp --enable-lightly-cplb --enforce-eager`

## API 调用章节格式

`## API 调用` 章节分两个子节：`### IFB` 和 `### PD 分离`。

### SGLang

````markdown
## API 调用

### IFB

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:30000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT4-w4a8",  # 替换为实际使用的模型名
    messages=[...],
    max_tokens=2048,
)
```

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5-Channel-INT4-w4a8", "messages": [...], "max_tokens": 128}'
```

### PD 分离

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。Router 默认端口为 `30000`，
若与 P node 0 部署在同一机器上需指定其他端口（示例中为 `30001`）。

```python
client = OpenAI(base_url="http://<router_ip>:30001/v1", api_key="not-needed")
```

```bash
curl http://<router_ip>:30001/v1/chat/completions ...
```
````

### vLLM

````markdown
## API 调用

### IFB

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="hygon/GLM-5-Channel-INT4-w4a8",  # 替换为实际使用的模型名
    messages=[...],
    max_tokens=2048,
)
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "hygon/GLM-5-Channel-INT4-w4a8", "messages": [...], "max_tokens": 128}'
```

### PD 分离

vLLM PD 分离模式下，客户端请求直接发送到 P 节点的代理端口（`proxy_port`，示例中为 `10.16.1.36:30001`）。

```python
client = OpenAI(base_url="http://<P_node_ip>:30001/v1", api_key="not-needed")
```

```bash
curl http://<P_node_ip>:30001/v1/chat/completions ...
```
````

## 示例（SGLang GLM-5）

````markdown
## 模型列表

| 模型权重 | 量化方式 | SGLang 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | ----------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.10 | BW1000 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x)   |
|                                                                                                 | INT4 W4A8 | 0.5.10 | BW1000 | 32 | 2P2D | [**`>_`**](#glm-5-channel-int4-w4a8-2p2d-bw1000-32x) |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x)   |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x) |
| [hygon/GLM-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-FP8-w8a8)   |  FP8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-fp8-w8a8-ifb-bw1100-8x)    |
|                                                                                                 |  FP8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-fp8-w8a8-1p2d-bw1100-24x)  |
````

## 示例（vLLM GLM-5）

````markdown
## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.18.1 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1100-8x)   |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.18.1 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x)   |
|                                                                                                 | INT8 W8A8 | 0.18.1 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x) |
````
