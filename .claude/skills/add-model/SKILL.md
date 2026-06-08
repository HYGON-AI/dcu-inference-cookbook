---
name: add-model
description: Guide for adding a new model deployment doc to dcu-inference-cookbook. Use this when asked to add a new model or create a new model deployment page.
---

# 新增模型部署文档规范

## 信息收集

在生成任何文档前，**必须先依次向用户询问以下信息**，每次只问一个问题，等待回答后再问下一个：

1. **模型卡**：模型在 ModelScope 或 HuggingFace 上的完整路径或 URL。
   例如：`hygon/GLM-5-Channel-INT4-w4a8`、`LLM-Research/Meta-Llama-3.1-70B-Instruct`
   **后续所有文档严格使用此处用户指定的模型 ID，不得推断或替换为其他模型 ID。**

2. **框架**：`vLLM` 还是 `SGLang`（二选一）

3. **框架版本**：
   - 选择 vLLM 时只接受：`0.15` 或 `0.18`（其他版本需要用户重新输入）
   - 选择 SGLang 时只接受：`0.5.10`（其他版本需要用户重新输入）

4. **硬件平台**：从 `K100_AI`、`BW1000`、`BW1100` 中选择，可多选（其他值需要用户重新输入）

5. **启动命令**：
   - 必须向用户展示以下固定询问话术，不能只作为内部规则：
     ```text
     请提供完整的 `vllm serve` 或 `sglang serve` 启动命令；如果没有现成命令，也可以回复“没有”。

     如果提供命令，我会按照项目 `CONTRIBUTING` / 文档贡献要求进行检查和最小必要规范化，不会修改关键必要命令,你可按需修改。

     如果没有提供命令，我会根据模型信息和硬件平台生成模板命令，并提示你可按需修改。
     ```
   - 若用户提供了完整命令，按照上述话术和“启动命令规范化要求”处理。
   - 若用户未提供，根据模型信息和硬件平台**生成模板命令**，并告知用户可按需修改。

收集完以上全部信息后，再按照下方规范生成文档。

## 文档正文内容边界

生成或补充的部署 md 文档正文只写模型卡/模型列表、最佳实践启动命令和 API 调用示例；不要写入命令规范化过程说明、替换原因、风险提示、操作记录或最终回复类说明。此类说明只允许出现在给用户的最终回复中。

## 模型列表表格格式

模型部署文档的 `## 模型列表` 章节必须包含以下列，且顺序固定：

| 模型权重 | 量化方式 | 框架版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | -------- | -------- | ---- | -------- | -------- |

> 列名 `框架版本` 对应 vLLM 文档写 `vLLM 版本`，SGLang 文档写 `SGLang 版本`。

### 各列说明

- **框架版本**：使用信息收集阶段用户指定的框架版本（如 `0.18`、`0.5.10`）。

- **模型权重**：严格使用信息收集步骤 1 中用户指定的模型 ID，带 ModelScope 链接。不得推断、替换或猜测为其他模型 ID。
  - `[<MODEL-ID>](https://www.modelscope.cn/models/<MODEL-ID>)`
    例如：`[hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8)`
  - 同一模型的多个行，后续行的模型权重列留空（用空格对齐）。

- **量化方式**：使用标准格式，例如：`INT4 W4A8`、`INT8 W8A8`、`FP8 W8A8`、`BF16`。

- **推荐硬件**：只使用以下官方产品名称（不带 GB 规格前缀外的其他字样）：
  - `K100_AI`
  - `BW1000 64GB`（简写 `BW1000`）
  - `BW1100 144GB`（简写 `BW1100`）

  **表格行排序**：同一模型的多条行先按框架版本从新到旧排序（如 vLLM `0.18` 在 `0.15` 前），同一框架版本内再按硬件平台排序，顺序固定为 **BW1100 → BW1000 → K100_AI**。

- **卡数**：整数，表示所需 DCU 数量。

- **部署方式**：
  - `IFB`：单机批量推理
  - `xPyD`：PD 分离，例如 `2P2D`（2 个 prefill 节点 + 2 个 decode 节点）、`1P2D`

- **启动命令**：加粗的 `` >_ `` 图标作为锚点链接，跳转到文档内对应的启动命令章节。格式：
  `[**\`>_\`**](#<anchor>)`
  例如：`[**\`>_\`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x)`

  **锚点生成规则**（GitHub Markdown）：章节标题全部小写，空格转 `-`，非字母数字非 `-` 字符直接删除（`/` 不转换为 `-`，直接删除）。

## 启动命令规范化要求

若用户提供了完整的 `vllm serve` 或 `sglang serve` 命令，先按项目贡献要求中的文档规范检查是否需要改写。内部可保留原始命令用于判断，但默认不要向用户展示完整的原始命令对比。

- 若用户提供的是 `python -m sglang.launch_server` 或 `python3 -m sglang.launch_server`，必须先提示上传者该启动方式不符合文档规范，建议改为 `sglang serve`，并在获得用户确认后再替换；不得直接静默改写。
- 若用户命令中的模型路径或模型名称与信息收集阶段用户指定的模型 ID 不一致，必须停止生成并提示上传者确认；不得直接替换模型 ID 或继续生成文档。

只允许为了以下目的改写用户命令：

- 删除注释掉的代码和无关注释。
  - 删除以 `#` 开头的说明行、分隔线行和被注释掉的命令行，例如 `# xxx`、`# export ...`、`# vllm serve ...`、`#===============...`。
  - 删除命令行或环境变量行末尾的行内注释，只保留 `#` 前面的有效命令内容，例如 `export VLLM_REJECT_SAMPLE_OPT=0 # ...` 应规范为 `export VLLM_REJECT_SAMPLE_OPT=0`。
- 不要定义额外的 shell 变量（除了必要的 vllm / sglang 环境变量），删除仅用于日志保存、时间戳、日志目录、输出重定向或 `tee` 的 shell 包装逻辑；启动前置准备与复现部署环境相关的有效步骤默认保留。不要将 `VLLM_`、`SGLANG_`、`HIP_`、`NCCL_`、`GLOO_`、`HSA_`、`ROC_`、`ROCM_` 等框架/运行时环境变量视为 shell 包装逻辑，即使用于 profiling、debug 或诊断，也默认保留。
- 对用户命令中的 shell 变量，若只是核心启动参数的别名，则直接内联为实际值；若只用于日志、目录、时间戳、端口包装或输出重定向，则删除。模型路径变量统一替换为信息收集阶段用户指定的模型 ID。
- 若 shell 变量用于核心启动参数但用户未给出明确赋值，必须按原样保留该变量引用（例如 `--pp-size $pp`），不得推断、补写或替换为默认值（例如不得擅自写成 `--pp-size 1`）。
- 不得新增用户命令中没有出现的启动参数、环境变量或运维步骤；除非该参数是框架文档明确要求的必需项且缺失会导致命令不可用，此时必须先在最终回复中说明，并优先请用户确认，不得静默补写。
- 将本地路径或非规范模型名称替换为信息收集阶段用户指定的官方模型 ID；涉及模型 ID 替换时必须按下方流程说明原因和风险，并由用户确认。替换说明只写在最终回复中，不得写入部署 md 文档正文。
- 恢复框架默认端口：vLLM 默认 `8000`，SGLang 默认 `30000`；除非用户明确要求保留自定义端口。
- 调整换行、缩进和参数排列，使命令块符合文档格式。

以下内容属于最佳实践启动配置，必须保留并写入对应 bash 代码块：

- `export` 环境变量本体，尤其是通信、采样、缓存、融合算子、attention、MoE、量化、spec decode、lightop、profiling、debug、诊断等框架/运行时相关开关；保留变量行，不保留变量行上的解释性注释。
- `vllm serve` / `sglang serve` 的量化参数、并行参数、显存参数、上下文长度、prefix cache、KV cache、speculative decoding、compilation config 等核心参数。

除以上允许改写范围外，不得擅自删除、重排或改写用户提供的 shell 命令、环境变量、量化参数、并行参数、核心启动参数或运维步骤；不得为用户未提供的参数、未赋值变量或缺失配置补默认值。若不确定某一行是日志包装还是最佳实践配置，默认保留其有效命令内容；若不确定某个变量的值，默认保留变量引用原样；如需标注判断，只能写在给用户的最终回复中，不得写入部署 md 文档正文。

## 启动命令章节格式

**每一个表格行对应一个 `###` 章节**，章节标题格式固定为：

- **SGLang**（标题末尾加版本号）：
  ```
  ### <MODEL> <MODE> <HW> <Nx> SGLang <VERSION>
  ```

- **vLLM**（标题末尾加版本号）：
  ```
  ### <MODEL> <MODE> <HW> <Nx> vLLM <VERSION>
  ```

- `<MODEL>`：模型权重名（不含 `hygon/` 前缀，因为 `/` 会破坏锚点生成）
- `<MODE>`：`IFB` 或 `xPyD`（如 `2P2D`、`1P2D`）
- `<HW>`：推荐硬件简写（如 `BW1000`、`BW1100`）
- `<Nx>`：总卡数加 `x`（如 `8x`、`32x`、`24x`）
- `<VERSION>`：vLLM 版本（如 `0.18`、`0.15`）

例如（SGLang）：
- `### GLM-5-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.10` → anchor `#glm-5-channel-int4-w4a8-ifb-bw1000-8x-sglang-0510`
- `### GLM-5-Channel-INT4-w4a8 2P2D BW1000 32x SGLang 0.5.10` → anchor `#glm-5-channel-int4-w4a8-2p2d-bw1000-32x-sglang-0510`

例如（vLLM）：
- `### GLM-5-Channel-INT4-w4a8 IFB BW1100 8x vLLM 0.18` → anchor `#glm-5-channel-int4-w4a8-ifb-bw1100-8x-vllm-018`
- `### GLM-5-Channel-INT8-w8a8 1P2D BW1100 24x vLLM 0.18` → anchor `#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-vllm-018`

### SGLang IFB 章节结构

SGLang IFB 章节只有一个 bash 代码块，无需子标题：

**缩进规范**：`sglang serve \` 单独占第一行，后续每个参数独占一行，统一缩进 **2 个空格**。

````markdown
### GLM-5-Channel-INT4-w4a8 IFB BW1000 8x SGLang 0.5.10

```bash
sglang serve \
  --model-path hygon/GLM-5-Channel-INT4-w4a8 \
  --trust-remote-code \
  --tp-size 8 \
  ...
```
````

### SGLang PD 分离章节结构

SGLang PD 分离章节开头加一行 IB 网卡配置说明，然后用 `####` 划分各节点。**缩进规范同 IFB**：`sglang serve \` 首行，后续参数缩进 2 个空格。

````markdown
### GLM-5-Channel-INT4-w4a8 2P2D BW1000 32x SGLang 0.5.10

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

**缩进规范**：`vllm serve <model-id> \` 单独占第一行，后续每个参数独占一行，统一缩进 **2 个空格**。

````markdown
### GLM-5-Channel-INT4-w4a8 IFB BW1100 8x vLLM 0.18

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

vLLM PD 分离的代理（proxy）内置于 P 节点进程中，通过 `--kv-transfer-config` 的 `proxy_port` 对外暴露，无需独立 Router 进程。**缩进规范同 IFB**：`vllm serve <model-id> \` 首行，后续参数缩进 2 个空格。章节开头加一行说明 P 节点和 D node 0 的示例 IP，然后用 `####` 划分各节点：

````markdown
### GLM-5-Channel-INT8-w8a8 1P2D BW1100 24x vLLM 0.18

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

`## API 调用` 章节分两个子节：`### IFB` 和 `### PD 分离`。已有子节保持不动；本次新增 IFB 且缺少 `### IFB` 时才补 `### IFB`，本次新增 PD 分离且缺少 `### PD 分离` 时才补 `### PD 分离`，两者都新增且都缺少时才同时补两个子节。

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
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-sglang-0510)   |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-sglang-0510) |
| [hygon/GLM-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-FP8-w8a8)   |  FP8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510)    |
|                                                                                                 |  FP8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-fp8-w8a8-1p2d-bw1100-24x-sglang-0510)  |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.10 | BW1000 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x-sglang-0510)   |
|                                                                                                 | INT4 W4A8 | 0.5.10 | BW1000 | 32 | 2P2D | [**`>_`**](#glm-5-channel-int4-w4a8-2p2d-bw1000-32x-sglang-0510) |
````

## 示例（vLLM GLM-5）

````markdown
## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.18 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1100-8x-vllm-018)   |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-vllm-018)   |
|                                                                                                 | INT8 W8A8 | 0.18 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-vllm-018) |
````

## 文件命名规范

**不要为每个具体模型变体单独创建文件。** 使用**模型系列**文件，将同系列的不同版本/规格合并在一个文件中：

- ✅ 正确：在 `qwen3.md` 中增加 Qwen3-235B-A22B 的章节
- ❌ 错误：新建 `qwen3-235b-a22b.md`

**判断规则**：
- 对带显式版本号的模型系列，优先匹配到相同大/小版本的文件；若仓库已按版本拆分（如已有 `deepseek-v3.md` 与 `deepseek-v3.2.md`），新增 `DeepSeek-V3.1` 应使用或新建 `deepseek-v3.1.md`，不要合并到 `deepseek-v3.md`。
- 若模型 ID 在显式大/小版本后带后缀（如 `DeepSeek-V3.1-Terminus`、`Qwen3.5-Instruct`），仍归入对应大/小版本系列文件（如 `deepseek-v3.1.md`、`qwen3.5.md`），不要为后缀单独建文件。
- 若 `docs/model-deployment/{vllm|sglang}/` 下已存在同系列文件（如 `qwen3.md`、`kimi-k2.md`），在该文件中补充新模型，不要新建具体变体文件。**严禁修改文件中已有的任何内容**，包括已有行的模型 ID、命令参数、章节标题等。
- 只有当模型权重、量化方式、框架版本、推荐硬件（以及卡数、部署方式）均与用户本次信息匹配时，才可视为同一条记录并更新对应内容；框架版本不同（如 `0.18` 与 `0.15`）时必须保留旧记录并新增新版本记录。用户在第 4 步只选择了部分硬件（如仅 `BW1000`）时，只匹配/新增这些硬件，不得自动扩展到其他硬件平台。
- 新模型行应插入 `## 模型列表` 表格中的合理位置，而不是一律追加到末尾。排序规则只用于确定本次新增记录的插入位置，严禁为了重新排序而修改、移动或重写已有记录；已有记录即使不完全符合新规则，也应保持不变。若新增记录需要插入到已有记录之间，只插入新增行，不调整原有行内容。
- 插入位置先按模型规模从小到大排列（如 4B、9B、27B、35B、122B、397B）；同一规模下，先按基础模型名称主体的自然字典序排列，并将同一基础模型及其量化/后缀变体归为同一组，例如 `Qwen3.5-27B`、`Qwen3.5-27B-W8A8`、`Qwen3.5-27B-W8A8-INT8` 应连续排列。
- 同一基础模型组内，基础模型优先，其后先排列基础模型自身的量化变体，量化/变体类型优先级为：`BF16` → `FP8` → `INT8/W8A8` → `INT4/W4A8` → `AWQ`；再排列指令/聊天/推理等后缀变体，后缀名称按自然字典序排列，例如 `Chat` → `Instruct` → `Reasoning` → `Thinking`。同一后缀变体自己的量化版本仍按上述量化/变体类型优先级排序；未列出的变体按模型 ID 的自然字典序排列。同一模型变体的多条记录先按框架版本从新到旧排序，例如 vLLM `0.18` 放在 `0.15` 上方；同一框架版本内再按硬件记录 **BW1100 → BW1000 → K100_AI** 排序。若现有文件已有清晰顺序，优先沿用该文件的局部顺序。
- `## 启动命令` 中对应的 `###` 章节也应放在与 `## 模型列表` 表格一致的相对位置；若新模型行插入或调整到表格中部，必须同步把对应启动命令章节移动到同一相对顺序，不能留在原末尾；只有无法判断合理位置时，才追加到该章节末尾，并在最终说明中提示原因。
- 若不存在同系列文件，新建以系列命名的文件（如 `deepseek-v3.md`、`glm-5.md`）。

## 最终步骤：更新 README 支持矩阵

生成部署文档后，**必须同步更新 `README.md` 的 `📋 模型列表` 支持矩阵**。

### 矩阵结构

矩阵为 HTML 表格，列顺序固定：**厂商 → 模型 → 框架 → K100_AI → BW1000 → BW1100**。

每个模型在矩阵中占两行（vLLM 行 + SGLang 行），模型名称列使用 `rowspan="2"`：

```html
<tr>
  <td rowspan="2">ModelName</td>
  <td>vLLM</td>
  <td align="center">-</td>
  <td align="center"><a href="docs/model-deployment/vllm/filename.md">✅</a></td>
  <td align="center">-</td>
</tr>
<tr>
  <td>SGLang</td>
  <td align="center">-</td>
  <td align="center">-</td>
  <td align="center">-</td>
</tr>
```

单元格取值：
- **有文档且支持该硬件**：`<td align="center"><a href="docs/model-deployment/{vllm|sglang}/{filename}.md">✅</a></td>`
- **已知不支持**：`<td align="center">-</td>`
- **计划中/待验证**：`<td align="center">🚧</td>`

### 操作步骤

1. **在矩阵中查找该模型**：
   - **找到了**：直接修改对应框架行（vLLM 或 SGLang）中各硬件平台的单元格，将已验证的平台改为带链接的 ✅，未覆盖的平台保持原值（`-` 或 `🚧`）。
   - **找到了且目标框架/硬件已经是带链接的 ✅**：保持 README 不变，记录为已检查；最终回复必须说明该矩阵项已覆盖并给出对应链接路径。
   - **找不到**：在对应厂商的 `<tbody>` 区块末尾插入两行新 `<tr>`（vLLM 行 + SGLang 行），参照上方结构模板。

2. **文档路径**格式为：`docs/model-deployment/{vllm|sglang}/{filename}.md`，`{filename}` 与新建文档的文件名一致。

3. **只改动必要行**，不调整其他模型的内容，保持 diff 最小。
