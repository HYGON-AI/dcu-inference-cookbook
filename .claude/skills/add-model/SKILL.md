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
   - 必须向用户展示以下固定询问话术，不能只作为内部规则；展示时必须原样输出 fenced `text` 代码块（即灰底、左上角显示 `text`、右上角带复制按钮的样式），不要使用 HTML `<div>`、普通段落或引用块。固定话术如下：

     ```text
     请提供启动命令，格式可选：
     - <完整 vllm serve 或 sglang serve 启动命令>、
     - 没有，参考 <模型名>
     - 没有,替我生成

     可选：
     硬件：如果你选择了多个硬件平台，请按硬件分别提供或标注命令；不标注则把同一份命令复制到所选全部硬件
     部署方式：不写则按 IFB 处理；如果需要同时新增 IFB 和 PD，需要你分别指定
     ```
   - 若用户提供了完整命令，按照上述话术和“启动命令规范化要求（提供命令时）”处理。
   - 若用户提供了完整命令，且选择多个硬件平台但命令没有逐段明确区分硬件，按上述话术把同一份命令复制到用户选择的所有硬件下面。
   - 若用户未提供命令，根据模型信息、硬件平台和参考模型名（如有）查找参考或生成模板命令；具体规则见“启动命令模板生成规则（未提供命令时）”。

收集完以上全部信息后，再按照下方规范生成文档。

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
  - 同一模型的多个行，后续行的模型权重列必须留空（用空格对齐），不得重复写模型链接。
  - 若本次新增行插在已有同一模型记录之后，即使框架版本、硬件平台、卡数或部署方式不同，也属于后续行，模型权重列仍必须留空。

- **量化方式**：使用标准格式，例如：`INT4 W4A8`、`INT8 W8A8`、`FP8 W8A8`、`BF16`。

- **推荐硬件**：只使用以下官方产品名称（不带 GB 规格前缀外的其他字样）：
  - `K100_AI`
  - `BW1000 64GB`（简写 `BW1000`）
  - `BW1100 144GB`（简写 `BW1100`）

  **表格行排序**：同一模型的多条行先按框架版本从新到旧排序（如 vLLM `0.18` 在 `0.15` 前），同一框架版本内再按硬件平台排序，顺序固定为 **BW1100 → BW1000 → K100_AI**；同一硬件下 `IFB` 在前，后面紧跟该硬件对应的 PD 分离记录（如 `1P1D`、`2P2D`）。

- **卡数**：整数，表示所需 DCU 数量。

- **部署方式**：
  - `IFB`：单机批量推理
  - `xPyD`：PD 分离，例如 `2P2D`（2 个 prefill 节点 + 2 个 decode 节点）、`1P2D`

- **启动命令**：加粗的 `` >_ `` 图标作为锚点链接，跳转到文档内对应的启动命令章节。格式：
  `[**\`>_\`**](#<anchor>)`
  例如：`[**\`>_\`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x)`

  **锚点生成规则**（GitHub Markdown）：章节标题全部小写，空格转 `-`，非字母数字非 `-` 字符直接删除（`/` 不转换为 `-`，直接删除）。

## 启动命令规范化要求（提供命令时）

若用户提供了完整的 `vllm serve` 或 `sglang serve` 命令，先按项目贡献要求中的文档规范检查是否需要改写。内部可保留原始命令用于判断，但默认不要向用户展示完整的原始命令对比。

只允许做以下最小必要规范化：

- 中文与编码：读取或编辑中文 Markdown 时必须使用 UTF-8；不得把终端显示出的乱码文本复制写回文档。
- 启动入口：`python -m sglang.launch_server` 或 `python3 -m sglang.launch_server` 改为 `sglang serve`，保留原命令关键参数，并在最终回复中说明。
- 注释清理：删除说明性注释、分隔线、被注释掉的命令（如 `# xxx`、`# export ...`、`# vllm serve ...`、`#===============...`），以及命令行或环境变量行末尾的行内注释；只保留有效命令内容，例如 `export VLLM_REJECT_SAMPLE_OPT=0 # ...` 应规范为 `export VLLM_REJECT_SAMPLE_OPT=0`。
- 变量处理：不要定义额外的 shell 变量（除了必要的 vllm / sglang 环境变量）,若只用于日志、目录、时间戳、端口包装或输出重定向，则删除；若是核心启动参数的别名且已明确赋值，则直接内联为实际值；若用于核心启动参数但未明确赋值，必须停止生成并请用户提供明确赋值，不得保留变量引用、推断或补写默认值。
- 模型与端口：若用户命令中的模型路径或模型名称与信息收集阶段用户指定的模型 ID 不一致，必须提示用户将本地路径或非规范模型名称替换为信息收集阶段用户指定的官方模型 ID，并在获得用户确认后才能继续生成文档；端口按部署方式处理：IFB 恢复框架默认端口（vLLM `8000`、SGLang `30000`），PD 分离按对应章节结构处理 P/D 节点服务端口与 Router/proxy 端口，除非用户明确要求保留自定义端口。相关替换说明只写在最终回复中，不得写入部署 md 文档正文。
- 格式整理：调整换行、缩进和参数排列，使命令块符合文档格式。

以下内容属于最佳实践启动配置，必须保留并写入对应 bash 代码块：

- `export` 环境变量本体，尤其是通信、采样、缓存、融合算子、attention、MoE、量化、spec decode、lightop、profiling、debug、诊断等框架/运行时相关开关；保留变量行，不保留变量行上的解释性注释。
- `vllm serve` / `sglang serve` 的量化参数、并行参数、显存参数、上下文长度、prefix cache、KV cache、speculative decoding、compilation config 等核心参数。

除以上允许改写范围外，不得擅自删除、重排或改写用户提供的 shell 命令、环境变量、量化参数、并行参数、核心启动参数或运维步骤；不得为用户未提供的参数、未赋值变量或缺失配置补默认值。若不确定某一行是日志包装还是最佳实践配置，默认保留其有效命令内容；如需标注判断，只能写在给用户的最终回复中，不得写入部署 md 文档正文。

## 启动命令模板生成规则（未提供命令时）

当用户回复“没有”或未提供现成启动命令时，仅表示用户没有提供现成命令；

处理顺序：

1. 若用户额外提供参考模型名，先在已有部署文档中查找该参考模型的命令；没有找到该参考模型，或没有对应硬件/部署方式的参考命令时，告诉用户没有对应参考，停止生成，并询问是否先跳过。
2. 若用户没有提供参考模型名，在已有部署文档中查找同框架、同版本、架构/量化方式相近的启动命令。
3. 找到参考命令后，可直接作为模板来源生成启动命令；最终回复中说明参考了哪个已有模型/文件，以及做了哪些必要替换或估算。
4. 仍无合适参考时，生成基础模板命令，只使用用户指定模型 ID、按规则估算的 TP/卡数、框架通用必要参数和量化必要参数，并在最终回复说明未套用其他模型的优化参数。

生成模板命令时必须遵守以下规则：

- 参考已有 md 命令时，必须把来源命令中的模型 ID 替换为信息收集第 1 步用户指定的模型 ID；不得把参考模型名或已有 md 命令中的其他模型 ID 写入本次文档。
- 不得跨框架、跨版本或跨硬件平台套用启动参数、环境变量或运行时开关，尤其不得在 vLLM `0.18` 与 `0.15` 之间互相参考，也不得用 `BW1000`、`BW1100`、`K100_AI` 之间任一硬件的命令去生成另一硬件的命令。
- `tensor-parallel-size` / `--tp-size` / `-tp` 必须按下方显存规则重新估算，不能直接沿用参考命令里的卡数：
  - 必须按下方固定表格读取所选硬件的单卡显存。

   | 硬件平台  | 单卡显存 |
   | --------- | -------- |
   | `K100_AI` | 64 GB    |
   | `BW1000`  | 64 GB    |
   | `BW1100`  | 144 GB   |

  - 根据模型权重和单卡显存计算所需 TP：`required_tp = model_weight_GB / gpu_mem_GB`，并预留 20%~30% 显存余量。
  - TP 取值必须满足硬件拓扑约束：当估算结果 `<= 8` 时，向上取最近的 2 的幂（`1`、`2`、`4`、`8`）；当估算结果 `> 8` 时，只能取 `8 * 2^n`（`16`、`32`、`64`、`128` ...），不得取 `12`、`24`、`40` 等非 `8 * 2^n` 值。
  - 当参考命令的卡数 `<= 8`、重新估算后的总卡数 `> 8`，且框架为 vLLM `0.18` 时，不要把总卡数直接写成单节点 `-tp <总卡数>` / `--tensor-parallel-size <总卡数>`；应按 `节点数 = 总卡数 / 8` 生成多节点命令，且此时每个节点的 TP 固定为 `8`。只新增多节点参数，不改写参考命令已有的其它并行参数。
    例如：参考命令单节点 8 卡，本次估算总卡数为 16，则生成 2 个节点；两个节点都保留参考命令原有的每节点卡数参数（`-tp 8` 或 `--tensor-parallel-size 8`），并分别增加 `--nnodes 2`、`--node-rank 0/1`、`--master-addr <node1_ip>`，最后一个节点追加 `--headless`。
  - 若用户提供的现成启动命令中 TP 不满足上述取值约束，必须先提示用户确认是否按规则调整 TP；获得确认后再修改命令和表格卡数，不得静默改写。

模型权重可按以下方式估算：

- BF16 ≈ 参数量 * 2 GB
- FP8 / INT8 W8A8 ≈ 参数量 * 1 GB
- FP4 / INT4 W4A16 / INT4 W4A8 / AWQ ≈ 参数量 * 0.5 GB
- MoE 模型使用总权重大小（all experts）估算，而不是 active params。

写入或展示模板命令时，必须说明参考来源；若未找到合适参考，则说明生成的是基础模板。

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

SGLang PD 分离章节开头加一行 IB 网卡配置说明，然后用 `####` 划分各节点。**缩进规范同 IFB**：`sglang serve \` 首行，后续参数缩进 2 个空格。P/D 节点服务端口使用 `30000`，Router 使用 `30001`，除非用户明确要求保留具体自定义端口。

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

PD 分离模式下，客户端请求发送到 SGLang Router，而非直接发送到 P/D 节点。示例中 Router 端口为 `30001`。

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
| [hygon/GLM-5-Channel-FP8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-FP8-w8a8)   |  FP8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-fp8-w8a8-ifb-bw1100-8x-sglang-0510)    |
|                                                                                                 |  FP8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-fp8-w8a8-1p2d-bw1100-24x-sglang-0510)  |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.5.10 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-sglang-0510)   |
|                                                                                                 | INT8 W8A8 | 0.5.10 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-sglang-0510) |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.5.10 | BW1000 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1000-8x-sglang-0510)   |
|                                                                                                 | INT4 W4A8 | 0.5.10 | BW1000 | 32 | 2P2D | [**`>_`**](#glm-5-channel-int4-w4a8-2p2d-bw1000-32x-sglang-0510) |
````

## 示例（vLLM GLM-5）

````markdown
## 模型列表

| 模型权重 | 量化方式 | vLLM 版本 | 推荐硬件 | 卡数 | 部署方式 | 启动命令 |
| -------- | -------- | --------- | -------- | ---- | -------- | -------- |
| [hygon/GLM-5-Channel-INT8-w8a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT8-w8a8) | INT8 W8A8 | 0.18 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-vllm-018)   |
|                                                                                                 | INT8 W8A8 | 0.18 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-vllm-018) |
|                                                                                                 | INT8 W8A8 | 0.15 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int8-w8a8-ifb-bw1100-8x-vllm-015)   |
|                                                                                                 | INT8 W8A8 | 0.15 | BW1100 | 24 | 1P2D | [**`>_`**](#glm-5-channel-int8-w8a8-1p2d-bw1100-24x-vllm-015) |
| [hygon/GLM-5-Channel-INT4-w4a8](https://www.modelscope.cn/models/hygon/GLM-5-Channel-INT4-w4a8) | INT4 W4A8 | 0.18 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1100-8x-vllm-018)   |
|                                                                                                 | INT4 W4A8 | 0.15 | BW1100 |  8 | IFB  | [**`>_`**](#glm-5-channel-int4-w4a8-ifb-bw1100-8x-vllm-015)   |
```` 

## 文件命名规范

**不要为每个具体模型变体单独创建文件。** 使用**模型系列**文件，将同系列的不同版本/规格合并在一个文件中：

- ✅ 正确：在 `qwen3.md` 中增加 Qwen3-235B-A22B 的章节
- ❌ 错误：新建 `qwen3-235b-a22b.md`

**判断规则**：
- 文件归属：优先使用 `docs/model-deployment/{vllm|sglang}/` 下已有的同系列文件（如 `qwen3.md`、`kimi-k2.md`），不存在时才新建以系列命名的文件（如 `deepseek-v3.md`、`glm-5.md`）。带显式版本号的模型优先归入相同大/小版本文件；若仓库已按版本拆分（如已有 `deepseek-v3.md` 与 `deepseek-v3.2.md`），新增 `DeepSeek-V3.1` 应使用或新建 `deepseek-v3.1.md`，不要合并到 `deepseek-v3.md`。显式大/小版本后的后缀（如 `DeepSeek-V3.1-Terminus`、`Qwen3.5-Instruct`）不单独建文件，仍归入对应大/小版本系列文件（如 `deepseek-v3.1.md`、`qwen3.5.md`）。
- 原有内容保护与记录修正：补充新模型时，已有记录和已有启动命令的内容必须保持原样；只有用户再次提供相同模型、框架、版本和硬件时，才视为修正对应已有记录，可按本次输入更新该记录的量化方式、卡数、部署方式和启动命令。框架版本不同必须保留旧记录并新增新版本记录；用户只选择部分硬件时，只匹配或新增这些硬件，不得自动扩展到其他硬件平台。除此之外，不得改写、补齐、规范化、合并或拆分已有表格行、`###` 标题、命令参数、环境变量和说明文字。为插入本次新增内容，可以最小范围移动已有行/章节，但必须保持表格行与对应启动命令章节成对同步。默认排序和章节同步规则只用于判断本次新增内容的位置和相对顺序。
- 默认排序：插入位置先按模型规模从小到大排列（如 4B、9B、27B、35B、122B、397B）；同一规模下按基础模型名称主体的自然字典序排列，并将同一基础模型及其量化/后缀变体归为同一组（如 `Qwen3.5-27B`、`Qwen3.5-27B-W8A8`连续排列）。同一基础模型组内，基础模型优先，其后按 `BF16` → `FP8` → `INT8/W8A8` → `INT4/W4A8` → `AWQ` 排列量化变体，再按自然字典序排列指令/聊天/推理等后缀变体（如 `Chat` → `Instruct` → `Reasoning` → `Thinking`）；同一后缀变体的量化版本仍按上述量化优先级排序，未列出的变体按模型 ID 自然字典序排列。
- 章节同步：先确定本次新增记录在 `## 模型列表` 表格中的位置，再把对应 `###` 启动命令章节放到 `## 启动命令` 中相同的相对位置；模型列表行与启动命令章节必须一一对应、位置一致。不得为了减少 diff 或避免打断已有块而把新增命令追加到末尾。只有无法判断新增章节的合理位置时，才追加到章节末尾并在最终说明中提示原因。

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
   - **找不到**：在对应厂商的 `<tbody>` 区块末尾插入两行新 `<tr>`（vLLM 行 + SGLang 行），参照上方结构模板。

2. **文档路径**格式为：`docs/model-deployment/{vllm|sglang}/{filename}.md`，`{filename}` 与新建文档的文件名一致。

3. **只改动必要行**，不调整其他模型的内容，保持 diff 最小。
