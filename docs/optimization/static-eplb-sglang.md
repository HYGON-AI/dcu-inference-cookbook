# Static EPLB 原理与使用最佳实践（适配 MoE 大模型 P/D 分离部署）


MoE 模型中，专家并行（EP）会导致各个 GPU 之间的工作负载分布不均。这种不均衡会迫使系统等待计算或通信最慢的 GPU，从而浪费计算周期，并因专家激活而增加内存占用。随着 GPU 数量（即 EP 规模）的增加，不均衡问题会变得更加严重。

为了解决这一问题，DeepSeek 开发了专家并行负载均衡器（EPLB）。EPLB 以专家分布统计数据作为输入，计算出最优的专家排布方案，以最小化不均衡程度。用户可以分配冗余专家（例如额外增加 32 个专家），与原有的 256 个专家结合后，形成一个包含 288 个专家的资源池。这个资源池允许 EPLB 对专家进行策略性的放置或复制——例如，将使用最频繁的专家复制多份，或者将一个使用频率中等的专家与几个很少使用的专家组合放置在同一个 GPU 上。

EPLB 分为静态（Static）与动态（Dynamic / Online）两种模式。本文针对静态模式给出原理与操作指南。Static EPLB 的最简用法：**dump 一份专家负载分布 pt → 传给 `--init-expert-location` → sglang 启动时自动 rebalance 一次 → 运行时零开销**。


## 一、EPLB 核心算法

EPLB 的核心是**两个基础组件**（`balanced_packing` + `replicate_experts`）加上**两种装箱策略**（Hierarchical / Global）。策略由 `rebalance_experts` 入口按 `num_groups % num_nodes == 0` 分派。

### 1.1 两个基础组件

| 组件 | 作用 | 输入 | 输出 | 约束 | 算法 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `balanced_packing(weight, num_packs)` | n 个带权物品装到 m 个箱子，每箱恰好装 n/m 个，各箱总重尽量均衡 | `weight: [X, n]` 物品权重、`num_packs: m` 箱数 | `pack_index: [X, n]` 物品所属箱、`rank_in_pack: [X, n]` 箱内次序 | `n % m == 0`；`X` 为批次维（通常为 num_layers） | LPT 贪心（降序排 → 塞当前最轻且未装满的箱） |
| `replicate_experts(weight, num_phy)` | 给 num_log 个 logical 分配副本共 num_phy 个 physical，使摊平后负载最均衡 | `weight: [X, num_log]` 每 logical 负载、`num_phy` 目标 physical 数 | `phy2log: [X, num_phy]` 物理→逻辑映射、`rank: [X, num_phy]` 第几副本、`logcnt: [X, num_log]` 每 logical 副本数 | `num_phy >= num_log`；`logcnt.sum(-1) == num_phy` | 水填：每次挑 `weight/logcnt` 最大的 logical 加副本 |

### 1.2 分层负载均衡（Hierarchical）

#### 1.2.1 触发条件

`num_groups % num_nodes == 0`，且 `num_groups > 1, num_nodes > 1` 才有实质效果。

#### 1.2.2 三步流程

1. **Group → Node 装箱**：`balanced_packing(tokens_per_group, num_nodes)`，将同 group experts 尽量装到同一 node
2. **Node 内冗余复制**：`replicate_experts(node_weight, num_phy / num_nodes)`，热点副本只能在本 node 内加
3. **Physical → GPU 装箱**：`balanced_packing(tokens_per_phy, num_gpus / num_nodes)`，node 内 GPU 间均衡

#### 1.2.3 特性

- **副本约束**：热点副本必须在自己所属 node 内，不能跨 node
- **通信优势**：同 group experts 同 node → dispatch 走 node 内 NVLink → 跨 node all-to-all 流量下降
- **均衡度打折**：某个 node 天然更热时，其他 node 的空闲 GPU 帮不上忙

#### 1.2.4 适用场景

- Prefill 阶段（计算密集、通信瓶颈明显）
- 小 EP + 多机部署
- 有 group 的模型（如 DeepSeek V3 `n_group=8`）

### 1.3 全局负载均衡（Global）

#### 1.3.1 触发条件

`num_groups % num_nodes != 0`；或单机（`num_nodes=1`）、无 group（`num_groups=1`）等价场景。

#### 1.3.2 代码复用与三步退化

Global 不是独立实现，而是用 `num_groups=1, num_nodes=1` 参数塞进同一个 `rebalance_experts_hierarchical` 函数，让三步中的前一步退化：

1. **Step 1 退化**：1 group 装 1 node，恒等映射
2. **全局冗余复制**：`replicate_experts(weight, num_phy)`，全部 logical 参与竞争副本，可分到任意 GPU
3. **Physical → GPU 全局装箱**：`balanced_packing(tokens_per_phy, num_gpus)`，全部 GPU 一起装箱

#### 1.3.3 特性

- **无副本约束**：热点副本可分到任意 GPU
- **均衡度最优**：全局装箱自由度最大
- **通信不考虑拓扑**：副本可能分到任意 node，跨 node 流量不控制

#### 1.3.4 适用场景

- Decode 阶段（访存密集、负载均衡决定 TPOT）
- 大 EP 部署
- 无 group 的模型（如 Hy3、DeepSeek V4）

### 1.4 两种策略对比

| 维度 | Hierarchical | Global |
| :---: | :---: | :---: |
| 触发判据 | `num_groups % num_nodes == 0` | 反之或退化场景 |
| 副本可否跨 node | 否 | 是 |
| 装箱范围 | node 内 GPU | 全局 GPU |
| 均衡度 | 中 | 高 |
| 通信友好度 | 高 | 低 |
| 适用阶段 | Prefill | Decode |
| 适用规模 | 小 EP | 大 EP |


## 二、基础概念

### 2.1 Logical vs Physical Expert

| 概念 | 说明 |
| :---: | :---: |
| Logical Expert | 模型定义的原始 expert（如 Hy3 为 192，DeepSeek V3 为 256，V4 为 384） |
| Physical Expert | 部署时实际放置的 expert slot，数量 = logical + 冗余 |
| Redundant Expert | 热点 logical 的额外副本，由 `--ep-num-redundant-experts` 指定 |
| num_physical | `num_logical + ep_num_redundant_experts` |

### 2.2 三张核心映射表

- `physical_to_logical_map [L, num_phy]`：每个 physical slot 对应哪个 logical
- `logical_to_all_physical_map [L, num_log, maxcnt]`：每个 logical 的所有副本 slot 列表
- `logical_count [L, num_log]`：每个 logical 的**副本数**（EPLB 算法输出；注意与 dump pt 中同名字段"调用次数"语义不同）

### 2.3 Group 概念

MoE routing 常按 group 分（如 DeepSeek V3 每 8 experts 一 group，选 top-k group）。是否有 group 决定装箱策略走向：

| 模型 | n_group | topk_group | EPLB 实际走向 |
| :---: | :---: | :---: | :---: |
| DeepSeek V3 | 8 | 4 | 多机走 Hierarchical |
| DeepSeek V4 | 无字段（default 1） | 无 | 恒走 Global |
| Hy3 | 无字段（default 1） | 无 | 恒走 Global |

### 2.4 Static vs Dynamic EPLB

| 维度 | Static EPLB | Dynamic EPLB |
| :---: | :---: | :---: |
| 触发时机 | 服务启动一次装载 | 运行时按周期重排 |
| 权重迁移 | 无（映射即最终态） | 有（P2P 跨 rank 交换） |
| 性能开销 | 零运行时开销 | 有 rebalance 开销 |
| 依赖数据 | dump 的专家负载分布 | 在线 recorder 实时统计 |
| 关键参数 | `--init-expert-location` | `--enable-eplb` |
| MTP draft 层 | 不参与 | 不参与 |


## 三、Dump 专家分布

### 3.1 P/D 部署要求

#### 3.1.1 Recorder 启用位置

| 端 | 是否开 recorder | 参数 |
| :---: | :---: | :---: |
| P 端 | 否 | 不加 recorder 相关参数 |
| D 端 | 是 | `--expert-distribution-recorder-mode stat`（**必需**）<br>`--enable-expert-distribution-metrics`（可选） |

#### 3.1.2 关键约束

- dump 阶段**不需要**加 `--ep-num-redundant-experts`，`logical_count` 与冗余无关；冗余在装箱阶段注入
- dump 阶段**不需要**开 `--enable-eplb`；动态 EPLB 会周期性 rebalance，dump 到的分布不是"自然分布"而是"动态优化态"，作为静态输入会稀释热点信号
- P、D 两端负载分布不同，**必须分别 dump、分别加载**，不要共用一份 pt

#### 3.1.3 Dump 步骤

首先在 D 端设置环境变量,指定 dump 输出目录:

```bash
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/path/to/eplb_trace
```

然后在 D 节点依次发送以下请求(端口以 sglang 启动时的 `--port` 为准,默认 `30000`):

```bash
# Step 1: 开启记录
curl -X POST http://127.0.0.1:30000/start_expert_distribution_record

# Step 2: 压测真实测试场景数据集(此步骤在客户端跑压测)

# Step 3: 停止记录
curl -X POST http://127.0.0.1:30000/stop_expert_distribution_record

# Step 4: 落盘到 SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR
curl -X POST http://127.0.0.1:30000/dump_expert_distribution_record
```

**dump 产物说明**:dump 完成后,`SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR` 目录下会生成单个 pt 文件,直接作为 `--init-expert-location` 输入即可。

#### 3.1.4 Dump 产物

`_StatAccumulator.dump` 输出字段：

| 字段 | Shape | 含义 |
| :---: | :---: | :---: |
| `logical_count` | `[L, num_log]` | 每层每 logical expert 的调用次数总和 |
| `average_utilization_rate_over_window` | `[L]` | 每层平均利用率 |

#### 3.1.5 Pt 文件用途

直接作为 `--init-expert-location` 输入，sglang 启动时会读取 `logical_count` 字段，自动调用 `rebalance_experts` 产出映射并装载。

#### 3.1.6 验证 dump 是否有效

```python
import torch
d = torch.load('xxx.pt')
print(d.keys())                    # 应含 'logical_count'
print(d['logical_count'].shape)    # [L, num_log]
print(d['logical_count'].sum(-1))  # 各层总调用量，应非零且分布合理
print(d['logical_count'].std(-1) / d['logical_count'].mean(-1))  # 变异系数，>0.3 说明有明显热点
```

## 四、服务加载与启动

### 4.1 启动参数

| 参数 | 说明 | 是否必填 |
| :---: | :---: | :---: |
| `--init-expert-location <pt>` | dump 产物路径 | ✅ 必填 |
| `--ep-num-redundant-experts N` | 冗余专家数 | ✅ 必填 |
| `--enable-expert-distribution-metrics` | 加载后观测平衡度 | 可选（建议开） |
| `--enable-eplb` | 动态 EPLB 开关 | ❌ 静态场景禁止开启 |
| `--ep-dispatch-algorithm` | Dispatch 策略 | ❌ 不需要，静态映射天然走 static |

### 4.2 加载流程

sglang 启动时加载路径：

1. 从 `--init-expert-location` 指定的 pt 读 `logical_count`
2. 走 `init_by_eplb` 分支，调用 `eplb_algorithms.rebalance_experts(logical_count, num_replicas, num_groups, num_nodes, num_gpus)`
3. 根据 `num_groups % num_nodes` 判据选 Hierarchical 或 Global
4. 产出 `physical_to_logical_map`、`logical_to_all_physical_map`、`logical_count`
5. 装载到 EPLB manager，权重按映射摆放
6. 完成后进入正常推理，**运行时无 rebalance**

### 4.3 典型启动参数示例

**只加载静态 EPLB**：

```bash
--init-expert-location <path>.pt \
--ep-num-redundant-experts N \
--enable-expert-distribution-metrics
```

## 五、效果验证

### 5.1 效果判定指标(D端)

| 指标 | 观察目标 | 期望方向 |
| :---: | :---: | :---: |
| 平衡度 | 各rank上的平均token与最大token之比 | avg/max 趋近 1 |
| TPOT P99 | D 端主目标 SLA | 稳定下降 |
| 整机吞吐 QPS | 系统满载吞吐 | 上升或持平 |
| 单节点吞吐 | 单个 D 节点 QPS | 上升 |

## 附：核心要点一句话

- **Static EPLB 用法**：dump 一份 pt → 直接传 `--init-expert-location` → sglang 启动时 rebalance 一次 → 运行时零开销
- **算法本质**：水填加冗余副本 + LPT 装箱到 GPU；Hierarchical 与 Global 只是装箱范围（node 内 vs 全局）不同
- **策略选择**：`num_groups % num_nodes == 0` 且有 group 才走 Hierarchical，否则一律 Global
- **P/D 必须分别 dump 分别加载**，两端专家负载分布不同

