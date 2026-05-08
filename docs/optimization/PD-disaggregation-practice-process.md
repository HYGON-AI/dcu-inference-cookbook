# PD 分离测试最佳实践流程（适配大模型 Ling1T 4K 输入 / 1.5K 输出、TTFT 3s、TPOT 100ms）

&#x20;

## 核心前提

&#x20;

不同业务指标、延迟 SLA、模型规格无统一标准 PD 分离最优方案，需按固定步骤逐阶测试，敲定并行方案、单节点吞吐、PD 节点配比、系统最大吞吐。

&#x20;

## 一、测试背景基准

&#x20;

模型：Ling1T输入：4k tokens｜输出：1.5k tokensSLA 约束：TTFT≤3s、TPOT P99≤100ms目标：测出系统最大吞吐、PD 节点配比、最优并行组网

&#x20;

***

&#x20;

## 二、步骤 1：单测 P 节点（Prefill）最大吞吐

&#x20;

### 1. 各类并行方案特性对比

&#x20;

表格

|        并行方案       |   吞吐特征  |  延迟特征  |  适用场景  |                核心原因               |
| :---------------: | :-----: | :----: | :----: | :-------------------------------: |
|      PP 流水线并行     |   高吞吐   | 高 TTFT | 延迟约束宽松 | 卡间通信量小；多请求同时 Prefill，单请求 TTFT 被拉高 |
|      TP 张量并行      |   低吞吐   | 低 TTFT | 低延迟强诉求 |      卡间通信量大；单请求多卡串行推理，TTFT 更低     |
| attentionDP+moeEP | 高吞吐、高延迟 | 高 TTFT |  节点间组网 |        逻辑同 PP，多请求并行 Prefill       |
| attentionDP+moeTP | 高吞吐、高延迟 | 高 TTFT |  节点内组网 |         适配 MoE 模型节点内高吞吐诉求         |

&#x20;

### 2. 项目选型

&#x20;

本项目 TTFT 3s 约束宽松，选择 PP 并行：PP4TP2

&#x20;

### 3. Prefill 最大吞吐测试方法

&#x20;

1.  调参约束：通过 `chunk-size`、`max-running-request`、`--pp-max-micro-batch-size` 限制单次 Prefill 推理 Token 数；Token 数越大吞吐越高、延迟越高，做权衡取舍。
2.  部署要求：必须启用 PD 分离服务（IFB 模式测试吞吐不准）；Decode 并行方案暂随意，不干扰 Prefill 测试。
3.  压测配置：请求数 = 256、请求率 = 无限大；Decode 输出长度设为 1，彻底消除 Decode 对 Prefill 的资源抢占与吞吐干扰。
4.  结果取值：压测输出 QPS 即为P 节点单节点最大 Prefill 吞吐。

&#x20;

### 4. 项目实测结果

&#x20;

Ling1T P 节点单节点吞吐：3.4 QPS

&#x20;

***

&#x20;

## 三、步骤 2：计算 D 节点（Decode）最大吞吐

&#x20;

### 1. Decode 并行方案特性

&#x20;

*   PP 并行：跨节点降低显存占用，显存充足不优先选用
*   TP 并行：低吞吐、低 TPOT 延迟
*   attentionDP+moeEP/TP 低延迟模式：吞吐上限高、延迟下限高

&#x20;

### 2. 项目选型

&#x20;

TPOT 要求 P99≤100ms，选定 EP16DP16 并行；补充：Ling1T 未启用 MTP，原生延迟偏高；若后续降为 50ms 延迟诉求，可启用 MTP 或加大 EP 分片。

&#x20;

### 3. Decode 最大吞吐测试方法

&#x20;

1.  部署前提：PD 分离部署，利用木桶效应，规避 P 节点成为瓶颈。
2.  去瓶颈手段：P 节点开启 `radix-cache（prefix-cache）`，使用`generated-shared-prefix`数据集拉高缓存命中率，彻底消除 P 侧瓶颈。
3.  压测方式：高请求率下发流量，逐步放大 Decode batch，观察服务日志。
4.  TPOT 达标判定逻辑TPOT 要求 100ms → 单请求每秒生成 Token 数需 ≥ 10 tokens/s随着 batch 增大，单请求均分吞吐下降，需找到满足：`running_request × 10 tokens/s = 服务日志Token吞吐` 的临界 batch-size
5.  Decode QPS 计算公式$\text{Decode QPS} = \frac{\text{日志Token吞吐} \times \text{DP-size}}{\text{输出序列长度}}$

&#x20;

### 4. 项目实测计算

&#x20;

$\text{QPS} = 280 \times 16 \div 1536 \approx 2.91 \ \text{QPS}$

&#x20;

***

&#x20;

## 四、步骤 3：PD 节点配比计算

&#x20;

### 1. 配比核心原则

&#x20;

理想均衡：（$\text{P实例数} \times \text{P单节点QPS} \approx \text{D实例数} \times \text{D单节点QPS}$）无需绝对相等，数值接近即可。

&#x20;

### 2. 本项目配比

&#x20;

P 节点 QPS=3.4，D 节点 QPS=2.91，数值接近；最终组网：1P（单节点）+ 1D（两节点）

&#x20;

***

&#x20;

## 五、步骤 4：整机 Bench 压测 & 结果验证

&#x20;

### 1. 压测基准规则

&#x20;

按木桶原理，系统最大吞吐以 P/D 中 QPS 较小值 为基准请求率，可小幅上浮试探极限。

&#x20;

### 2. PD 分离压测三阶段

&#x20;

1.  初始阶段：Decode batch 逐步爬坡，系统未满载，实际 QPS＜理想值
2.  平稳阶段：P 持续匀速打请求、D 稳态消费，batch 稳定，系统真正满载，QPS 等于理想理论值
3.  结束阶段：P 请求发完，Decode batch 逐步回落，再次非满载

&#x20;

> 关键：拉长平稳阶段占比，测试结果才会贴近理论最大吞吐。

&#x20;

### 3. 调参优化 & 项目结果

&#x20;

*   优化手段：调高 `num_prompt`（本项目设为 10000），拉长平稳阶段时长
*   最终实测：整机 QPS=2.85，趋近理论 2.91 QPS；TPOT P99=100ms，完全满足 SLA，测出系统极限吞吐。

&#x20;

***

&#x20;

## 六、可复用标准化流程总结

&#x20;

1.  按 TTFT/TPOT 延迟 SLA，选定 P 侧、D 侧并行方案
2.  PD 分离隔离压测，测出 P 单节点最大 QPS
3.  P 开前缀缓存去瓶颈，标定 D 侧合规最大 batch，计算 D 单节点 QPS
4.  按 P、D QPS 做节点数量配比均衡
5.  大 num\_prompt 跑整机 Bench，以平稳阶段满载 QPS 作为系统最优吞吐

