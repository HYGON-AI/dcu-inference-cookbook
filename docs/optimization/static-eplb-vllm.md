# Static EPLB：离线专家布局的记录与加载

## 1. 功能简介

EPLB（Expert Parallel Load Balancing）会根据 MoE 模型中各专家的实际负载，调整逻辑专家到物理专家槽位的映射，并利用冗余专家缓解热点专家造成的负载不均。

常规动态 EPLB 需要在服务运行期间收集负载，再计算并应用新的专家布局。Static EPLB 在此基础上增加了离线布局能力：

- **记录模式**：将运行过程中最新一次已提交的 EPLB 布局写入 JSON 文件。
- **加载模式**：服务启动时读取该 JSON，并在正式推理前恢复对应布局。

这适合流量分布相对稳定、希望复用已验证布局的场景，可以提高不同实例之间初始专家布局的一致性，减少冷启动阶段重新收集负载带来的性能波动。

> Static EPLB 固定的是服务的**启动布局**。加载完成后，动态 EPLB 仍会按照 `step_interval` 继续运行，并可能再次调整专家布局。当前版本没有单独的“加载后永久冻结布局”开关；如果希望服务运行期间保持该布局，可以将 `step_interval` 设置得足够大，使其在预计的服务生命周期内不触发后续重排。

## 2. 适用版本与前提

使用前需要满足以下条件：

1. 模型是支持 Expert Parallel 和 EPLB 的 MoE 模型。
2. 启动时同时设置 `--enable-expert-parallel` 和 `--enable-eplb`。
3. `num_redundant_experts` 大于 0 时，模型及其权重加载实现必须支持冗余 physical experts。
4. 记录和加载 map 时，应保持以下配置一致：
   - 模型及模型版本；
   - TP、DP、EP 拓扑；
   - MoE 层数和 logical expert 数；
   - `num_redundant_experts`；
   - speculative decoding/MTP 配置。
5. 所有进程都必须能访问 `expert_map_path` 指向的文件；记录目录需要对 EP rank 0 可写。

## 3. 配置参数

Static EPLB 通过 `--eplb-config` 中的两个字段控制：

| 参数 | 说明 |
| --- | --- |
| `expert_map_record_path` | 将最新已提交的 EPLB 布局写入指定 JSON 文件。 |
| `expert_map_path` | 启动时从指定 JSON 文件加载专家布局。 |

这两个参数互斥，不能在同一次启动中同时配置，否则服务会直接报错。

常用 EPLB 参数如下：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `num_redundant_experts` | `0` | 冗余 physical expert 数量。 |
| `window_size` | `1000` | 参与负载统计的滑动窗口大小。 |
| `step_interval` | `3000` | 两次动态专家重排之间的 step 间隔。 |
| `use_async` | `false` | 是否使用异步 EPLB。 |
| `log_balancedness` | `false` | 是否记录负载均衡指标；开启后会产生额外通信开销。 |
| `log_balancedness_interval` | `1` | 负载均衡指标的日志间隔。 |
| `communicator` | 自动选择 | 专家权重迁移后端，例如 `torch_nccl`、`torch_gloo`、`nixl` 或 `pynccl`。 |

## 4. 推荐工作流

推荐分成“生成 map”和“加载 map”两个阶段。

### 4.1 使用代表性流量生成 map

下面以 HY-V3、MTP 和 DP8EP8 部署为例。该命令会启用动态 EPLB，并把最新已提交布局写入 `/data/eplb.json`：

```bash
vllm serve /module/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
  --enable-expert-parallel \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 2,
    "quantization": "slimquant_marlin"
  }' \
  --all2all_backend=deepep_low_latency \
  --block-size 64 \
  -q slimquant_marlin \
  --enable-eplb \
  --eplb-config '{
    "expert_map_record_path": "/data/eplb.json",
    "num_redundant_experts": 16,
    "step_interval": 3000,
    "log_balancedness": true,
    "log_balancedness_interval": 100,
    "use_async": false
  }' \
  --max-num-batched-tokens 512 \
  --dtype bfloat16 \
  --data-parallel-size 8 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --kv_cache_dtype fp8_e4m3
```

启动后向服务发送具有代表性的生产流量或回放流量。建议至少等待一次动态 EPLB 重排完成，再保存生成的文件。

可以通过日志确认 map 已写入：

```text
Recorded offline EPLB expert map to /data/eplb.json.
```

注意：服务注册模型时就会写入一次初始布局；不要把刚启动、尚未完成负载采集时的文件误认为已经优化后的布局。

### 4.2 启动时加载 map

第二次启动时保持模型和并行配置不变，将 `expert_map_record_path` 替换为 `expert_map_path`：

```bash
vllm serve /module/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3 \
  --enable-expert-parallel \
  --speculative-config '{
    "method": "mtp",
    "num_speculative_tokens": 2,
    "quantization": "slimquant_marlin"
  }' \
  --all2all_backend=deepep_low_latency \
  --block-size 64 \
  -q slimquant_marlin \
  --enable-eplb \
  --eplb-config '{
    "expert_map_path": "/data/eplb.json",
    "num_redundant_experts": 16,
    "step_interval": 1000000000,
    "log_balancedness": true,
    "log_balancedness_interval": 100,
    "use_async": false
  }' \
  --max-num-batched-tokens 512 \
  --dtype bfloat16 \
  --data-parallel-size 8 \
  --tool-call-parser hy_v3 \
  --reasoning-parser hy_v3 \
  --enable-auto-tool-choice \
  --kv_cache_dtype fp8_e4m3
```

加载时会执行以下操作：

1. 读取并校验 map；
2. 将已加载的专家权重重排到目标 physical slots；
3. 更新 `physical_to_logical`、`logical_to_physical` 和 replica count；
4. 刷新 MoE routing table；
5. 跳过 profile 阶段的 dummy rearrangement；
6. 进入正常推理流程。

成功加载时可以看到类似日志：

```text
Loading offline EPLB expert map from /data/eplb.json for model /module/Hy3-CHANNEL-FP8-w8a8-sero-ignore-from-script3.
```

## 5. Map 文件格式

推荐使用程序自动生成的 version 2 多模型格式，不建议手工编辑。核心字段 `physical_to_logical_map` 是一个二维数组：

```text
[num_moe_layers][num_physical_experts]
```

数组中：

- 第一维表示 MoE layer；
- 第二维下标表示 physical expert slot；
- 每个值表示该 physical slot 当前承载的 logical expert ID。

例如，一个模型有 4 个 logical experts 和 2 个 redundant experts，某一层的布局可以是：

```json
[0, 1, 2, 3, 0, 1]
```

其中 physical slots 4 和 5 分别是 logical experts 0 和 1 的副本。

单模型 legacy 格式也可以直接加载：

```json
{
  "version": 1,
  "format": "vllm_offline_eplb_physical_to_logical",
  "model_name": "/models/example-moe",
  "model_class": "ExampleForCausalLM",
  "num_moe_layers": 2,
  "num_logical_experts": 4,
  "num_physical_experts": 6,
  "num_redundant_experts": 2,
  "physical_to_logical_map": [
    [0, 1, 2, 3, 0, 1],
    [0, 1, 2, 3, 2, 3]
  ]
}
```

加载器会进行以下校验：

- map 不能为空；
- map 必须是二维数组；
- physical expert 数必须与当前模型一致；
- logical expert ID 不能为负数；
- logical expert ID 必须小于当前模型的 logical expert 数；
- 每一层的每个 logical expert 至少要有一个 physical replica；
- 多模型格式必须包含与当前模型类名匹配的 key。

如果 map 的层数多于当前模型所需层数，但 physical expert 数一致，加载器会取最后 `num_moe_layers` 层，并输出提示日志；其他 shape 不一致的情况会直接失败。

## 6. 运行机制

### 6.1 记录机制

- 只有 EP device group 的 rank 0 写文件。
- 模型注册后会记录一次初始布局。
- 同步 EPLB 提交新布局后会再次记录。
- 异步 EPLB 完成全部 layer 的权重迁移和 map commit 后会再次记录。
- 写入时先生成同目录的 `.tmp` 文件，再通过原子替换更新目标文件，避免读到半写入 JSON。
- 多个模型共用记录路径时，会合并到 `model_maps`，不会主动覆盖其他模型的 entry。

### 6.2 加载机制

模型权重初始加载完成后，Static EPLB 根据离线 map 将权重从初始 primary placement 重排到目标布局，然后提交映射并刷新 routing table。这样 redundant physical experts 对应的权重和路由信息能够保持一致。

加载 map 不会关闭运行期的负载统计和动态重排。如果希望布局在一次短期实验中保持不变，可以将 `step_interval` 设置得大于预计运行 step 数；生产环境应结合流量变化和重新平衡需求谨慎设置。

## 7. 最佳实践

1. 使用具有代表性的流量生成 map，而不是只使用随机请求。
2. 至少等待一次 EPLB 重排完成后再固化 map。
3. 对 map 文件进行版本管理，不要在多个不同拓扑的服务之间共用同一路径。
4. 上线前使用相同模型和拓扑执行启动、正确性和性能验证。
5. map 加载失败时让服务快速失败，不要回退到不完整或缺失专家的布局。
6. 流量分布明显变化后重新采集并生成 map。
