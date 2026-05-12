# MORI-EP User Guide

​	MORI-EP 为专家并行（Expert Parallelism）提供高性能的 `MoE`（专家混合）分发与聚合核函数。它同时支持节点内（XGMI）和节点间（RDMA）通信。**目前版本仅为初版，后续会逐步迭代优化**。

注：当前版本 Dispatch 输出 layout 按照 deepep 结果格式进行单独转换，因此 mori-ep Dispatch 结果 可直接 传给 deepgemm 进行 计算。

同时 Deepgemm 两次计算完成后的结果 也可以直接 传给 mori-ep 进行 Token 计算结果的聚合。

## 初版支持限制：

初版目前仅支持 多节点下 低延迟 Dispatch 和 Combine 算子，且 当前实现 Dispatch 和 Combine 必须成对使用。原因在于 当前实现中 Combine 需要 Dispatch 算子写入的 多个 标记数组用于 Combine 的 结果归约，如果不成对使用，Combine 读取到的就是错误的 内容。

 **暂不支持** 以下特性：

1. Dispatch 不支持 FP8、INT8 量化；
2. 不支持 hook=True；
3. 不支持 TBO、SBO。

## 安装方式

### whl包安装：

​	见 内部 mori-ep 发版文档

### 源码编译：

#### 编译 `Rocshmem` 

- 拉取`rocshmem` 仓库最新代码：

  天龙网卡环境：

  ​	`git clone -b rocshmem_shca http://42.228.13.241:10068/dcutoolkit/deeplearing/rocshmem.git`

  其他网卡环境：

  ​	`git clone -b main http://42.228.13.241:10068/dcutoolkit/deeplearing/rocshmem.git`

- 建立 `rocshmem` 编译后的文件夹，这里以 `/root/rocm/rocshmem-install` 为例

- `mkdir build && cd build`

-  安装脚本增加可执行权限： `chmod +x  ../scripts/build_configs/gda_mlx5`

- 编译 `rocshmem`:

  `  MPI_ROOT=/opt/mpi UCX_ROOT=/opt/ucx INSTALL_PREFIX=/root/rocm/rocshmem-install CMAKE_PREFIX_PATH="/opt/dtk:$CMAKE_PREFIX_PATH" ../scripts/build_configs/gda_mlx5`

- 单机进行验证：

  ` mpirun --allow-run-as-root -np 8 -x ROCSHMEM_MAX_NUM_CONTEXTS=8 -x HSA_DISABLE_CACHE=1 -x ROCSHMEM_ALLOWED_IBV_DEVICES="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9" ./examples/rocshmem_put_signal_test`

  ![image-20260512094440595](C:\Users\29051\AppData\Roaming\Typora\typora-user-images\image-20260512094440595.png)

#### 编译 mori-ep

- 拉取最新 mori-ep 代码：

  ​	`git clone -b mori-ep http://42.228.13.241:10068/dcutoolkit/deeplearing/mori.git`

- 编译 mori-ep：

  - 指定 mori-ep 使用 `rocshmem` 库: 修改 mori-ep/build.sh 中 SHMEM_INSTALL_PREFIX 为 `roshmem` 编译路径。

  - 编译mori-ep:  `./build.sh rocshmem`

- `cd ./dist`， 然后安装生成的 whl 包

## 环境变量

#### 环境变量及含义

| 环境变量                         | 含义                                                         | 如何设置                                              |
| -------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX | 每 Rank 上需要为低延迟 创建的 QP总数，目前一个QP绑定到一个专家上 | EP 总专家数                                           |
| HIP_VISIBLE_DEVICES              | 每个 计算节点上 可见 DCU 编号，以 一个计算节点上8个rank 为例，每个 rank 会按rank 选择其中一个 DCU 进行绑定 | 每计算节点上 所有 DCU 编号                            |
| ROCSHMEM_MAX_NUM_CONTEXTS        | `Rocshmem` 中 除 默认上下文之外的 上下文空间总数。`rocshmem` 以 Context 组织所有 可用资源（DCU 和 NIC 以及 QP CP 等），其中 默认 Context 必定存在，其他的 Context 的数目通过 该环境变量控制，默认值为 32 | 32                                                    |
| ROCSHMEM_HEAP_SIZE               | 每个 Rank 绑定  DCU 上申请 对称堆的 大小（字节）             | 建议1g大小                                            |
| ROCSHMEM_TOPO_FILE_FORCE         | 以配置文件方式，为每个 DCU 指定 绑定 的NIC 网卡编号          | 811目前指定文件： [topo.config](..\a_811\topo.config) |
| ROCSHMEM_SQ_SIZE                 | 指定 QP 队列 和 CQ 队列的长度                                | 建议1024                                              |

其他环境选择NIC 网卡参考文档进行设置：https://kcnm6g5dkw5p.feishu.cn/wiki/U13gw7A3YicMtPkxBMJcROaXnag

#### 环境变量设置示例：

```shell
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ROCSHMEM_MAX_NUM_CONTEXTS=32 
export ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1,shca_2,shca_3
export ROCSHMEM_HEAP_SIZE=10737418240
export ROCSHMEM_TOPO_FILE_FORCE=/root/topo.config  # 这里 设置为 DCU 选择RDMA 网卡的拓扑文件路径
export ROCSHMEM_SQ_SIZE=1024
```

## 性能 benchmark

1. 拉取 mori-ep 代码到 多个节点的共享文件夹中`git clone -b mori-ep http://42.228.13.241:10068/dcutoolkit/deeplearing/mori.git`

2. 进入 测试文件夹：

   2node:   mori/tests/2nodes

   4node:   mori/test/4nodes

3. 修改 所有 sh 文件中 `--master-addr=`  参数为 0 号 Rank 的IP地址。

4. 每个节点按照索引顺序，分别执行各自的 sh文件：

   ​	2node: 

   ​		0 node:  ./0.sh

   ​      	  1 node:  ./1.sh 

   ​	4node: 

   ​		0 node:  ./0.sh

   ​      	  1 node:  ./1.sh 

   ​		2 node:  ./2.sh 

   ​		3 node:  ./3.sh  

5. 观察各个 节点上的 终端输出。

## 调用流程

为减少迁移 难度，增加使用便利性，mori-ep 侧接口 尽量保持 和 `Deepep` 低延迟接口一致，仅添加一些 mori-ep 必须一些入参。

### 导入 mori_ep 库：

import mori_ep

### 获取 RDMA 缓冲区大小：

#### 调用示例：

```Plain
num_rdma_bytes = mori_ep.Buffer.get_low_latency_rdma_size_hint(
                                                        num_tokens, 
                                                        hidden, 
                                                        num_ranks, 
                                                        num_experts, 
                                                        num_topk)
```

#### 函数接口：

| 入参        | 含义                                                         |
| ----------- | ------------------------------------------------------------ |
| num_tokens  | 每个 rank dispatch 的 最大 token 数， 所有的 rank 必须是相同的值 |
| hidden      | 每个 token 的 hidden size                                    |
| num_ranks   | EP 组的 rank数，一个rank 绑定一个 DCU                        |
| num_experts | EP 总专家数                                                  |
| top_k       | 每个token 选择几个专家                                       |

| 返回值         | 含义                        |
| -------------- | --------------------------- |
| num_rdma_bytes | 低延迟算子需要的 对称堆大小 |

```Python
    def get_low_latency_rdma_size_hint(
        num_max_dispatch_tokens_per_rank: int, 
        hidden: int, 
        num_ranks: int, 
        num_experts: int, 
        top_k: int
    ) -> int:
```

### 实例化 Buffer 

#### 调用示例：

```Plain
buffer = mori_ep.Buffer(group,
                        num_max_dispatch_tokens_per_rank=num_tokens,
                        top_k=num_topk,
                        num_rdma_bytes=num_rdma_bytes,
                        low_latency_mode=True, 
                        num_qps_per_rank=4, 
                        convert_stand_alone=True,
                        num_experts=num_experts
                        )
```

#### 函数接口：

| 入参                              | 数据类型                       | 含义                                                         | 特性支持情况 |
| --------------------------------- | ------------------------------ | ------------------------------------------------------------ | ------------ |
| group                             | torch.distributed.ProcessGroup | torch `Dist` 模块 建立的 通信组，用来交换`rochshmem`初始化需要的必要信息 |              |
| num_max_dispatch_tokens_per_rank  | int                            | 每个 rank Dispatch的 最大 Token 数，每个 rank 持有相同数量的 Token |              |
| top_k                             | int                            | 每个 token 选择 多少个专家                                   |              |
| convert_stand_alone               | Bool                           | 是否 进行单独的 布局转换                                     | 仅支持True   |
| num_experts                       | int，默认值256                 | 专家数                                                       |              |
| num_nvl_bytes                     | int，默认值 0                  | `nv-link` 缓冲区的大小（字节）                               | 仅支持0      |
| num_rdma_bytes                    | int，默认值 0                  | 低延迟算子需要使用的 RDMA 缓冲区大小                         |              |
| low_latency_mode                  | Bool，默认值 False             | 是否是低延迟模式                                             | 仅支持 True  |
| num_qps_per_rank                  | int, 默认值24，要改成4         | 每个rank 上使用QP 数，推荐 4                                 |              |
| allow_nvlink_for_low_latency_mode | Bool，默认值 True              | 对齐 deepep 接口，实际无作用                                 |              |
| allow_mnnvl                       | Bool，默认值False              | 对齐 deepep 接口，实际无作用                                 |              |
| explicitly_destroy                | Bool, 默认值 True              | 是否显示调用 mori-ep 销毁函数                                |              |
| enable_shrink                     | Bool，默认值False              | 对齐 deepep 接口，实际无作用                                 |              |

返回值 ：

​	mori-ep 对象实例

### low_latency_dispatch

目前仅支持 hook = False.且Dispatch 仅支持 bf16

#### 调用示例：

```Plain
packed_recv_x, packed_recv_count, handle, event, hook = \
        buffer.low_latency_dispatch(
                                    x, 
                                    topk_idx, 
                                    topk_weights,
                                    num_tokens, 
                                    num_experts
                                    )
```

#### 函数接口：

| 入参                               | 数据类型               | 含义                                                         | 支持情况                                                   |
| ---------------------------------- | ---------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- |
| x                                  | torch.Tensor           | size: [num_tokens, hidden]<br />数据类型：torch.bfloat16<br />当前 rank 上持有的 token 数据 |                                                            |
| `topk_idx`                         | torch.Tensor           | size: [num_tokens, num_topk]<br />数据类型：torch.int64<br />当前 rank 上 token 选择 `topk` 个 专家 |                                                            |
| topk_weights                       | torch.Tensor           | size：[num_tokens, num_topk]<br />数据类型：torch.float32<br />当前 rank 上 token 选择 topk 个专家对应的权重 |                                                            |
| num_max_dispatch_tokens_per_rank   | int                    | 当前 rank 上持有的 token 总数，所有rank 上必须持有相同的值   | 当前 rank 上持有的 token 总数，所有rank 上必须持有相同的值 |
| num_experts                        | int                    | EP下，专家总数                                               |                                                            |
| cumulative_local_expert_recv_stats | Optional[torch.Tensor] | 记录 等待时间                                                | 目前不支持                                                 |
| dispatch_wait_recv_cost_stats      | Optional[torch.Tensor] | 记录 等待时间                                                | 目前不支持                                                 |
| use_fp8                            | Bool，默认值 False     | 是否启用 FP8 转换                                            | 目前不支持                                                 |
| round_scale                        | Bool，默认值 False     | 是否将缩放因子四舍五入到 2 的幂次方                          | 目前不支持                                                 |
| use_ue8m0                          | Bool，默认值 False     | 是否使用 UE8M0  作为 scale 的格式                            | 仅支持 False                                               |
| async_finish                       | Bool，默认值 False     | 是否异步结束                                                 | 仅支持 False                                               |
| return_recv_hook                   | Bool，默认值 False     | 是否返回 hook                                                | 仅支持 False                                               |

| 返回值       | 数据类型       | 含义                                                         | 支持情况 |
| ------------ | -------------- | ------------------------------------------------------------ | -------- |
| `recv_x`     | `torch.Tensor` | size: [num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden] <br />数据类型:  torch.bfloat16<br /> 存储每个本地专家接收到所有 Token 数据 |          |
| `recv_count` | `torch.Tensor` | size:[num_local_experts]<br />存储 每个本地专家从所有 Rank 接收的 Token 总数 |          |
| `handle`     | `Tuple`        | Combine 算子需要 Dispatch 提供的数据元组                     |          |
| `event`      | `EventOverlap` | 执行 send kernel 后的一个 event                              | 暂不支持 |
| `hook`       | `Callable`     | 用于调用 `recv` 算子的函数钩子                               | 暂不支持 |

### low_latency_combine

#### 调用示例：

```Plain
combined_x, event, hook = buffer.low_latency_combine(
                                                    simulated_gemm_x,
                                                    topk_idx,
                                                    topk_weights,
                                                    handle
                                                    )
```

#### 函数接口：

| 入参                         | 数据类型               | 含义                                                         | 支持情况   |
| ---------------------------- | ---------------------- | ------------------------------------------------------------ | ---------- |
| x                            | torch.Tensor           | deepgemm 计算结果                                            |            |
| `topk_idx`                   | torch.Tensor           | size: [num_tokens, num_topk]<br />数据类型：torch.int64<br />当前 rank 上 token 选择 `topk` 个 专家 |            |
| topk_weights                 | torch.Tensor           | size：[num_tokens, num_topk]<br />数据类型：torch.float32<br />当前 rank 上 token 选择 topk 个专家对应的权重 |            |
| handle                       | tuple                  | Dispatch 返回的 数据 元组                                    |            |
| use_logfmt                   | bool，默认值False      | 是否支持 10bit 量化                                          | 目前不支持 |
| zero_copy                    | bool，默认值False      | 是否支持 零拷贝赋值                                          | 目前不支持 |
| async_finish                 | bool，默认值False      | 是否异步结束                                                 | 目前不支持 |
| return_recv_hook             | bool，默认值False      | 是否 返回 recv 算子的 Hook                                   | 目前不支持 |
| out                          | Optional[torch.Tensor] | 输出 Tensor, **这里不要传入**                                |            |
| combine_wait_recv_cost_stats | Optional[torch.Tensor] | 记录算子等待时间，用于排错                                   | 目前不支持 |



| 返回值       | 数据类型       | 含义                                                         | 支持情况 |
| ------------ | -------------- | ------------------------------------------------------------ | -------- |
| `combined_x` | `torch.Tensor` | Token 选择的 topk 个专家计算后的结果，size: [num_combined_tokens, hidden] <br />数据类型：torch.bfloat16 |          |
| `handle`     | `Tuple`        | Combine 算子需要使用的 数据                                  |          |
| `event`      | `EventOverlap` | 执行 send kernel 后的一个 event                              | 暂不支持 |
| `hook`       | `Callable`     | 用于调用 `recv` 算子的函数钩子                               | 暂不支持 |

### 资源回收：

#### 调用示例

```Plain
buffer.destroy()
```

