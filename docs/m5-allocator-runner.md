# M5 单项分配器候选入口与CPU验收

2026-10-19。针对[有界观测确认的reserved余量越界](m5-memory-observe-results.md)，另存独立入口及运行配置。**本轮只实现并完成CPU/默认inspect验收，0真实权重加载、0CUDA初始化、0GPU对照、0生成。** expandable_segments只是待验证候选，尚不称已修复资源问题；test封存、Note2Lecture暂停。

## 入口与隔离方式

- src/minigpt_cn/m5_allocator_run.py：标准库启动器、配置留证、显式GPU worker。
- tests/test_m5_allocator_run.py：新49项CPU测试。
- scripts/m5_allocator_wsl.sh：固定独立M5环境、离线运行；脚本不export分配器配置。
- configs/m5_allocator_v1/runtime.json：绑定旧738份冻结产物、新实现和本地接口/CPU证据。

默认命令只做文件与环境契约检查：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_allocator_wsl.sh
```

默认inspect不导入Torch、Transformers、PEFT或NumPy，不构造模型/tokenizer、不读取权重张量或调用CUDA，不生成子进程或实验目录。摘要检查会读取已绑定文件字节。返回allocator_verified=false，支持状态标为尚未检查；配置字符串存在不等于GPU生效。

显式probe阶段才会启动一个同解释器的新Python子进程。启动器复制环境，只将子进程的PYTORCH_CUDA_ALLOC_CONF设为expandable_segments:True，不修改父进程或全局环境。继承值仅允许未设置、空字符串或该精确候选；其他配置、别名或cache开关非空均拒绝，不静默覆盖、拼接或归一化。

launch.json记录五个相关环境变量的实际继承值和子进程值、命令、runtime身份及父进程PID，不导出整个环境。新目录logs/m5_allocator_experiment_20261019_v1独占创建；worker_started.json也独占写入，拒绝复用或二次启动。子进程退出或启动失败后不重试。父进程只有在worker回执、文件摘要、预算及实际配置证据均通过时才接受正常返回；launch_result.json单独记录父进程判定，不纳入先前worker封存的文件集。

## 已核对的固定接口

本轮在/home/luoyu/.venvs/minigpt-cn-m5中仅用标准库读取已安装Torch2.8.0+cu126的memory.py及两个allocator头文件，保存摘要和文本快照；未导入Torch或调用配置API。worker将重新验证相同环境前缀、版本及三个文件摘要。

本地Python包装提供get_allocator_backend和_snapshot接口。与v2.8.0原生实现对应的快照元数据包括最终配置字符串、expandable_segments有效布尔值，以及每个segment的is_expandable。配置getter在不支持的平台上可返回false并警告，因此仅记录环境字符串不够。[v2.8.0配置与快照实现](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDACachingAllocator.cpp)、[Python快照字段映射](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/csrc/cuda/Module.cpp)

expandable_segments在Torch2.8中是实验选项，用于改善分配大小变化的场景；这只支持把它作为候选，不证明本次故障由碎片造成，也不保证在本机有效。[PyTorch2.8内存管理说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)

## GPU阶段的生效门槛（本轮未执行）

worker先确认模型库尚未导入、候选已在进程启动前设置，再验证本地接口身份、固定43项环境、真实输入布局和原step45身份。

| 检查时点 | 必需证据 | 不满足时 |
| --- | --- | --- |
| 真实基座加载前 | 后端为native；原生快照最终字符串精确匹配候选；有效expandable_segments布尔值为true | 保存allocator_before_load.json与失败记录，停止，不加载真实权重 |
| 完整step45恢复后、首次前后向前 | 上述条件保持，device0实际存在total_size>0且is_expandable=true的segment | 保存allocator_after_restore.json与失败记录，停止，0训练前后向 |

两次配置快照是明确列出的配置检查，不是训练前后向预检，也不计入原69次资源采样。缺字段、警告、后端不符、读取失败或无实际可扩展segment均停止；不自动切换backend、启用更多选项或调用动态setter。只保存配置元数据与分段数量，不写完整显存分块地址或张量。

后续是否能抵达原n08边界、峰值是否下降，仍需独立GPU对照。原进程未记录继承allocator配置，本次也不假设原基线使用默认native设置或两个GPU权重轨迹逐位相同。

## 计算、预算与产物边界

直接调用冻结m5_memory_observe.bounded_loop，不复制训练数学。模型构造、完整step45 adapter/392份Adam/scheduler/RNG/部分epoch游标恢复、原学习率、精度、数据、标签、loss归一化、梯度裁剪和guard位置均复用。

GPU对照仅拟定一次，最多step46–51的6更新、microbatch181–205的25次前后向及69次原guard；末条n08仍使用完整step52组的417分母。反向后原guard通过也结束，不进入第二条或step52更新。reserved≤设备总量−1GiB和1800秒训练guard墙钟保持不变；配置恢复后检查的耗时计入该guard计时。

0额外训练预检、0生成/评测/test、不保存权重或checkpoint。失败仅留failure和现场，不写成功回执；未复现时只能写有界观测完成且本次未复现，training_complete=false，不能视作完成原60步训练、通过质量门槛或保证全程8GB内。

## CPU验收范围

新49项测试覆盖环境副本及精确冲突规则、真实CPU子进程在模型库导入前获得配置且父环境不变、模型库已导入/启动设置缺失的拒绝、本地API前缀/版本/摘要变化、配置字段不明/不支持/非native后端、字符串true与布尔true区别，以及配置被接受但实际segment缺失时的停止。

CPU替身worker验证加载前拒绝时0权重加载、恢复后拒绝时0训练调用、正常流程对冻结循环的七组/417分母委托、25条账本/6条更新日志、两次配置快照、回执封存和父进程预算/生效状态复核；损坏或伪成功均不能通过。它不代表GPU分配器支持已获验证。

原42项观测CPU测试与精确参数/Adam/scheduler/RNG/梯度对照继续纳入M5回归，训练循环源码保持冻结。因此本轮新增的是配置隔离和门槛证据，没有声称在CPU验证了CUDA分配效果。最终回归、双端受保护默认main([])、接口只读证据及摘要链在logs/m5_allocator_runner_20261019_v1。

下一项直接执行冻结新入口的一次有界GPU候选对照并只读验收；不再拆出额外准备环节，不自动启动完整续训或53次评测。唯一下一任务见[current-work.md](current-work.md)。
