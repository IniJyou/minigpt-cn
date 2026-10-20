# M5 独立资源观测入口与CPU验收

2026-10-19。实现[冻结观测定义](m5-memory-observability-plan.md)中的独立入口，交付范围为代码、CPU故障测试与默认inspect。**未运行真实GPU诊断，未加载0.6B权重或执行真实续训、生成。** 原失败原因仍待实际观测，test封存、Note2Lecture暂停。

## 入口与默认行为

- 源码：src/minigpt_cn/m5_memory_observe.py。
- 测试：tests/test_m5_memory_observe.py。
- WSL封装：scripts/m5_memory_observe_wsl.sh，固定独立M5环境并离线运行。
- 新runtime：configs/m5_memory_observe_v1/runtime.json，绑定旧冻结链、新源码/测试、定义与CPU证据。

默认命令仅检查文件：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_memory_observe_wsl.sh
```

inspect校验全部绑定摘要、旧失败定义、原step45来源、25条顺序/布局及6更新/25条/69采样上限。模块顶层和inspect路径仅使用Python标准库，不导入torch、Transformers、PEFT或NumPy，不读取权重张量或构造tokenizer/模型，不调用CUDA或创建实验阶段目录。摘要核对会顺序读取已绑定权重文件的字节，不是加载模型张量。

显式probe阶段供后续单次GPU诊断使用；本入口交付轮不执行。它重新走冻结的43项环境与模型身份检查、真实tokenizer布局核对及完整step45状态恢复，不把inspect的纯文件结果当GPU资源证明。旧失败目录和新目录已存在时均不能覆盖或恢复；新入口没有训练到105步、额外预检、评测、重试或权重保存阶段。

## 保存哪些证据

MemorySampler保持原顺序：synchronize → max_memory_allocated → max_memory_reserved → device total。不增加CUDA同步、allocator查询、峰值reset或运行中清缓存。采样中途失败会保存已经读取的字段和同步/读取阶段；完整返回值只复制一次，失败后不重读。

ObservedGuard仍检查1800秒墙钟、总量变化、0≤allocated≤reserved以及reserved≤总量−1GiB。它保存全部可判定条件，缺数据的条件为unknown；异常类型遵循原短路优先级。失败记录包含实际observed、阈值、样本身份、阶段、整组分母、耗时及堆栈。

observations.jsonl的进入/结果、forward/backward及optimizer/scheduler进入/返回事件均flush+fsync；追加前检查现存记录未被改动。任何写入或fsync错误立即停止下一项模型操作，stderr尽力保留原错误，异常链不因留证失败丢失。optimizer调用进入但未返回时，即使权重可能已经改变，也只记更新状态未知。清理前先写失败证据；清理异常不会覆盖原失败。

## 有界控制流

复用原collate、监督token计数、完整状态恢复、梯度有限检查与范数裁剪。保留step46–51四条一组的loss归一化、原学习率与全部guard位置，含on_step里原有的重复检查。

step52仍从完整四条组计算分母417，只执行第一条n08，张量形状`[1,1791]`。其反向和原guard返回后主动结束，禁止第二条样本和step52的optimizer.step。无提前失败时恰好25次前后向尝试、6次更新、69次资源采样；不保存模型权重。部分组梯度仅存在于该次进程，不形成可续训检查点。

新目录固定为logs/m5_memory_observe_experiment_20261019_v1，独占创建。若实际门槛失败，保存failure.json及已有账本，不生成成功回执；若抵达上限仍未复现，回执状态是bounded_probe_complete_no_reproduction、training_complete=false，不算完成原60步续训或解决资源问题。

## CPU验收与实际检查

新增42项CPU测试覆盖三项门槛的全部八种组合、阈值相等、设备总量字段缺省、非法或缺失峰值、原异常优先级、采样各阶段抛错、超时、日志被改动/删除、fsync失败和更新边界。另用CPU替身走完整运行流程，分别检查有界回执、资源失败现场，以及清理异常不覆盖原失败。

CPU轨迹对照使用小随机模型、冻结基座参数、0.05 dropout及非重入checkpointing保留RNG；用真实84条ID、真实输入/目标长度和冻结样本顺序构造合成token值，完成合成step45后比较原控制流与观测控制流的25条后缀。参数、Adam、scheduler、Python/NumPy/Torch RNG、梯度、样本顺序、六步loss及norm逐项精确相同；第25条分母确认为417，基座不变、可训练参数确有更新。这是CPU合成数值证明，不是0.6B训练或CUDA峰值复现。

全部M5回归结果和新源码摘要绑定在logs/m5_memory_observe_runner_20261019_v1/cpu_checks.json；五项skip为Windows缺少的可选PEFT/Transformers依赖，未安装依赖或绕过检查。初次完整流程测试发现新输出目录父目录缺失时创建失败，已改为独占创建所需父目录并完整回归；初始记录保留。

实际默认main([])在禁止导入模型库的保护下执行；Windows及独立M5 WSL入口检查记录一并保存。它们核对新runtime和真实文件布局，没有GPU资源结论。运行证据、前后活动文档快照及最终摘要在logs/m5_memory_observe_runner_20261019_v1，旧冻结产物保持原字节。

下一项才是运行冻结入口的一次有界GPU观测；资源门槛和6更新/25条上限不变，0生成/test。不自动重试旧失败目录或接续原60步实验；唯一任务入口见[current-work.md](current-work.md)。
