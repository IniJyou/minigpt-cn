# M5 固定数据训练拟合诊断入口

按[冻结定义](m5-train-fit-plan.md)另存实现、测试、WSL入口和configs/m5_train_fit_v1/runtime.json。定义及旧运行器/数据/标签/权重保持原字节，原定义中的pending标记不回写。本交付轮仅完成CPU精确状态恢复与实际默认inspect，没有执行真实基座加载、GPU预检、60步续行、53次生成或test。

默认命令为：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_train_fit_wsl.sh
```

不传phase即inspect。只读核对冻结摘要、84条完整train的真实target/EOS布局、105组固定日程及45组实际前缀、13353新增监督tokens，另核对14条训练回放、24条新检查、15条旧valid的完整输入/目标身份及复用基准分数。target不进入生成消息；不截断、packing或修补标签。

inspect会读取本地基座配置/tokenizer、在meta设备构造同版本Qwen/LoRA结构，并只读加载约20MB的adapter及保存的Adam/RNG到CPU，以核对392个参数的名称、枚举顺序、dtype、权重摘要、Adam计数、scheduler和部分epoch游标。**它不加载0.6B基座权重，也不分配真实基座参数或初始化CUDA。** 这比仅看张量shape多了参数构造路径、位置身份与实际adapter摘要的交叉核对。

原optimizer未显式存param_names；位置映射来自冻结原构造代码/同版本模块注册顺序、meta注册表、连续ID和真实模型加载后再次核对的同一注册表。不是从相同shape推测语义，也不是独立保存的历史参数名证明。未来加载时先核对原初始adapter摘要，复制原step45的每个命名tensor后核对训练后摘要，再核对optimizer/scheduler状态；注册顺序、源代码、状态或摘要变化即停止。

恢复过程不重置Adam、不重新warmup、不额外scheduler.step。模型、optimizer、数据和预检操作结束后，最后恢复Python/NumPy/Torch/CUDA RNG；第一次新更新必须是绝对step46，lr=7.941176470588235e-5。只执行冻结后缀60组/240条次，累计监督曝光23010；仅保存step105及完整状态，写入新的显式parameter_registry和下一游标。真实GPU精度/非确定性仍不能由CPU等价试验完全证明。

未来phase仅为inspect、preflight、train-evaluate。preflight独占创建logs/m5_train_fit_experiment_20261019_v1，最多2次最长1961-token前后向、0优化步；检查结束需清梯度、核对权重与Adam未变、完整恢复RNG并出具封存回执。train-evaluate必须先验证同runtime、同状态身份且2次/0步的独立预检回执，已有阶段目录不覆盖、不重试、不恢复失败运行。

训练在每次microbatch前持久记录绝对microbatch/sample_id/step并flush/fsync，预检最多2次、正式续行最多240次；第241次或日程不符全局停止。loss仍按整个四条组的监督token数归一化，不再除4；每组一次裁剪、optimizer.step及scheduler.step。全过程在microbatch及optimizer边界检查墙钟、allocated/reserved与GPU总量，reserved超过总量减1GiB、读数不一致、非有限值或OOM即停。历史reserved异常未解释，不放宽门槛或承诺全程8GB内。

只有60步完整完成、step105目录树锁定后才允许53次固定greedy推理，顺序为训练原例14→新检查24→旧valid15。切换为全参数冻结/eval/inference_mode，同提示、seed42及显式do_sample=False/use_model_defaults=False。每次先核对有效配置与固定身份，再fsync记账，随后才前向；第54次、无记账返回、token预算或配置异常全局停止。普通逐条生成异常保留空回复和全部分母，OOM等致命异常保留失败目录，不重试。

三组分开报告严格分数、所有样本字段错误、标题遗漏/多取、reason混淆、非法kind和结构失败。训练回放另报两组/五完整pair，新检查另报12完整pair与P1–P6；所有分数与冻结修订step45的同集合输出比较，不重跑基座或控制组。result.json记录拟合改善、两套开发各八项不退步、原九项绝对门槛及全部指标变化，不改变原selection或test状态。成功回执只在完整训练、53次调用、报告和全部摘要复核完成后产生。

CPU验收新增54项全部通过，连同M5回归为257 passed / 5 skipped，Ruff通过。5项跳过来自Windows缺少可选PEFT/Transformers；实际独立M5环境另用随机极小Qwen/PEFT验证：非零LoRA dropout=0.05、非重入checkpointing并保留RNG、Adam/180步scheduler、45步第三轮偏移12的保存重建，与不中断105步的逐参数/optimizer/scheduler/RNG及样本轨迹完全一致；冻结base未变、adapter更新。合计210次为极小模型CPU合成更新，不是本地0.6B训练，未调用generate或初始化CUDA。

实际CPU输入预验收核对84/14/24/15共137条记录角色（含重复引用的14条训练原例）、43项依赖、安装版API、原step45的392个参数身份与保存状态。runtime绑定此证据和实现/测试；最终另封存真实默认main([])的结果与runtime摘要，避免自引用。

证据在logs/m5_train_fit_runner_20261019_v1。下一项是独立GPU预检与只读验收；实际60步/53次留待预检结果核定后推进。Note2Lecture继续暂停。
