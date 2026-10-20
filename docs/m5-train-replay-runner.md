# M5 固定训练原例回放入口

实现与验收范围为CPU测试、独立M5环境中的实际默认inspect；真实权重加载、训练、生成、test均为0。冻结的[诊断定义](m5-train-replay-plan.md)、原选集、旧运行器和旧实验产物保持原样；运行绑定另存configs/m5_train_replay_v1/runtime.json，定义中的历史pending状态不回写。

默认入口：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_train_replay_wsl.sh
```

不传phase即inspect。它校验冻结产物的传递摘要、UTF-8原行定位、完整input/target身份、14条17块与两组分母、五个完整pair、旧新step45目录树、独立环境43项依赖和安装版生成API。实际tokenizer按原提示处理完整input；target只用于身份核对和评分。输入必须逐条等于冻结token布局且不超过4096，不截断。纯配置代理执行安装版GenerationMixin合并逻辑，验证greedy及显式do_sample=False/use_model_defaults=False，不构造模型、初始化CUDA或创建回放目录。

未来执行入口为同一脚本加`--phase replay`，本交付轮不运行。唯一输出目录固定logs/m5_train_replay_diagnostic_20261019_v1，独占创建后才允许加载权重。旧step45先14条、新step45再14条，均用相同完整记录、seed42、BF16/SDPA、eval/冻结参数/inference_mode、1024输出上限；无基座组、预热、补样、训练、test或参数覆盖入口。两条件分别加载，避免adapter叠加。检查点树和运行绑定在加载前及结束前再次核对。

每次前向前写账本并flush/fsync；顺序、重复、第29次、配置或持久化异常全局停止。OOM/MemoryError不降级为空回复。普通生成异常保留空回复并计入全部分母；没有记账的异常不能被当成有效尝试。失败目录与failure.json保留，不重试、续跑或覆盖；仅完成28次并写齐报告才生成receipt.json。

每条件报告总14条17块、common_train 4条7块、revision_only_train 10条10块，复用冻结严格评分器；结构失败整条零分，不修复输出。report.json包含固定八类macro-F1、逐类/逐字段指标、所有样本字段错误与旧新曝光、非法类别、完整覆盖样本上的标题缺失/多出及reason混淆、结构失败块数、五pair的双成员全字段结果。all_sample_fields.jsonl保留全部14条；result.json另列共同训练原例的四个旧新转变及解释边界。definition/heading支持为0，完美结果的固定八类macro-F1仍为0.75。

这只是所选训练示范的拟合诊断。新增原例上的旧模型失败不作为遗忘证据；共同原例退步也不独立证明单一原因。所选14条全对不等于全train或泛化通过，不套用开发门槛、不选新检查点、不触发test。原开发结果仍引用[修订对照结果](m5-lora-revision-results.md)，不增加开发生成或重新构造缺失形态。

CPU测试检查原行/字节/字段身份、完整上下文与target隔离、默认inspect只读、28次账本与第29次、fsync失败/账本篡改、普通错误全分母、致命失败、预算/配置/检查点变动、严格整条评分、五pair和CLI边界。实际M5验收另禁止模型构造、权重加载、CUDA初始化及generate，直接运行默认main入口；无网络、安装或模型下载。

新增50项定向CPU测试全部通过，连同M5回归为203 passed / 5 skipped，Ruff通过。5项跳过来自Windows环境缺少可选PEFT/Transformers；独立M5环境另核对43项实际依赖、安装版API和配置合并。实际CPU输入预检确认14条输入为1267–1321 tokens，两个完整记录中附带的非focus块也保留。

证据在logs/m5_train_replay_runner_20261019_v1。runtime绑定实现、测试、CPU检查和实际CPU输入预检；最终验收另将默认CLI输出与runtime摘要一并封存，避免运行绑定自引用。
