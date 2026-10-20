# M5成对训练修订：45步与87次开发对照结果

2026-10-19，按[修订定义](m5-lora-revision-plan.md)、[已验收数据](m5-lora-revision-data.md)和[冻结入口](m5-lora-revision-runner.md)，完成一次45步训练及87次greedy开发对照。**部分边界和覆盖改善，但新检查与旧valid的macro-F1均下降，未通过预设的方向改善和绝对质量门槛。候选保留为实验产物，test继续封存。**

## 实际执行

从固定Qwen3-0.6B基座及原零B初始化重新训练，未续训旧adapter。沿用任务规范v1.2、完整提示、rank8/alpha16/dropout0.05、196处/5,046,272个FP32可训练参数、BF16基座、AdamW及原10步warmup/180步调度前缀。

合并train84条183块；45优化步、180条次、9657个监督token曝光，实际为两整轮加第三轮12条，训练耗时约147.30秒。旧step45的曝光为12230，两个实验仅优化步数相同，不是等token预算或相同轮数。只保存并锁定step45，没有按分数挑选其他检查点。

生成依次为基座新检查24、旧step45新检查24、新候选新检查24、新候选旧valid15，精确87次，共6042个新增tokens，纯生成耗时约474.34秒（不含摘要核对与加载）。全部EOS结束，运行异常0、输出上限0，无重试。实际合并配置及持久账本均为greedy_search，显式do_sample=False/use_model_defaults=False；没有改变提示、标签、评分或门槛。旧valid的基座及旧step45分数复用已封存greedy结果，未重新生成。

运行清单为configs/m5_lora_revision_v1/runtime.json，SHA256保持06844ee745cf6d1803f1f2f6230d89f5c5563b51f52943163e8ad6a70b2bf230。

## 新开发检查：24条50块

结构失败整条计错，不修补非法标签或从无效回复中提取部分分数。

| 指标 | 同集基座 | 旧step45 | 新step45 |
| --- | ---: | ---: | ---: |
| JSON有效 | 23/24 | 24/24 | 24/24 |
| schema有效 | 3/24 | 23/24 | 22/24 |
| 编号完整有序（含schema前提） | 3/24 | 23/24 | 22/24 |
| 块全部字段正确 | 0/50 | 23/50 | 26/50 |
| 整片段完全匹配 | 0/24 | 4/24 | 14/24 |
| 八类macro-F1 | 0.000000 | 0.663539 | 0.614194 |
| 非空标题正确 | 0/6 | 4/6 | 2/6 |
| 非空reason正确 | 0/7 | 3/7 | 3/7 |
| 两成员均完全匹配的pair | 0/12 | 0/12 | 5/12 |

新候选的块全字段正确率增加6个百分点、整片段匹配增加10条，但macro-F1下降约0.049346，非空标题从4/6降至2/6。它只通过九项绝对门槛中的JSON、整片段匹配和相对同集基座增益3项；不能用局部改善替代全部门槛通过。

新候选的m5-rev1-p6-h02-a/b均输出非法kind=lemma，导致两个7/8块样本共15块按严格口径全部计错；旧step45只在其中a发生此问题。这两条是schema失败，不能写成漏块，也不能把lemma改为theorem后重新评分。标题仍有遗漏或无依据添加，具体原文与预测均在errors.jsonl保留。

每种现象各2个pair，完整pair要求两成员全部字段均正确：

| 现象 | 旧step45完整pair | 新step45完整pair |
| --- | ---: | ---: |
| P1 标题边界 | 0/2 | 0/2 |
| P2 英文显式角色词 | 0/2 | 1/2 |
| P3 生效标记与正文提及/代码引用 | 0/2 | 0/2 |
| P4 证明请求与证明正文 | 0/2 | 2/2 |
| P5 uncertain及reason | 0/2 | 1/2 |
| P6 追加末块 | 0/2 | 1/2 |

## 旧valid回归：15条62块

| 指标 | 旧step45 greedy（复用） | 新step45 greedy |
| --- | ---: | ---: |
| JSON有效 | 15/15 | 15/15 |
| schema有效 | 15/15 | 15/15 |
| 编号完整有序 | 14/15 | 15/15 |
| 块全部字段正确 | 35/62 | 47/62 |
| 整片段完全匹配 | 2/15 | 4/15 |
| 八类macro-F1 | 0.636535 | 0.571067 |
| 非空标题正确 | 9/16 | 9/16 |
| 非空reason正确 | 4/7 | 4/7 |

块匹配增加与macro-F1下降同时成立：body正确数从26/38升至36/38，heading从7/9升至9/9；theorem从2/3降至0/3，uncertain从1/1降至0/1，example仍为0/1。固定八类等权macro-F1反映了类别间的差异，不能用多数类的增量抵消少数类退步。

旧valid八项不退步检查通过7项，macro-F1失败；九项绝对门槛仅通过JSON、schema、覆盖及相对旧valid基座增益4项。新检查的macro-F1也未高于旧step45，因此directional_improvement=false，quality_passed_both_development_sets=false。旧selection仍为未接纳，未改成选择本轮候选。

## 资源与证据

训练峰值allocated为5,037,585,920 bytes（约4.692 GiB），reserved为19,058,917,376 bytes（17.75 GiB）。reserved再次高于设备标称总量，成因未核定，原始日志完整保留；[预检通过](m5-lora-revision-preflight.md)不能证明完整训练全程在8GB以内。

实际产物位于logs/m5_lora_revision_experiment_20261019_v1/train-compare：训练日志与状态、唯一step45、candidate_lock、87次调用账本、四条件原始预测/错误/指标/runtime、pair_summary/result及成功receipt。新adapter文件为checkpoints/step_000045/adapter_model.safetensors，SHA256为ca571b7fb6978ab469efacf3c24ce2279b19e4b069f7fc800d7f99c0cf4573d5。

独立验收位于logs/m5_lora_revision_train_compare_20261019_v1：647项核对通过，410份前项冻结产物原字节不变，严格评分、门槛及完整pair复算一致。实际M5环境在CPU核对87条输入token和原样解码；检查392个FP32 adapter张量、392份optimizer状态均到45步、scheduler与第三轮偏移12、数据顺序及RNG保存。验收没有构造基座、初始化CUDA或产生额外生成，未读取test记录。运行源码未改，不重跑前项CPU单元测试；本轮静态检查另记final_qa.json。

新检查为助手原创合成开发材料，旧valid已反复查看，标签没有独立人工金标，不能称盲测或泛化提升。本轮结果支持“此次固定预算修订改善了部分边界，但未满足接纳条件”，不足以确定数据、训练长度或模型规模的单一原因。下一任务以[current-work](current-work.md)为准；先只读复核失败与退步样本，不追加训练、生成或解封test。Note2Lecture继续暂停。
