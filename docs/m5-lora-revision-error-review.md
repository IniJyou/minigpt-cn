# M5修订失败样本与训练覆盖复核

2026-10-19。对[45步修订结果](m5-lora-revision-results.md)做只读复核：**发现具体形态覆盖缺口、原训练示范回放减少，以及已有示范未稳定迁移三类现象；尚不能确定单一质量原因。未发现需要改动当前规范或参考标签的确证。** 本轮0训练、0生成，不改标签、评分或历史结果，test继续封存。

## 分母与口径

逐个检查新开发集24条50块、旧valid15条62块，合计39条112块；配对读取旧、新step45共78份原始输出及冻结错误报告。39条均有复核记录，包括4条新旧都完全正确的样本。训练侧读取合并84条183块、实际45步180条次顺序与既有token布局。

官方指标直接引用封存metrics，不再次调用评分器。结构失败的整条仍全部计错；原始字段只用于定位问题，没有把lemma替换成theorem、补编号或另算修补分数。

| 配对变化 | 新检查：样本 / 块 | 旧valid：样本 / 块 |
| --- | ---: | ---: |
| 原来错、现在全对 | 12 / 12 | 2 / 14 |
| 原来全对、现在错 | 2 / 9 | 0 / 2 |
| 新旧都全对 | 2 / 14 | 2 / 33 |
| 新旧都错 | 8 / 15 | 11 / 13 |

块指“所有字段严格正确”，样本指整片段完全匹配。全体112块为26改善、11退步、47保持全对、28仍错；新候选仍失败21/39条。类别正确数的退步不必与块全字段退步数相同：旧块即使kind正确，也可能已因title/reason错误而不算全对。

## 全部失败样本

新检查的10条失败如下，前缀均为m5-rev1-；其余14条已逐个核对，包含P4两组完整pair成功等改善。

| 样本后缀 | 新候选的问题 | 参考依据与训练覆盖 |
| --- | --- | --- |
| p1-h01-a | 漏取Proof标题 | 粗体后紧接冒号应取标题；有同形态训练正例 |
| p1-h02-a | 漏取整个ATX标题 | 原文完整标题应保留Definition前缀；无英语ATX Definition训练形态 |
| p1-h02-b | definition误归body | 列表前缀使title=null，不取消角色；已有3块同类训练 |
| p2-h01-b | Examples误归body | v1.2已明确映射example；只有1块普通Examples训练示范 |
| p3-h01-a | Lemma 2误归body | 明确引理应为theorem；无普通/编号Lemma且title=null的训练正例 |
| p3-h02-b | 引用Example误归example，句中Proof字符串误归proof | 引用及正文提及应为body；已有引用负例，没有body中的Proof词面负例 |
| p5-h01-a | ambiguous误为mixed_roles | 单一Definition/Theorem复合标记；已有同形态训练 |
| p5-h01-b | mixed_roles误归body | 同一列表项内独立Definition和Theorem；已有同结构训练 |
| p6-h02-a / b | 非法kind=lemma | 两条7/8块样本整条失败，共15块；编号未漏，不记作覆盖遗漏 |

旧valid的11条失败如下。4条完全正确为v04、v08、Amann a01和a04；v08补回末块后五块全对，是实质改善。

| 样本 | 新候选的问题 |
| --- | --- |
| m5-eval-v1-v01/v02/v03 | 漏取题目标题；v02/v03的解答块误归body。两个题目标题是本次块全字段退步 |
| m5-eval-v1-v05 | ATX“定义”应有标题，仍漏取；三个无独立角色标记的列表项已修复为body |
| m5-eval-v1-v06 | 引用块内部例题仍误触发example |
| m5-eval-v1-v07 | 普通body块的粗体冒号标题仍漏取；body同样适用标题规则 |
| m5-amann-v1-a02 | 编号粗体Proposition误归heading；证明续段被编成heading/title=Proof，跨块继承了上块标记 |
| m5-amann-v1-a03/a05 | 两个编号粗体Proposition由theorem退为body；a05的Proof仍多取标题 |
| m5-eval-v3-amann-ii-01 | 编号粗体Examples仍误归heading并多取标题；其Proof标题已修复 |
| m5-eval-v3-smith-01 | Definition/Theorem应为uncertain/ambiguous，误归body；两段普通正文已修复 |

旧valid的theorem正确数2/3→0/3，uncertain为1/1→0/1，解释了多数body改善时macro-F1仍下降。非空标题9/16看似不变，实际是v08恢复两个标题与v02/v03丢失两个标题相互抵消。不能据总分不变判断这些边界没有变化。

## 实际训练曝光

每类“块出现过”与“训练中呈现多少次”分开统计；下面不是各类独立监督token数。

| 标签/目标 | 原36条中块数 | 合并84条中块数 | 旧step45块呈现次数 | 新step45块呈现次数 |
| --- | ---: | ---: | ---: | ---: |
| body | 33 | 50 | 165 | 105 |
| heading | 10 | 19 | 50 | 39 |
| definition | 9 | 25 | 45 | 51 |
| theorem | 9 | 20 | 45 | 43 |
| proof | 8 | 20 | 40 | 42 |
| example | 9 | 19 | 45 | 42 |
| other | 10 | 18 | 50 | 37 |
| uncertain | 7 | 12 | 35 | 25 |
| 非空title（与上列重叠） | 37 | 50 | 185 | 105 |
| mixed_roles（uncertain子集） | 5 | 7 | 25 | 14 |
| ambiguous（uncertain子集） | 1 | 3 | 5 | 6 |
| insufficient_context（uncertain子集） | 1 | 2 | 5 | 5 |

旧step45是原36条各5次，共180条次；新step45中原36条共77条次，新增48条共103条次，每条各2或3次。原训练内容的监督token曝光12230→5161，新增内容4496，总计9657。扩充样本后保持45步，使训练混合比例与回放频次同时改变，不能将结果单独归因于新增样本质量。

旧日志未记录逐步sample_ids，旧曝光由冻结shuffle规则和五个完整epoch重建；重建的45组监督token与旧日志逐组一致。新曝光直接取实际training.jsonl。183个训练块位置有161份不同全文，P6成对样本共享前缀；块数也不等于独立内容数。

回放减少是待验证因素，尚不是因果结论。比如body块呈现次数也从165降至105，但旧valid的body正确数反而26/38→36/38；仅靠某类曝光总数不能解释全部变化。

## 具体形态覆盖

以下只统计输入块text，不把每次重复的系统提示当训练实例。完整词检查区分Example与Examples；形态统计不替代语义标签。

| 形态 | 训练块位置 / 分组家族 | 新实验呈现次数 | 解释 |
| --- | ---: | ---: | --- |
| Lemma正例全部 | 2 / 1 | 4（旧实验10） | 两条分别为ATX和粗体冒号，都有非空title |
| Lemma正例且title=null | 0 / 0 | 0 | 普通Lemma:、Lemma 2:没有直接覆盖 |
| 编号在英语角色词之前 | 0 / 0 | 0 | 缺少编号粗体Proposition/Examples等组合 |
| Examples正例 | 1 / 1 | 2 | 只有普通前缀，没有编号粗体组合 |
| body正文中包含完整词Proof | 0 / 0 | 0 | 其他角色词有提及负例，Proof没有 |
| 英语ATX Definition | 0 / 0 | 0 | 有ATX Theorem和中文定义作类比 |
| 粗体题目/解答 | 0 / 0 | 0 | 有其他标题与范围外结构示范，但没有该直接形态 |
| 引用Example且target=body | 1 / 1 | 3 | 仍误触发，属于有示范但未稳定迁移 |
| 粗体Proof紧接冒号 | 2 / 2 | 5（旧实验5） | 同形态正例存在且呈现总量没减，仍发生标题遗漏 |
| 粗体Proof后空格接正文 | 1 / 1 | 2 | 本应title=null，部分新上下文仍多取标题 |
| 列表粗体Definition | 3 / 3 | 7 | 列表内definition误归body，不能说完全没教过 |
| 英语Definition/Theorem→ambiguous | 2 / 2 | 4 | reason仍混淆，旧valid还出现kind退步 |
| 同项Definition+Theorem→mixed_roles | 1 / 1 | 2 | 对应新检查由正确退为body |
| body带非空标题 | 1 / 1 | 2（旧实验5） | 正文标题规则有实例，尚未稳定迁移 |

具体训练定位与目标、频次在coverage_profiles.json列出，例如Lemma来自m5-sft-v1-spec-s11和m5-sft-v1-diagnostic-d04，普通Examples来自m5-rev1-p2-t01-b，引用Example来自m5-rev1-p3-t03-b。开发样本不复制进train，也不把已查看开发集称作新的盲测。

## 规范与候选标签

[v1.2规范](m5-task-spec-v1.2.md)及[冻结完整提示](../configs/m5_structure_prompt_v1_2.txt)已明确Lemma→theorem、Examples→example、引用/代码/正文提及不生效、角色与标题独立、复合标记与独立多角色使用不同reason。当前失败不能靠新增lemma输出类、改uncertain标签或改门槛消除。

本次逐块核对183个训练块及112个开发块：参考annotation的合法性、title与冻结提取契约均未发现冲突；对39个开发样本的语义角色复核，也没有发现必须重标或升级规范的确证。无显式标记的数学定义仍按body，证明续段不继承上块proof，数学内容正确性不属于本次复核。复核仍由助手完成，不构成独立人工金标或“标签绝无错误”的保证。

## 修订依据与下一项建议

可以确认的是形态缺口、已有示范上的迁移失败、回放与混合比例变化；尚无证据判断它们各占多大因果贡献，也不足以判定LoRA容量、学习率或训练实现是主要原因。

尤其尚未生成训练样本上的预测，不能判断模型在见过的正反例上是否已经学会契约。建议下一项先**定义固定训练样本回放诊断**：锁定现有训练样本、旧/新step45、相同提示和解码、明确调用上限，区分“已见示范也失败”与“未覆盖形态失败”，再决定训练分布修订。只定义，不在本轮实现或执行；不直接续训或修改rank/学习率，也不把训练集诊断当泛化成绩。

若后续修订训练材料，应明确列出普通/编号Lemma、编号粗体Proposition/Examples、Proof提及负例等覆盖，并记录原样回放和标题/reason目标的曝光。输出合法枚举约束若另作实验，应单独定义，不能修补本轮历史预测或与训练改动混在同一次归因中。

## 证据与验收

logs/m5_lora_revision_error_review_20261019_v1包含：39条case_ledger与manual_case_review、112条block_ledger、183条training_ledger、training_coverage、14类coverage_profiles、reference_review及acceptance。525项核对通过，453份前项冻结产物原字节未变；官方分数不变，原模型/数据/代码/运行清单均未改。脚本首次遇旧日志缺sample_ids、验收首次遇零计数缺键，已修正只读聚合逻辑，说明另存；没有因此重跑模型或覆盖结果。静态检查与文档链接另记final_qa.json；本轮不重跑产品CPU单元测试。

下一任务以[current-work](current-work.md)为准，Note2Lecture继续暂停。
