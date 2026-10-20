# M5 固定训练原例回放结果

2026-10-19。按[冻结定义](m5-train-replay-plan.md)及[已验收入口](m5-train-replay-runner.md)完成旧、新step45各14次，共28次真实回放。**新候选未稳定拟合所选训练原例：共同训练样本从4/4全对降至1/4，新增样本从3/10提高到5/10，总体7/14降至6/14。** 这确认了具体已见原例退步，尚不能确定是曝光、优化、实现或容量中的哪一个因素导致。

没有训练、重试、补充调用、新开发生成或test；原规范、提示、参考标签、运行绑定与检查点不变。Note2Lecture继续暂停。

## 执行与评分口径

运行绑定为configs/m5_train_replay_v1/runtime.json，SHA256=f7987afe0502374c88510d60413bd824dc7058d409d2ccd53138c09d7aacad76。旧step45来自原36条训练，新step45来自修订84条训练；本次各读取同一14条完整原始train记录17块，保留非focus块与原顺序。共同训练4条7块，新增训练10条10块；新增记录为五个完整pair。这里的“新增”只指旧微调没有见过该完整记录，不推断预训练是否见过相关内容。

BF16、cuda:0、SDPA、batch=1，eval/冻结参数/inference_mode，seed42，greedy且显式do_sample=False/use_model_defaults=False，输入1267–1321 tokens，无截断；每次最多1024新tokens。target只供离线评分，不进入生成消息。固定账本1–14为control_step45，15–28为candidate_step45；全部28次正常EOS结束，共936新tokens，生成计时合计86.885秒，无运行异常或长度截断。

沿用冻结严格评分：JSON/schema/编号覆盖失败时整条全部计错，不修补输出。definition和heading支持为0，固定八类macro-F1的本选集上限为0.75；不套用开发接纳门槛，不据此选择检查点或触发test。参考标注仍为助手核定候选，不是独立人工金标或盲测。

## 总体与分组

| 指标 | 旧step45 | 新step45 |
| --- | ---: | ---: |
| JSON合法 | 14/14 | 14/14 |
| schema及完整覆盖合法 | 14/14 | 13/14 |
| 整条全字段正确 | 7/14 | 6/14 |
| 块全字段正确 | 10/17 | 7/17 |
| kind正确 | 14/17 | 10/17 |
| title正确 | 13/17 | 11/17 |
| reason正确 | 14/17 | 10/17 |
| 非空参考title正确 | 5/5 | 0/5 |
| 非空参考reason正确 | 5/6 | 1/6 |
| 固定八类macro-F1 | 0.600000 | 0.394345 |
| 完整pair双成员全字段正确 | 0/5 | 1/5 |

| 分组 | 样本全对：旧→新 | 块全对：旧→新 | macro-F1：旧→新 |
| --- | ---: | ---: | ---: |
| common_train | 4→1/4 | 7→2/7 | 0.625000→0.270833 |
| revision_only_train | 3→5/10 | 3→5/10 | 0.302083→0.445833 |

共同原例s11、d04、s09由全对变错，s14保持全对；没有共同原例改善或新旧都错。新增原例5条改善、3条由旧全对变错、没有保持全对、2条新旧都错；旧模型没有微调过这10条，因此这组变化不作为遗忘证据。

## 全部14条原例

下表“对”表示整条所有字段严格正确。共同训练原例在旧训练各呈现5次、新训练各2次；新增原例旧训练呈现0次、新训练各2次，引用Example负例p3-t03-b为3次。

| 原例 | 旧→新 | 新候选核对结果 |
| --- | --- | --- |
| m5-sft-v1-spec-s11 | 对→错 | ATX Lemma的kind为theorem，但丢失标题Lemma: Identity map；同条Proof块正确 |
| m5-sft-v1-diagnostic-d04 | 对→错 | 练习块输出other却reason=null，违反other必须out_of_scope的schema；三个编号齐全，整条3块仍全部计错 |
| m5-sft-v1-spec-s09 | 对→错 | 中文复合定义/定理标记误归body，丢失title及ambiguous |
| m5-sft-v1-spec-s14 | 对→对 | 同一项Thm/Pf保持uncertain/mixed_roles |
| m5-rev1-p1-t01-a | 对→错 | 粗体Proof后紧接冒号，应取Proof标题，实际title=null |
| m5-rev1-p1-t01-b | 错→对 | 粗体Proof后空格接正文，正确保持title=null |
| m5-rev1-p3-t03-a | 错→对 | 普通Example正确归example且不编造title |
| m5-rev1-p3-t03-b | 错→错 | 相同文字外层为引用，仍误归example，应为body |
| m5-rev1-p3-t01-a | 错→对 | 活动Proposition正确归theorem |
| m5-rev1-p3-t01-b | 错→对 | 正文提及Proposition正确归body |
| m5-rev1-p5-t04-a | 错→错 | Definition/Theorem复合标记仍把ambiguous写成mixed_roles |
| m5-rev1-p5-t04-b | 对→错 | 同一列表项内独立Definition和Theorem误归body，应uncertain/mixed_roles |
| m5-rev1-p5-t03-a | 对→错 | 缺失所指结论的证明回指误归body，应uncertain/insufficient_context |
| m5-rev1-p5-t03-b | 错→对 | 完整中文证明正确归proof且不编造title |

本次没有非法kind，包括没有输出lemma。d04的schema失败来自other/reason组合，不是编号遗漏或非法类别；其引用body和粗体Lemma原始输出局部匹配，也不补发部分分数。

新候选在完整覆盖合法样本中漏title 3处、多取0处；另两处非空参考title位于d04，被整条结构失败归零，所以官方非空title为0/5，不能表述为五处原始输出都没有标题。旧模型多取title 4处、漏取0处。新候选合法覆盖样本中的reason错误包括ambiguous→null、ambiguous→mixed_roles、mixed_roles→null及insufficient_context→null各1处；d04的out_of_scope→null另按结构失败处理。

五pair中仅Proposition活动/正文提及p3-t01的新候选双成员全对；Proof标题、Example引用、缺失上下文/完整证明、ambiguous/mixed_roles四pair均未全对。逐类支持、TP/FP/FN、全部字段诊断及曝光在各条件report.json与all_sample_fields.jsonl中保留。

## 与既有开发失败的关系

冻结[失败复核](m5-lora-revision-error-review.md)记录的Proof标题、引用Example、ambiguous/mixed_roles问题，在相应完整训练原例中也失败。因此不能把这些失败全部解释为“训练原例学会了，仅新上下文迁移失败”。部分缺失上下文原例也未拟合。

两条已有Lemma训练原例中，s11的角色正确但标题丢失；d04的原始Lemma角色和标题正确，却被同条other/reason违规连带归零。这个结果不能直接解释开发集中非法kind=lemma的成因。没有plain/numbered Lemma、编号英语角色、正文提及Proof、英语ATX Definition或粗体题目/解答的新回放输入，这些形态依旧未测。

原训练回放减少与原例退步同时出现，但新旧实验还同时改变了训练组成、实际监督token量及优化路径；没有控制变量，不能确认减少回放导致遗忘，也不能据此直接判定LoRA容量不足。下一项只定义针对训练拟合问题的有界单变量对照，不自动延长训练或追加生成。

## 验收与产物

正式回放目录logs/m5_train_replay_diagnostic_20261019_v1已封存，receipt.json绑定12个输出文件与原runtime摘要。独立只读复算677项通过，523份前项产物原字节不变；实际独立M5 CPU tokenizer另核对28份保存token的输入长度、原始解码、仅去终止EOS的response、停止原因和预算，未追加权重加载、CUDA初始化或生成。

本次推理的Torch allocated/reserved峰值分别为：旧1,503,803,392/1,549,795,328 bytes，新1,537,357,824/1,591,738,368 bytes；它们是本次推理指标，不改变先前训练reserved超过设备标称总量的未解结论。

审计证据在logs/m5_train_replay_execution_20261019_v1：scope.json、cases.jsonl（两条件共28份）、summary.json、acceptance.json、cpu_token_acceptance.json。只读验收不向封存回放目录追加文件；没有下载、安装或Git写操作。
