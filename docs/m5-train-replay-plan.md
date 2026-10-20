# M5固定训练样本回放诊断定义

2026-10-19。依据[失败与训练覆盖复核](m5-lora-revision-error-review.md)，冻结14条既有train记录、旧/新两个step45及**未来最多28次生成**，检查模型在已见示范上能否遵守契约。本轮只定义：未实现运行入口、未加载权重、未构造tokenizer，0训练/0生成，test仍封存。

## 要回答的问题

开发集失败既有具体形态缺口，也有已有示范未稳定迁移。先检查固定训练原例，区分“选中的已见示范也失败”与“训练原例正确而类似开发片段仍失败”，为后续修订提供依据；本诊断不直接判断欠拟合、遗忘、模型容量或学习率的单一因果。

协议位于[configs/m5_train_replay_v1/protocol.json](../configs/m5_train_replay_v1/protocol.json)。它是定义文件，现有运行器不能接受；本轮没有可执行的新增推理命令。下一任务以[current-work](current-work.md)为准。

## 固定样本与分组

全部从data/processed/m5_lora_revision_candidate_v1/augmented.train.candidate.jsonl引用，保留完整input/target和原块顺序，不能抽取关注块拼成新样本。输入消息仅含原系统提示和input，target用于离线评分，不进入生成提示。

| 顺序 | sample_id | 分组 | 关注点 | 旧/新微调中该完整记录呈现次数 |
| --- | --- | --- | --- | ---: |
| 1 | m5-sft-v1-spec-s11 | common_train | ATX Lemma及普通Proof | 5 / 2 |
| 2 | m5-sft-v1-diagnostic-d04 | common_train | 粗体Lemma、引用及练习 | 5 / 2 |
| 3 | m5-sft-v1-spec-s09 | common_train | 原ambiguous及其标题 | 5 / 2 |
| 4 | m5-sft-v1-spec-s14 | common_train | 原列表混合角色 | 5 / 2 |
| 5 | m5-rev1-p1-t01-a | revision_only_train | 粗体Proof紧接冒号，应有标题 | 0 / 2 |
| 6 | m5-rev1-p1-t01-b | revision_only_train | 粗体Proof后空格接正文，title=null | 0 / 2 |
| 7 | m5-rev1-p3-t03-a | revision_only_train | 活动Example引入 | 0 / 2 |
| 8 | m5-rev1-p3-t03-b | revision_only_train | 引用Example不生效 | 0 / 3 |
| 9 | m5-rev1-p3-t01-a | revision_only_train | Proposition实际引入 | 0 / 2 |
| 10 | m5-rev1-p3-t01-b | revision_only_train | 句中提及Proposition | 0 / 2 |
| 11 | m5-rev1-p5-t04-a | revision_only_train | Definition/Theorem→ambiguous | 0 / 2 |
| 12 | m5-rev1-p5-t04-b | revision_only_train | 独立Definition+Theorem→mixed_roles | 0 / 2 |
| 13 | m5-rev1-p5-t03-a | revision_only_train | 缺指代上下文→insufficient_context | 0 / 2 |
| 14 | m5-rev1-p5-t03-b | revision_only_train | 明确证明过程→proof | 0 / 2 |

common_train为4条7块，旧/新两个微调都确实见过；revision_only_train为10条10块，只有新修订微调见过。旧模型在第二组的表现只作参照，不能当作遗忘证据。“见过”仅指本次微调的完整记录曝光，不判断预训练中是否出现相似内容。

合计14条17块、6个分组家族。后10条组成原有5个完整pair；前4条原始记录不拼成pair。全体均是已有合成train示范，选择来自训练覆盖与失败类别复核，没有训练样本预测可供挑选。不能把14条当作14个独立来源，也不代表完整84条train。

六类支持为body3、theorem3、proof4、example1、other1、uncertain5；definition与heading均0。非空title为5，非空reason为6，三个uncertain reason均有覆盖。仍可沿用固定八类评分，但含零支持类别的macro-F1只作诊断，不套用原开发集九项接纳门槛。

普通/编号Lemma且title=null、编号前置的英语Proposition/Examples、body中提及Proof等缺口不在现有train内。本轮不新造样本、不复制开发片段填补这些位置；因此本诊断不能直接证明它们已学会。

## 固定模型、输入及预算

两个条件均用冻结Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca及其tokenizer/模板，沿用m5-structure-v1.2和configs/m5_structure_prompt_v1_2.txt。BF16、cuda:0、SDPA、batch1、enable_thinking=False；加载adapter后只读eval，所有参数不可训练。

| 条件 | 检查点 | 新增调用编号 |
| --- | --- | --- |
| control_step45 | logs/m5_lora_experiment_20261018_v1/train-valid/checkpoints/step_000045 | 1–14 |
| candidate_step45 | logs/m5_lora_revision_experiment_20261019_v1/train-compare/checkpoints/step_000045 | 15–28 |

两条件都按表中同一顺序处理14条，每条最多一次。旧adapter文件SHA为3f8762e9389935bca779f52a0b02fab624ab18782c30a183bf49b5bd34d2d841，新adapter为ca571b7fb6978ab469efacf3c24ce2279b19e4b069f7fc800d7f99c0cf4573d5；协议绑定完整检查点文件树，不续训或挑选其他step。

greedy_search，显式do_sample=False/use_model_defaults=False；种子42，num_beams=1，repetition_penalty=1.0，最多新增1024 tokens，EOS/pad及dynamic cache沿用原冻结配置。不能为了加快诊断另改输出上限、量化、设备或解码。

输入预算4096，现有验收布局为输入1267–1321、目标含EOS 26–72、完整训练序列1294–1393 tokens；14条目标文本合计452 tokens。这里只复用此前真实tokenizer验收记录，本轮没有重新tokenize。未来入口须用实际tokenizer再次核对输入，禁止截断、packing或把target加入输入。

未来总预算是28次尝试，上限28672个新增tokens；其中4条common_train每模型各4次、10条revision_only_train每模型各10次。没有额外基座条件、warmup生成、示例生成、valid重跑、test、训练或前后向预检。已有39条开发结果仅作冻结参照。

## 评分与解释规则

- 保留所有14条/每模型17块及全部28次原始输出。JSON/schema/编号顺序、整片段/块全字段、kind/title/reason沿用冻结严格评分；结构失败整条计错，不修补、不跳过错误样本。
- 总表之外，强制分别报告common_train 4条7块和revision_only_train 10条10块；逐条列新旧状态与字段错误，单列非法kind、标题遗漏/多取、reason混淆。
- 5个原有训练pair分别报告两成员是否全部字段正确；关注块可用于解释，不能替代完整样本或完整分母。
- 新候选在选中训练记录上失败，说明存在“已见原例仍失败”，但不能直接确定训练不足、实现错误或容量不足。全部14条正确也只表示所选训练示范拟合，不能宣称完整train或开发/泛化质量通过。
- common_train中旧正确/新错误可记为该原例退步；新旧都错误则为已有问题。两者都不能单独证明回放比例是原因。revision_only_train不据旧模型失败判断遗忘。
- 仅当对应训练原例正确且冻结开发相似片段失败时，可描述所选原例与该开发片段的差异；缺失形态仍单列为未测。所有解释允许并存，不触发补跑、额外训练、改标签或自动选择检查点。

本诊断没有新增发布/接纳阈值，不修改旧selection；diagnostic_only=true、automatic_test_allowed=false始终保留。

## 未来入口边界

拟定输出目录固定为logs/m5_train_replay_diagnostic_20261019_v1，目前不存在。未来入口须在模型加载前核对协议/数据/检查点/环境/安装版生成API摘要，并拒绝覆盖、恢复或重试已有目录。

每次实际generate前核对有效配置与模型冻结状态，先将attempt、condition、sample_id、有效模式及显式覆盖参数flush+fsync落盘。第29次、错序、摘要/配置变化或账本无法落盘时全局停止。普通逐条生成异常保留空回复并计错；OOM、内存、配置或完整性错误全局停止，失败现场保留且无成功回执。完整28次结束后再封存预测、runtime、完整/分组指标、pair结果和诊断解释，不把未完成阶段报成完整结果。

## 本轮证据

logs/m5_train_replay_definition_20261019_v1保存selected_samples.json、planned_call_order.json与selection_summary.json：逐条绑定源train行号、UTF-8字节范围、原行及input/target摘要、真实微调频次和既有token布局。没有改写或导出新的训练内容。

协议及独立验收绑定前项476份冻结产物。现阶段只有定义与数据/摘要核对，尚无回放结果，不能据此说模型已学会示范。Note2Lecture继续暂停。
