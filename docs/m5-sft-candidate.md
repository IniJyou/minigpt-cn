# M5 最小SFT候选与评测清单 v1

日期：2026-10-18。本轮交付训练候选数据包与评测来源清单，未训练模型。任务规则仍为[m5-structure-v1.1](m5-task-spec-v1.1.md)，下一任务只看[current-work.md](current-work.md)。

## 候选包含什么

候选共36条、95块：16条真实笔记校准记录54块，16条已公开规范示例29块，以及4条已查看的合成诊断记录12块。保留每个原片段的完整块序列，不拆成单块、不重采样、不把single诊断副本重复加入；全部仅作拟训练用途。
旧review、split=null、规范示例的spec_demo及诊断材料均原样保留。新副本采用现有完整样本schema，split=train表示本候选的预期用途，不表示训练已启动或数据已通过正式入库。标注均由助手起草/复核，用户接受规范不等于逐条独立人工金标。

| 类别 | 真实54块 | 合成41块 | 合计 |
| --- | ---: | ---: | ---: |
| body | 23 | 10 | 33 |
| heading | 8 | 2 | 10 |
| definition | 4 | 5 | 9 |
| theorem | 4 | 5 | 9 |
| proof | 3 | 5 | 8 |
| example | 4 | 5 | 9 |
| other | 6 | 4 | 10 |
| uncertain | 2 | 5 | 7 |

保留多块完整输出、null/reason、标题、证明请求与混合角色的正反例；新增训练副本不改变旧基线结果。ambiguous/insufficient_context只由已有合成规范示例提供，不据此宣称真实来源已覆盖这些边界。小规模且合成占41/95，适合验证SFT接口，不能预设泛化收益。

私人数据包在`data/processed/m5_sft_candidate_v1`：train.jsonl为36条完整记录；messages.train.jsonl为带sample_id的三消息审阅副本；provenance.json保留父记录、来源定位和新旧用途；source_families.json为来源及预留清单；coverage.json为类别/字段/块数统计。source/sample/split及定位不进入模型输入。

## 固定的提示和训练接口

[候选配置](../configs/m5_sft_candidate_v1/protocol.json)绑定[完整提示](../configs/m5_sft_candidate_v1/prompt.txt)。提示逐字继承诊断C的去例版本；只是后续SFT的固定候选，不替换旧正式提示或把诊断C称为通过。基座与LoRA后续比较必须共用本候选提示，不把旧54块基线直接当作同提示的未见对照。

- 基座仍为Qwen/Qwen3-0.6B，同revision、tokenizer、聊天模板与独立WSL环境；enable_thinking=False。未引入PEFT/TRL依赖或确定LoRA超参数。
- 用户消息严格使用`json.dumps(input, ensure_ascii=False, separators=(',', ':'))`；目标仅以相同方式序列化target。不加入理由文字、父记录、来源、split或模型失败回复。
- 构造system/user消息，再按绑定聊天模板取得`add_generation_prompt=True`的完整推理前缀P。模板已有的空think段属于P并屏蔽loss，不训练思考正文。
- 目标T为紧凑JSON字符串。整体编码`P + T + '<|im_end|>'`，add_special_tokens=False；必须核对整体的前缀token与单独编码P逐项一致，不能假定字符串边界就是token边界。
- labels在P和未来padding位置为-100，仅T与最后一个im_end token计loss；最后EOS=151645。不要用pad_token_id=151643直接批量屏蔽同值token来替代位置mask，不把BOS或额外EOS附加进目标。
- 原生完整assistant模板含末尾换行；须验证它等于`P + T + '<|im_end|>\n'`。训练仅省略终止EOS之后的该换行，不去除目标原字符或模板前缀。
- 无截断、无packing、无多样本拼接。输入最多4096 tokens，目标含EOS最多1024，总长最多5120；超限退回数据整理。未来batch的input_ids/attention_mask/labels形状均为[B,L]，padding为右侧；causal LM内部shift一次，不在数据侧再shift。

本轮只加载本地tokenizer检查36条长度、模板、目标往返与loss位置，结果保存在`logs/m5_sft_candidate_20261018_v1/token_preflight.json`。没有模型权重加载、前向、梯度或生成。真正训练collator仍须在训练准备轮对可变长batch和padding位置核验，不能仅凭单条预检认为训练器已验收。

## 来源家族与评测预留

已查看真实6文档放入同一保守排除桶，所有切片/复制/修订/派生材料只能留在拟train或隔离区。此桶表示共同排除，**不是声称6文档同属一个真实家族**；更广关系仍未核定。所有已公开规范例、12块诊断及其single副本同样排除出未见评测。

在用户授权笔记目录只列根目录文件，预留5份新文件的本地快照与摘要，形成4个暂定来源桶，拟valid两桶、拟test两桶。其中明显的两个版本先保守合并为一桶。私人路径、文件名、SHA和暂定归属仅在忽略的source_families.json中。未切块、标注或跑预测，不把这些预留桶计作已核定家族或正式valid/test。

实际valid.jsonl/test.jsonl尚未创建，已验收valid/test条数均为0。进入正式划分前逐项完成：

1. 在本次有限来源清单内核对原件/副本、命名版本、修订导出、内部链接、共享例题及人工可识别的改写派生关系；未知关系保持隔离。扩展范围先记录具体文件，不递归扫描全库。
2. 精确SHA、去空白全文相等及字符32-gram Jaccard>=0.60列为归并候选；另查相同片段、相邻/包含章节与共享例题。低于阈值不能证明独立，短于80字符不做32-gram判断。清单固定后，所有确认的关系按连通分量归并。
3. 同家族跨train/valid/test时不得保留两边。若预留材料与已查看训练桶同源，优先移入train或隔离，并在生成评测片段前补充来源、另冻新版清单；不能将训练材料移动去test来制造“未见”。
4. 先接受家族划分，再按v1.1人工核定块边界、取连续片段并标注；切片和增广继承父家族。valid可供开发；test规则和目标先冻结，后续禁止查看模型test表现来改提示、标签或checkpoint。助手参与编制目标须如实披露，不能称为独立人工盲评。
5. 目标为valid/test各至少2个确认家族、8片段、24块，并各包含八类至少1块、非空标题、other/uncertain及多块片段。不为凑门槛改写原文或强标uncertain；不足就补有限新来源，保留缺口。小规模不强凑80/10/10比例。

文件摘要与32-gram只能提供线索，本轮未完成新来源的语义家族审计，因此训练准备的“评测就绪”门槛当前未满足。这不影响审阅36条训练候选和已冻结接口。

## 预先固定的验收门槛

以下是最小实验的工程/质量接纳线，不是统计显著性保证；在查看新评测预测前固定，修改必须升版并解释。

| 阶段 | 要求 | 当前状态 |
| --- | --- | --- |
| 候选包完整性 | 36条/95块；原文与目标无改动；完整schema字段、编号、title/reason及来源定位检查通过；旧冻结产物不变 | 本轮验收 |
| tokenizer与loss接口 | 36条均满足4096/1024/5120；模板前缀一致；仅JSON+一个EOS监督；无截断 | 本轮验收 |
| 训练前数据就绪 | 来源家族核定，无跨集合复制/派生泄漏；valid/test满足上节最小规模；目标复核、文件SHA与清单冻结 | 未完成 |
| 训练器就绪 | 新轮固定LoRA参数、步数/资源上限、checkpoint选择和停止条件；验证batch mask/padding及基座冻结，所需安装由用户执行 | 未完成 |
| 新valid接纳 | JSON/schema/完整覆盖均100%；严格块全字段>=80%、整片段>=50%、固定八类macro-F1>=0.75，较同提示基座提高>=0.15；非空标题、非空reason各>=80% | 未评估 |
| 一次新test报告 | 只在valid通过并锁定单一checkpoint后运行基座/LoRA/固定规则；沿用相同门槛并逐项报告是否通过、全部分母与原始失败 | 未评估 |

严格评分仍复用冻结evaluate：坏schema/覆盖的整回复不得捞取部分正确TP。无重试、修JSON、补块、去围栏或忽略失败；固定八类macro-F1，额外报告逐类支持/召回、混淆、标题含null及非空子集、reason含null及非空子集。
未来评测沿用已冻结Qwen预算和采样/种子42，基座与LoRA相同；运行失败先停止并保留证据，不根据质量追加搜索。若规则基线更强，直接保留比较，不把超过基座自动解释为超过规则。

本轮不宣称数据/模型质量通过。唯一下一任务是按私人预留清单完成有限来源家族核对，再制作并冻结valid/test片段与候选标签；随后才准备LoRA训练运行器。
