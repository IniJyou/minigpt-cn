# M5 微调任务与样本规范 v1.2

日期：2026-10-18。用户已明确同意增加 `Examples → example` 并升级版本。
下一任务以 [current-work.md](current-work.md) 为准；本页只规定标注契约。

本版继承 [v1.1](m5-task-spec-v1.1.md) 及其沿用的 [v1](m5-task-spec.md) 条款，仅增加英文完整词 `Examples`，忽略大小写，映射为 `example`。`Example`、`Examples`、`Ex` 都必须真正引入例子；正文中引用其他例子的编号不算新的引入，`Explain` 等子串不匹配。未增加其他复数或缩写。

记录版本为 `m5-structure-v1.2`；[schema](../configs/m5_sample_v1_2.schema.json) 只修改版本常量和文档指向，八类、输出字段、reason、标题提取及长度限制不变。[完整提示](../configs/m5_structure_prompt_v1_2.txt) 继承 SFT 候选 v1 的无具体 JSON 格式例提示，只修改版本与 Examples 枚举。[规范示例](examples/m5_spec_examples_v1_2.json) 保留原16条内容并迁移版本，另有一条原创 Examples/Explain 对照；新增公开示例不加入本轮训练或评测。

列表仍按最外层列表项切块并保留全部嵌套内容；单项中独立引入多个角色使用 `uncertain/mixed_roles`。单一原生复合标记同时给出互斥类别、无法唯一选择时，沿用 `uncertain/ambiguous`。不得合并独立段落制造混合，也不得把未写清的规则标成 uncertain。PDF 转录须记录原段落/公式边界与转换，不能把提取器丢换行当作原生混合。

旧规范、提示、数据和模型结果保留原字节。新版候选另存：历史记录仅迁移 schema_version，正文、边界、编号、来源家族、split 与目标不变；新增材料单独记录来源和审核依据。旧评测候选从未用于模型预测，但已被助手检查和标注，不宣称独立人工金标、未知预训练数据或统计充分的盲测。

提示已经改变，旧基座成绩只能作为历史记录；未来比较必须让基座和 LoRA 共用新版提示与冻结数据。测试集在配置/检查点锁定后一次使用，不能用于改提示、选参数或重标答案。数据覆盖通过不表示模型质量通过或训练就绪。
