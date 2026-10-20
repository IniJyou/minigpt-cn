# M5 v1.1：完整提示、规则参照与校准协议

日期：2026-10-16。任务入口为[current-work.md](current-work.md)。本轮使用已有16片段54块候选标注，不新增读取笔记、下载模型或训练。

## 固定输入和提示

[完整系统提示](../configs/m5_structure_prompt_v1_1.txt)包含[v1规范](m5-task-spec.md)的沿用条款和[v1.1](m5-task-spec-v1.1.md)的覆盖规则，不只给类别名称或依赖模型自行读取文档。用户消息仅为input的JSON序列化，助手目标仅为annotations对象；来源、review_id、split和人工标签不进入推理输入。
提示明确标记别名、列表项边界、真实混合角色、证明请求、标题原文及reason关系；不要求重写正文或输出思考过程。当前没有绑定Qwen revision/tokenizer/聊天模板，也没有模型输出；这些在后续环境与基座任务中确定。同一版本的基座与LoRA将共用完整提示，不能在测试时补答案或改规则。

[运行协议](../configs/m5_rule_baseline_v1.json)绑定两版规范、完整提示、输出schema、规则/评分代码、运行代码、测试文件、54块JSONL及来源manifest的SHA256。先完成合成边界测试并生成协议，再首次运行本批规则，不按校准得分修改规则、提示或候选标签。运行前后都检查摘要，已有输出目录拒绝覆盖；版本改变须新协议和新输出。

## 简单规则的具体范围

[规则与评分实现](../src/minigpt_cn/m5_structure.py)使用有限前缀、正则表达式和少量论证连接词，不调用模型。它只接收input；[运行器](../src/minigpt_cn/m5_rule_run.py)先生成全部预测，再独立读取参考标签评分。

- 代码、引用、纯公式保护；从块内活动行识别固定语义标记和Thm/Pf/Ex别名，忽略英文大小写，要求标记边界。
- 题目/习题/解答等范围外前缀直接other；“请证明/求证”等或条件叙述中的证明要求用有限词面规则判断。
- Proof/证明/Pf后的内容有有限论证开头或连接词时作为proof；否则近似为题目请求。此项是简单参照的局限，不等同于规范要求的语义理解。
- 同块多个角色为mixed_roles；有少量互斥标记与缺指代模式。规则只看本块，空证明标记不利用邻块补上下文；它不保证完整实现提示中允许的片段上下文判断。
- 标题按规范的ATX或行首粗体前缀原样提取；不去列表前缀取标题，不继承前块类别。

这些词面特征来自已讨论规范；54块本来就是校准集，不能称作对本轮开发隐藏的测试。系统提示绑定供后续Qwen使用，规则代码没有通过该提示生成，也不评价提示的模型效果。

## 评分口径（首次运行前确定）

所有分母包括失败回复，不自动补JSON、去围栏、删额外字段、补块、重排或选成功样本。

1. JSON解析：拒绝重复键、NaN、Infinity、解释/围栏及截断JSON。输出schema逐项检查精确字段、类型、枚举和kind/reason关系；当前实现对应schema中实际使用的约束，不是通用JSON Schema库。
2. 覆盖：只有schema有效且编号逐项同序、恰好覆盖输入，整条才可逐块获得正确分。多块的任何格式/编号错误都会使该条所有金标计FN，合法可辨认的预测类别项计FP；未知类别项另计invalid_kind_items。不让正确前半段抵消坏后半段，也不跳过该条。
3. 类别：有效回复按输入顺序比较；错类给金类别FN、预测类FP。逐类precision=TP/(TP+FP)、recall=TP/(TP+FN)、F1=2TP/(2TP+FP+FN)。无分母用0；macro-F1固定平均八类，micro-F1先合计八类TP/FP/FN。未知类不会加入第九类稀释macro，仍使金标漏检并另报非法项。
4. 分类准确率、全部字段块匹配率和整片段完全匹配率同时报告；整片段比较解码后JSON对象，忽略键顺序/非语义空白，不忽略字符串内部空格。
5. 标题精确匹配包含null，并单列金标题非空子集；reason同样报告全体及非空金reason子集。格式坏整条的所有标题/reason均计错，分母不缩小。
6. 混淆矩阵有missing_or_invalid列，坏回复可辨认类别的FP在逐类表体现；完整预测、错误字段、参考值、原文和原始回复均保留。格式有效子集的额外分数不是本轮主结果。

输入人工参考若不合接口，运行停止，不把坏参考当模型错误。本轮未使用正式test、选checkpoint或比较训练loss；原候选不因规则得分而修改。

## 运行和复现

使用现有Windows项目环境，不安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_m5_structure.py --tb=short
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m minigpt_cn.m5_rule_run --protocol configs/m5_rule_baseline_v1.json --output-dir logs/m5_rule_baseline_replay_v1
```

最后一条必须指定不存在的输出目录。只有代码的新克隆不含私有54块及来源manifest，须使用本机准确版本，不能自动生成替代数据后声称复现。输出含predictions.jsonl、metrics.json、errors.jsonl、report.md和run_manifest.json，均保存在Git忽略目录。
CPU测试只使用合成文本。Windows测试临时目录限定在项目logs下由测试新建/清理，避免依赖沙箱系统临时目录；不修改用户数据或环境。

## 判读与停止点

本轮先交付固定规则在54块上的实际结果，正文见本地logs/m5_rule_baseline_20261016_v1/report.md。高分说明当前显式标记任务可能已适合简单规则；不代表未见效果，也不能预设LoRA一定改善。低分则按实际错误区分规则近似、规范缺口或候选标注问题，不通过加特例或改金标制造通过。
来源集中、类别支持少、uncertain只有mixed_roles、代码样例少等局限继续保留。后续先只读盘点独立M5环境/本地模型是否就绪，再确定基座固定运行；需要下载、安装和正式训练时按具体范围另行授权。没有自动进入Note2Lecture、扩语料或云作业。
