# M5 valid错误复核与下一版最小改动

2026-10-19。本轮仅读取冻结train/valid及既有回复，重算原分数、逐块核对错误，并定义一个零训练的贪心解码对照。原[训练结论](m5-train-valid-results.md)仍为三个检查点均未接纳；test仅做既有文件摘要核对，不解析其标注，不进入生成或选择。Note2Lecture继续暂停。

## 错误分解

先区分整条结构失败和结构有效后的字段错误。以下分解只帮助定位问题，不给失败整条补发局部分数，主指标与原报告完全一致。

| 检查点 | schema/覆盖失败影响的块 | 结构有效块 | 其中全部字段正确 | 其中字段有误 | 结构有效块中的title错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step45 | 5 | 57 | 39 | 18 | 15：5漏填、10多填 |
| step90 | 5 | 57 | 43 | 14 | 10：5漏填、5多填 |
| step180 | 17 | 45 | 35 | 10 | 8：4漏填、4多填 |

每行前两列合计62块；结构有效块内“完全正确+字段有误”等于结构有效块数。title错误与kind/reason错误可能发生在同一块，不能相加当作总错误块数。step180的结构有效子集更小，其字段错误数较少不能解释为整体改善。

完整覆盖方面，三个检查点都在`m5-eval-v1-v08`输出b0001–b0004而漏掉b0005。这是5块输入；训练集中已有两条5块样本，也有6块与7块样本，所以不能断言模型“没训练过5块”。该错误没有触及1024-token输出上限。

title方面，step45在结构有效块中的15处错误包含7处“只有title错”，其余8处伴随kind或reason错误。典型漏填有`**题目**:…`、`**解答**：`和ATX语义标题；典型多填有`**Proof** 后接正文`、`**编号 Proposition** 后接正文`，以及普通正文的首句。既定规范要求非ATX粗体标题之后紧接冒号或行尾，因此“粗体标记后接空格和正文”的title应为null。train95块与valid62块的候选title均和冻结提取公式一致；这只验证标题公式，不是独立人工复审所有角色或数学内容。

reason方面，step45结构有效块有7处错误，其中6处伴随kind错误，只有1处是kind正确但reason错：Smith的原生复合标记已判uncertain，却把ambiguous写成mixed_roles。step90/180结构有效块中reason错误都伴随kind错误。step180另有3条整条schema失败，包括other复制正文作为reason、uncertain使用非法值`"uncertain"`、uncertain使用null；这些失败没有纳入上表的“结构有效块”字段诊断。

角色方面，三个检查点均把Amann第二卷的example块判作heading；step45还把引用块里的例题判为example，把独立证明续段判为uncertain，把正文对命题的引用判为theorem。以上是既有边界规则下的可见错误，不能靠改评测标签或给输出补规则来计分。

## 已有训练覆盖与未确定成因

训练集36条95块，body33、heading10、definition9、theorem9、proof8、example9、other10、uncertain7；非空title37块。uncertain的reason为mixed_roles5、ambiguous1、insufficient_context1。类别已出现，仍不等于边界形式覆盖充分。

按忽略大小写的完整词字面出现统计，train中Examples、Proposition、Theorem均为0块；Proof为2块，Example为1块，Definition为5块。此统计包含代码和引用，不等价于语义角色计数。已有英文Proof训练例分别是无粗体的冒号形式和粗体后紧接冒号的形式，未覆盖本轮Amann的粗体后接空格正文形式。新版规范已加入Examples，但新增公开示例被预先排除于训练集，旧train目标未变。

这些事实支持“部分表面形式覆盖有限”，不能证明它是唯一原因。原生成使用temperature0.7的固定种子采样；现有结果没有同权重的贪心对照，不能区分采样路径、训练覆盖、模型学习等因素各自的贡献。训练loss下降而valid变差同样不足以单独确定过拟合机制。显存reserved统计差异是另一个未确定问题，不在本轮顺便调整运行配置。

## 唯一后续实验：贪心解码对照

机器定义为[greedy诊断协议v1](../configs/m5_greedy_diagnostic_v1/protocol.json)。唯一有效干预是将随机采样改为greedy；不训练，不改模型/adapter、v1.2提示、valid、目标、评分、预算或输入格式。

| 项目 | 冻结定义 |
| --- | --- |
| 模型条件及顺序 | 未适配基座15条 → 固定step45 adapter15条 |
| adapter依据 | 旧valid宏F1最高的step45，用于事后诊断；旧`selected_step=null`不变，不称其已接纳 |
| 新模型调用上限 | 30次，逐条完整valid；不试step90/180，不搜索种子或其他解码参数 |
| 解码 | `do_sample=False`、`num_beams=1`；temperature/top_p/top_k/min_p清为null，避免残留无效采样字段 |
| 保持项 | BF16/cuda:0/SDPA、非思考、逐条种子42、输入4096/新增1024 tokens、同模板/EOS及严格原样评分 |
| 对照 | 新greedy基座对新greedy adapter计算全部原九项门槛；另与旧同权重采样回复按样本编号配对描述变化 |
| 输出 | 两组原始回复/token/运行异常、metrics、errors、实际配置/环境/峰值显存、配对变化及完整回执 |
| test与规则 | test不解析或运行；规则只引用原冻结结果，不新生成 |
| 停止 | OOM、摘要/环境/输入预算不符、目标目录已存在即停；其他逐条异常计空回复，保留分母；不修复或重试 |

当前实际Transformers4.57.6的GenerationConfig已在独立M5环境做CPU检查：原配置为sample，新配置为greedy_search；43项包未变，没有构造模型或初始化CUDA。改变的序列化字段共5个，其中4个是停用采样字段；不是将temperature设为0继续采样。greedy不代表跨平台逐位复现承诺。

这是一项使用已检查valid的事后开发诊断。若漏块/schema/title等改善，只能说明新解码策略在本次配对观察中产生了影响，不能据单次运行确定全部原因；若未改善，不自动换检查点或追加训练。即使全部原门槛通过，也先报告新的valid观察，不自动启用test、部署或改写旧实验的失败结论。不能把旧采样基座分数用于新greedy adapter的主增益判断。

本轮只定义；后续须另存诊断运行器、通过有意义的CPU边界检查并绑定源码，再独占创建`logs/m5_greedy_diagnostic_20261019_v1`执行一次。复用旧检查点进行诊断时允许旧selection未接纳，但新入口必须禁止test路径和任何训练调用，不能放宽旧运行器的test门槛。

## 证据与当前状态

私有逐块原文/目标/实际输出、结构失败详情和训练对照片段在`logs/m5_valid_error_review_20261019_v1/cases.private.json`；分解统计在同目录review.json，API检查在generation_api_check.json。它们用于复核，不回写数据或原始回复。原冻结产物、旧分数、schema、规则、提示和三个检查点均保留。

本轮新增模型生成0、优化步0、模型加载0；只做只读统计、局部人工复核、GenerationConfig检查和产物验收。下一任务以[当前工作](current-work.md)为准。
