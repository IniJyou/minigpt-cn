# M5 成对训练修订候选数据验收

2026-10-19。按[已冻结修订定义](m5-lora-revision-plan.md)完成48条新增训练示范和24条开发检查候选；旧36条train原样保留，另存84条合并训练。**本轮只制作和验收数据，没有加载权重、训练或模型生成。** Note2Lecture继续暂停。

## 实际材料

| 数据 | 样本 | 块 | 内容家族 | 非空标题 |
| --- | ---: | ---: | ---: | ---: |
| 新增训练 | 48 | 88 | 24 | 13 |
| 原训练+新增训练 | 84 | 183 | 26 | 50 |
| 新开发检查 | 24 | 50 | 12 | 6 |

原36条只有两个已冻结来源家族，所以合并后是26个家族。新增的36个家族严格使用定义时分配的24个训练槽位和12个检查槽位，成对a/b均留在同一集合。六类问题各新增4对训练/2对检查：标题边界、英文角色形式、活动标记与引用、证明请求与正文、uncertain原因、多块完整覆盖。

| 类别 | 新训练块 | 新检查块 |
| --- | ---: | ---: |
| body | 17 | 13 |
| heading | 9 | 4 |
| definition | 16 | 7 |
| theorem | 11 | 5 |
| proof | 12 | 8 |
| example | 10 | 6 |
| other | 8 | 4 |
| uncertain | 5 | 3 |

新训练的uncertain原因为ambiguous2/mixed_roles2/insufficient_context1；检查各1。两侧都有非空/null标题、八类、三种不确定原因及5–8块输入。训练P6为4→5、5→6、6→7、7→8块，检查为5→6和7→8块；只追加最后一块，已有块的正文、编号和目标不变。没有把代码、公式或同一顶层列表项拆开凑数量。

新增内容均为助手直接编写的原创合成材料，没有调用Qwen编写，也没有扫描本地电子书/笔记库或搬用旧valid/test。检查侧使用词操作、概率事件、路径/桥边、队列记录、函数复合、余数、条带覆盖、平面反射、闭球和逻辑真值等具体载荷；不把训练文字仅换变量或数字当作新家族。标签由助手依据原v1.2契约核定，非独立人工金标。数学正确性不是该任务的评测目标，仍不改写正文。

## 文件与复核

数据目录为data/processed/m5_lora_revision_candidate_v1，Git忽略。新[数据协议](../configs/m5_lora_revision_v1/dataset.json)绑定全部文件摘要；原protocol.json、规范、提示、运行器和历史结果保持原字节。

- [新增训练候选](../data/processed/m5_lora_revision_candidate_v1/new.train.candidate.jsonl)：48条，split=train。
- [合并训练候选](../data/processed/m5_lora_revision_candidate_v1/augmented.train.candidate.jsonl)：84条，原文件字节直接作为前缀，后接48条新记录。
- [新开发检查候选](../data/processed/m5_lora_revision_candidate_v1/boundary.valid.candidate.jsonl)：24条，按现有schema使用split=valid，外部role为new_boundary_diagnostic，与旧valid分别报告。
- [逐对审阅](../data/processed/m5_lora_revision_candidate_v1/review.md)、pairs.jsonl、block_reviews.jsonl：72条的138个块均有候选标签依据、原文摘要、实际字段变化和差异；没有用规则预测批量代替语义标注。
- source_families.json、family_review.json、coverage.json、manifest.json：家族、内容复核、覆盖统计与封存清单。

12条说明附件以及旧公开规范示例均不进入新增训练或新检查；旧valid保留为开发回归。检查侧正文/目标不用于训练、提示、样本权重或检查点选择。设计者已查看标签，所以新检查是“训练未使用的开发检查”，不称盲测。

## 重复与保真检查

新138个块中，P6两成员按设计共用前缀，形成34次正文重复；按家族内相同正文缩并后有104个新正文。对照原train/历版已查看valid、54块校准、公开规范示例、诊断示例与修订说明附件，共175个唯一参考正文，文件范围及摘要完整记录。没有解析旧test的正文或标签，只复核已有摘要。

对新正文之间及新正文对参考正文共23,556组比较，检查原字符、去空白相等与固定32-gram交集占较短集合≥0.8。达到阈值的16组全部属于预定同一pair的变体，跨家族或对参考正文的候选为0。短于32字符涉及的11,510组比较另有逐对索引及助手结合完整正文清单的内容复核记录；同组派生允许，跨组未发现相同载荷或仅换变量数字的派生。

这些是有限文件范围的候选筛查及助手内容判断，不能证明所有语义重复均已排除，也不是独立人工审查。阈值没有按结果调整；共享角色前缀和JSON骨架不被当作独立来源证据。逐块字符串、公式反斜线/换行、顺序编号、target schema和标题提取均核对；旧训练字节前缀与原文件完全一致。

## 真实tokenizer与训练顺序

只使用现有M5环境的本地tokenizer，在CPU上检查84条合并训练与24条检查，共108条完整序列。模板、enable_thinking=False、紧凑JSON、目标/EOS单次监督及prefix labels=-100沿用冻结接口；没有截断、packing或CUDA初始化。

| 项目 | 合并训练84条 | 新检查24条 |
| --- | ---: | ---: |
| 输入tokens范围 | 1259–1908 | 1266–1504 |
| target含EOS范围 | 26–180 | 26–181 |
| 完整序列范围 | 1289–1961 | 1292–1685 |
| 每条各一次监督tokens | 4602 | 1207 |

原train每轮2446个监督tokens，新增48条贡献2156，合计4602。按照原36条后接48条、每轮seed42+epoch_index打乱，完整执行两轮84条后取第三轮12条，得到45组×4条=180条次。每组实际目标含EOS计数已封存到training_schedule.json，总监督曝光**9657**；旧step45为5×2446=**12230**。这是相同步数的数据包对照，不能称等token预算。

最长训练序列仍为1961 tokens。该数值只是CPU预算检查，不能替代未来GPU预检，也不解决原训练reserved统计高于设备总量的未定成因。

## 验收与下一任务

只读验收在logs/m5_lora_revision_data_20261019_v1/acceptance.json：**1773项核对通过，340份历史产物摘要未变**。实际文件树、固定槽位/配额、schema/标题/原文字节、逐块复核、pair变化、重复复核、token布局和45步顺序/曝光一致。数据协议封存时的pending字段保留为制作阶段历史状态，是否通过以该独立验收回执为准；training_ready/execution_authorized仍为false。

本轮Ruff、文档链接、git diff --check及数据Git忽略检查另存同目录final_qa.json；没有修改产品运行源码或重跑pytest，没有安装、下载或Git写操作。数据和验收证据全部留存。

下一任务是实现并验收修订对照入口：正确处理45步/第三轮中途停止，复用原学习率前缀与修复后的greedy边界，默认只读inspect，通过有意义的CPU边界检查并绑定新数据回执。后续训练/GPU预检及拟定87次生成不在本数据制作轮执行；test继续封存。唯一下一任务以[current-work.md](current-work.md)为准。
