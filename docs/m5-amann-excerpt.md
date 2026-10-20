# M5 Amann–Escher 第一卷局部转录与标注

日期：2026-10-18。本轮按用户指定页取样，交付**5片段22块**的页图对照转录、候选标注及合并评测候选v2。没有运行基座、规则预测或LoRA。下一任务仍以[current-work.md](current-work.md)为准。

## 页码与来源

所用文件为用户本地的Herbert Amann、Joachim Escher《Analysis I》，Gary Brookfield英译，Birkhäuser Verlag，版权页2005年、ISBN 3-7643-7153-6；没有据此推定未注明的版次。书名/作者/出版信息核对PDF页2/3。

| 用户指定的PDF页 | 书内印刷页 | 本次采用内容 |
| --- | --- | --- |
| 31 | 18 | 单射/满射/双射小节标题及定义性正文 |
| 32 | 19 | Proposition 3.5及其证明分段、Inverse Functions、Proposition 3.6 |
| 35 | 22 | 等价关系小节、等价类/划分正文、Proposition 4.1及证明 |

此处页码先按PDF阅读器从1开始的页序解释，两套页号均保留。三页有文字层，但原始提取包含错位、控制字符、缺失的集合括号/并集等数学符号；不能直接用于训练。
页图、layout/raw提取、书目页提取及原PDF摘要在Git忽略的`logs/m5_amann_excerpt_20261018_v1`。第二/三卷仅做文件摘要与元数据盘点，尚未提取其正文。

## 转录和块边界

局部规范文本在`data/raw/m5_amann_excerpt_v1/analysis_i_selected.canonical.md`；可直接审阅的原文/目标对照在`data/processed/m5_eval_candidate_v2/review.md`。
规范文本是从原页视觉核对后生成的Markdown，不冒称与PDF字节或原始pdftotext输出一致。

- 真实小节标题映射为Markdown H3，原文粗体角色/术语保留；不翻译、摘要、纠正数学或补证明。
- 数学字符按原页转录为LaTeX，保留编号、变量、方向、下标和量词；把断行接回原段落，合并排版断词，连字展开，弯引号按记录映射为ASCII引号。
- 独立段落和显示公式保持独立，不把完整命题与证明合成一个块来制造mixed_roles。Proof后独立的(ii)/(iii)续段没有自己的活动角色标记，按v1.1仍为body。
- 第31页的交换图、3.4 Examples与函数图不纳入本批文字样本；不丢掉图后把残缺例子冒称完整材料，也不把Examples复数静默加入既有标记表。其余未选区域明确列在转换清单中，不宣称完整转录三页。

每块记录PDF页、印刷页、原段落锚点、原页PNG、规范文本字符/UTF-8字节偏移和摘要；规范文本的偏移只定位规范文本，不能冒称PDF原始字节范围。视觉核对包括组合逆映射的顺序、恒等映射下标、等价类集合括号、商集和并集下标；未改变原数学陈述。

## 为什么部分定义仍是body

本任务仍使用[m5-structure-v1.1](m5-task-spec-v1.1.md)及冻结提示，要求显式“定义/Definition”引入；不是根据数学内容识别一切定义。书中的部分定义用加粗术语或“is called”引入，没有该前缀，因此保留body，不添加Definition标题来获得definition标签。
原编号Proposition明确引入命题，归theorem；Proof且已有论证归proof；把后续命题证明留作exercise的原句归other/out_of_scope，不把证明请求当证明正文。标题仍只按规范从本块首行提取，不把编号命题的粗体前缀臆造为title。

| 本轮新增类别 | 块数 |
| --- | ---: |
| body | 13 |
| heading | 3 |
| theorem | 3 |
| proof | 2 |
| other | 1 |
| definition / example / uncertain | 各0 |

标签与转录均为助手候选、非独立人工金标。没有用规则预测生成目标，也没有因模型结果修改提示或标签。此批适合核对编号命题、实际证明、无标记定义及证明续段的边界，不能宣称已覆盖八类。

## 同书分组与合并结果

先把同作者关联三卷整体固定为一个valid来源家族，再写规范文本和样本；后续不能把其他卷、章节、复制或修订版混入train/test作为独立来源。对选中段落与已有11份快照做有限匹配/显式书籍引用检查，复查了旧线性映射/行等价笔记中的通用概念，未发现本批编号命题或具体证明段落的复制证据。
这是限定范围的来源控制，不保证全书或未知翻译/派生史独立；自动匹配不能消除语言/LaTeX转换造成的盲区。原图、家族预留先后顺序、自动线索和人工判断均保留。

合并评测候选另存`data/processed/m5_eval_candidate_v2`，旧v1与训练包均不覆盖：

| 集合 | 片段 | 块 | 家族数（限定范围） | 剩余缺口 |
| --- | ---: | ---: | ---: | --- |
| valid | 13 | 56 | 3 | example、uncertain |
| test | 8 | 35 | 1 | uncertain、第2个家族 |

v1的16条69块原样保留在v2，新增5条22块全部进入valid候选；test内容与v1完全相同。同一本书不能同时用来补valid和test家族数。跨train/valid/test精确、去空白及既定32-gram近重复候选检查无命中，不将此等同完整语义去重。

[v2候选协议](../configs/m5_eval_candidate_v2.json)绑定旧协议、规范/schema/提示、规范文本、来源预留、转换/定位、候选及复核清单，继承全部门槛。原文定位、模板/长度与摘要等检查保存在`logs/m5_amann_excerpt_20261018_v1`；仅使用现有本地tokenizer，没有权重加载、推理、训练、安装或下载，未改源码或重跑pytest。

当前training_ready仍为false，不将缺类材料强标uncertain、不降低八类或来源隔离门槛。唯一下一任务：继续在明确的少页范围补valid的example/uncertain与独立test来源；若原始材料确实不支持这些类别，保留缺口，先解决材料与任务边界，不直接开始LoRA或全库OCR。Note2Lecture继续暂停。
