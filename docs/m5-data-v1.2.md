# M5 v1.2 数据候选补齐

日期：2026-10-18。本轮交付一个版本一致的数据候选包，执行 [任务规范 v1.2](m5-task-spec-v1.2.md)，协议为 [评测候选 v3](../configs/m5_eval_candidate_v3.json)。只新增 `Examples → example`；八类、标题、reason、模型输出及评测门槛沿用旧版。

在 [SFT 候选 v1](m5-sft-candidate.md) 与 [评测候选 v2](m5-amann-excerpt.md) 上另存新版，历史57条记录只迁移 schema_version，原文、目标、家族和划分不变。新增3条7块，最终 train 36条95块、valid 15条62块、test 9条36块。valid 有4个来源家族，test 有2个；两侧八类均有支持，原定最低覆盖门槛通过。所有标签仍是助手候选。

新增材料分别为：

- 用户指定的 Amann–Escher 第二卷 PDF 第45页/印刷44页，选取 5.7 Examples 的 (a) 陈述、完整分段公式、独立 Proof 段，共3块。陈述为 example，公式为 body，已给出的引用论证为 proof。三卷仍共同属于 valid 的同一家族。
- [Karen E. Smith 的商群讲义](https://sites.lsa.umich.edu/kesmith/wp-content/uploads/sites/1309/2024/07/FirstIsomGroups.pdf) 第1页，相邻的预备正文、原生 Definition/Theorem 段、Remember 正文，共3块。复合标记无法唯一归入 definition 或 theorem，沿用 uncertain/ambiguous；并未合并两个独立段落。作者在页上注明 CC BY-NC-SA 4.0；本地派生转录保留署名、链接、转换说明与该许可，未纳入仓库 MIT 许可。
- [Chandan Singh 的实分析笔记](https://csinva.io/notes/math/analysis.html) 中一个完整顶层列表项，保留其中 pf 与 corollary 的嵌套结构，属于 uncertain/mixed_roles。原始 Markdown 与仓库 MIT 许可已核对，另记录其自述参考 Abbott 的 Understanding Analysis；该作者笔记及该书的已知派生作为同一 test 家族预留。

新增两份网上材料先确定家族与 split，再制作片段。来源检查限于已知11份本地快照、此前选取的 Amann 段落与本轮材料；核对署名/来源线索、原文重叠及跨集合固定32-gram候选。未声称完成全库谱系追踪、跨语言语义去重，或确认材料不在 Qwen 预训练中。

本地候选在 `data/processed/m5_eval_candidate_v3`：train/valid/test、训练 messages、覆盖数、家族预留、转换/定位清单和 `review.md`。个人原文、网页快照、PDF 只读缓存、页图和审核证据均在 Git 忽略目录；不公开书籍原文。日志在 `logs/m5_eval_completion_20261018_v1`，包含来源/分块检查、摘要、实际 tokenizer 长度与验收回执。

训练接口继承原 SFT 契约：模型只接收 system + 紧凑 input JSON，监督紧凑 target JSON 和一个终止 EOS；前缀/右侧 padding 的 labels 为 -100，不拼接、不截断。新版全部60条只做本地 tokenizer 预检，不加载模型权重。冻结规则代码保留为历史比较器，未增加 Examples 识别补丁；未来该比较器对新标记的限制应如实报告。

这是最低覆盖检查，valid 的 example 与 uncertain、test 的 uncertain 各仅1块，test 新家族仅1条，不能据此声称类别覆盖充分或泛化可靠；真实 insufficient_context 仍缺。尚未冻结 LoRA 参数、准备训练依赖或启动训练，`data_coverage_ready=true`、`training_ready=false`。下一项是制定最小 LoRA 实验与验证选择规则，并只读确认独立 M5 环境的依赖缺口。具体安装仍交由用户执行。
