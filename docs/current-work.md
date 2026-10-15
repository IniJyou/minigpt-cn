# 当前工作与交接

更新：2026-10-15。**本页是唯一的下一任务入口；参考文档与实验日志不是待办清单。**

## 当前阶段与唯一下一任务

用户已决定结束本批33M优化，完成M3收尾后授权M4展示。**M3实验已结束，生成连贯性不足作为最终限制保留；当前阶段为M4项目展示。** 不以展示质量不足为由自动追加训练。
用户已确认按交付步骤完成推送、PR合并及M3收尾；本地实际提交为bbc6e36、分支codex/m3-closeout，本轮开始暂存区为空，仅保留原无关未跟踪文件。助手未执行Git写操作或远程核验，用户报告与独立远端验证区分记录。M3历史证据见 [阶段收尾](m3-closeout.md)。

用户最新授权“M4全权交给助手，用户做收尾验收”，本轮完成一个整体验收包：**[项目展示](showcase.md)、[模型卡](../MODEL_CARD.md)、[本地复现](local-reproduction.md)和[M4用户验收](m4-acceptance.md)**。固定原12208步final；展示页保留6个提示词×两种模式全部12条全文，模型卡按原顺序仅摘前2条各80字符，不挑最好结果。
已有Matplotlib离线生成PNG/SVG：611条原train记录、26次原完整valid、6次续训valid，原始点连线未平滑，区分云端训练过程和本地独立评估。图和展示页由scripts/build_showcase.py绑定7份原始记录SHA生成，默认拒绝已有输出；复建须新目录及已有完整绘图环境，不改模型环境或安装依赖。
前项文档CPU8-token命令在Windows PowerShell5.1实际执行，证据在logs/m4_model_card_v1与logs/m4_local_reproduction_v1；本轮只读回结果、核对当前命令块未变，不重复生成。首次Get-FileHash不可用的前项失败日志保留。CUDA长续写未执行，不把短命令通过当质量验收；本轮不读test、训练或重跑完整CPU/WSL回归，历史736/741不当新结果。
展示构建、图片视觉检查和全部12条原文/指标/链接/格式/源摘要读回在logs/m4_showcase_v1/acceptance.json；阶段状态为“M4技术交付完成，待用户整体验收/Git发布”，生成质量仍未通过。旧M4模型卡回执保留当时文档快照，由新终验回执绑定最终文件。
唯一下一任务：**用户按M4验收清单收尾**。用户通过后自行创建M4分支、暂存明确展示文件、提交/推送/PR/CI/合并并核对关闭里程碑；本轮没有Git写操作。若有反馈，只修复阻塞本批验收的问题；不自动进入LoRA或追加训练/语料/参数搜索，独立0.6B阶段须另行授权。

## M3最终结论

- 模型33104384参数，8层/8头、context512，RMSNorm/RoPE/SwiGLU、权重共享。
- 同版16K tokenizer：16000词表（含EOS256）、15743 merges；旧2048 tokenizer/权重不混用。
- 最终冻结train为73520篇/48187994个token位置；valid/test为461/450篇、297390/251517个位置。训练仅使用train。
- 原final从随机权重训练12208步，处理100007936 tokens，约2.075倍算术曝光；不是1亿唯一语料token或实际epochs。训练约49.52分钟，资源统计见 [原实验报告](m3-experiment-report.md)。
- 原final本地独立valid/test loss=3.962893003020671/3.9957795353415366，完整覆盖297389/251516目标。12条固定生成全部保留，明显重复、离题或句意混乱，展示质量未通过。
- 原始预测/生成、缓存与因果前缀对照未在固定样本发现对应链路错误；不能据此断言唯一质量原因。可选重复惩罚默认1.0；固定采样惩罚1.1缓解表面重复，但连贯性不足，见 [生成诊断](generation-repetition-guide.md)。
- 有界续训仅将12208步延至14208，追加2000步/16384000 tokens，保留原调度，追加LR均3e-5；恢复原优化器、采样器与RNG。原兼容云端训练，本地不同Torch版本不绕过resume保护。
- 两模型同一PC重新完整valid：3.962893003020671→3.9458841234747717，下降0.017008879545899358，达到预定0.01门槛；原valid存档差0，原18条采样ID全部精确重现。
- 同6篇valid原64-token前缀×种子42/43/44，temperature0.8/top-p0.9/penalty1.1/BF16/无缓存/新增最多128，每模型18条。完整18对审阅：1改善、12无明确整体变化、5变差，未达至少10改善/至多3变差。完整重复均值7.780309%→7.655319%，共同正文长度后6.523893%→7.704540%；不能凭降loss或短输出接纳生成质量。
- 语义审阅是助手逐对定性判断，非盲评、独立人工或事实准确率；6篇valid反复查看，不宣称泛化提升。原test仅用于原final一次报告，不参与续训对照、调参或选择。
- 按冻结协议停止，保留两份权重和全部结果；不追加训练、挑周期checkpoint、扩大语料或搜索参数。M3工程实验完成，生成质量限制如实保留，M4负责展示与复现交付。

## 当前可用产物

| 产物 | 本地位置 / 版本 |
| --- | --- |
| tokenizer | data/processed/pilot_tokenizer_16k_v1.json；SHA=21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470 |
| 原12208步final | logs/m3_training_report/m3_pretrain_100m_return_v1/ckpt_final.pt；SHA=f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d |
| 续训14208步final | logs/m3_contiune_2000_report/ckpt_final.pt；SHA=abd3d468c09b724c746e66bac769487073b17e734c8fc3f7f18956bc5bbf9f48；保留实际目录拼写 |
| 本地valid/test | data/processed/m3_final_eval_inputs_v1；本批对照未读取test，最终train.bin仍在云端 |
| 原训练/独立评估验收 | logs/m3_training_report/artifact_acceptance.json；logs/m3_final_local_evaluation_v1/artifact_acceptance.json |
| 续训原件/独立对照验收 | logs/m3_contiune_2000_report/artifact_acceptance.json（35项）；logs/m3_continuation_comparison_v1/artifact_acceptance.json（31项） |
| 全部18对文本与审阅 | logs/m3_continuation_comparison_v1/comparison_review.md、semantic_review.json/md |
| 本轮收尾证据与用户Git步骤 | logs/m3_closeout_20261015_v1 |

原M3报告完整字节归档到logs/m3_closeout_20261015_v1/m3-experiment-report.original.md，SHA=dfb21c08006423644df8173ce7a36c725bb1480eccd8ab01a688b1a447461ded。发布文档仅删除文件末尾一个多余空行，19255字节、新SHA=341f5a1dc3bc817383c0c3ca731f5a67fd64b5fbc63ea9a386ccb11bd0968b1b，内容与12条样例不变。旧协议/验收摘要对照原归档，保留历史快照，不能修改旧协议使失败对照变通过。
原100M完整输入冻结清单SHA=60ff2f6cdca5b240803e4fa63fad31a87905c6907414dd12f125dbc3677682fe。云端实际路径为/root/autodl-tmp/m3_training_input_frozen_v1/manifest.json；本地小报告导出副本名为frozen_manifest.json，不能据导出名误判云端输入缺失。
详细来源/编码/训练步骤按需查README文档导航；不重做已验收阶段。数据、tokenizer、checkpoint、日志与缓存均Git忽略，不提交或自动删除。

## 工作区、检查与协作边界

开始收尾时分支m2/tokenizer-data、HEAD=7f4cc52，暂存区为空；用户随后已运行兼容脚本，创建codex/m3-closeout并暂存33份指定文件。当前补充修正报告末尾空行及两份说明，助手未修改暂存区；用户只重新暂存docs/m3-experiment-report.md、docs/m3-closeout.md和docs/current-work.md后检查，不重跑暂存脚本。保留无关未跟踪文件，不改main，不使用git add .、清空暂存区或自动回滚。
M2历史Git收尾：提交7f4cc52，[PR #4](https://github.com/IniJyou/minigpt-cn/pull/4)已合并，merge commit25cc933，M2里程碑已关闭；不是本轮M3提交或当前远端检查结果。
本轮CPU完整回归/Ruff与文档、产物摘要和Git边界检查见 [M3收尾说明](m3-closeout.md#本轮检查)及logs/m3_closeout_20261015_v1/closeout_acceptance.json。历史702/707、724/729及各轮定向结果不能冒称本轮运行。
本页收尾前内容已原样备份到logs/m3_closeout_20261015_v1/current-work.before.md，仅供追查；不作为后续任务入口。

项目D:\aiworkspace\transformer；Windows环境.venv，WSL项目环境/home/luoyu/.venvs/minigpt-cn。作业环境/home/luoyu/.venvs/cs336-basics保持独立。当前PC Python3.12.4/Torch2.14.0+cu126，原云端Python3.12.3/Torch2.8.0+cu128；CPU读取/本地推理兼容不代表跨版本精确resume。
中文沟通，用户执行所有下载、安装与Git写操作；Windows Git HTTPS第一次远程尝试前require_escalated。保留旧pilot/tokenizer/权重、作业仓库、CS336_note.md和原始诊断产物。
数据工具已完成并冻结扩建，通用质量/隔离门槛不放宽；不自开维基模板、章节分析、全站扫描或审批平台建设。旧logs/docs_closeout_20261012_v1不读作任务计划。
预算50元含实例保留与历史费用，用户明确不再询问账单，超额由用户告知；本轮不操作云实例，也不将程序耗时当实际账单。
独立0.6B LoRA在M4之后，使用基座自己的权重与tokenizer，不套用33M的16K token ID。
