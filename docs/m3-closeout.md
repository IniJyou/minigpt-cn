# M3阶段收尾：33M中文有界预训练

收尾日期：2026-10-15。用户决定结束本批优化，转入项目展示。

**M3本地实验与文档收尾完成；生成连贯性不足作为最终限制保留。**
本次阶段完成指模型实现、可追溯输入、固定训练、权重保存/恢复、独立评估及优化对照已有完整证据。没有把生成质量判为通过，也没有把本地收尾冒称GitHub提交、CI或里程碑已关闭。

## 已交付结果

| 项目 | 最终记录 |
| --- | --- |
| 模型 | 33104384参数，8层/8头，context512，权重共享 |
| tokenizer | 16000词表，15743 merges，EOS256；始终使用同版token ID |
| 冻结train规模 | 73520篇，48187994个token位置；不是1亿份不同token |
| 原训练 | 从随机权重训练12208步，处理100007936 tokens；约49.52分钟 |
| 原模型本地独立valid/test | loss=3.962893003020671 / 3.9957795353415366 |
| 原模型生成 | 12条固定输出全部保留；明显循环、离题或句意混乱 |
| 解码改动 | 可选repetition_penalty，默认1.0；不宣称修复连贯性 |
| 固定采样对照 | 6篇valid×3种子×惩罚1.0/1.1，共36条；表面重复缓解，连贯性未通过 |
| 有界续训 | 恢复原12208步final至14208步，追加2000步/16384000 tokens，LR全程3e-5 |
| 两模型同环境完整valid | 3.962893003020671→3.9458841234747717，下降0.017008879545899358，超过预定0.01门槛 |
| 续训18对连贯性审阅 | 1改善、12无明确整体变化、5变差；至少10改善/至多3变差门槛未通过 |
| 最终决定 | 本批实验停止，保留两份权重与全部结果，不继续加步数或搜索解码参数 |

续训对照两模型均使用固定6篇valid的原64-token前缀、种子42/43/44、temperature0.8/top-p0.9/repetition_penalty1.1、BF16/无缓存/最多新增128 token。原模型18条重新生成，ID与旧采样基线逐条精确一致；两份完整valid均覆盖297389个目标，原valid与本地存档差0。
完整输出平均重复token 4-gram为7.780309%→7.655319%；剔除EOS并截共同正文长度后为6.523893%→7.704540%。较短、较少重复或更低loss都没有替代连贯性判据。
语义判断是助手逐对完整审阅，附两侧原文证据；不是盲评、独立人工或事实准确率。6篇valid已反复查看，不宣称未见数据泛化改善。续训对照没有读取test，也没有按test选择权重或延长训练。

## 证据与保留产物

- [原M3实验报告](m3-experiment-report.md)：内容和全部12条输出保持，提交前仅删除文件末尾一个多余空行；原始字节另存logs/m3_closeout_20261015_v1/m3-experiment-report.original.md，不改历史结论或旧协议。
- [生成重复诊断与采样对照](generation-repetition-guide.md)：诊断、可选惩罚和受控结果。
- [续训执行与本地独立对照](bounded-continuation-execution.md#本地独立对照结果2026-10-15)：固定协议、实际结果和停止结论。
- 原12208步final：`logs/m3_training_report/m3_pretrain_100m_return_v1/ckpt_final.pt`，397351864字节，SHA-256=`f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d`。
- 续训14208步final：`logs/m3_contiune_2000_report/ckpt_final.pt`，397352056字节，SHA-256=`abd3d468c09b724c746e66bac769487073b17e734c8fc3f7f18956bc5bbf9f48`。沿用实际目录拼写，两份权重均不覆盖。
- tokenizer：`data/processed/pilot_tokenizer_16k_v1.json`，SHA-256=`21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470`。
- 原训练/独立评估回执：`logs/m3_training_report/artifact_acceptance.json`、`logs/m3_final_local_evaluation_v1/artifact_acceptance.json`。
- 续训原始产物/独立对照回执：`logs/m3_contiune_2000_report/artifact_acceptance.json`、`logs/m3_continuation_comparison_v1/artifact_acceptance.json`；全部18对原文与审阅均在后一个目录。
- 本轮收尾核对：`logs/m3_closeout_20261015_v1`。旧回执中的pending、m3_complete=false等保留为当时快照，由本页与本轮收尾回执补充当前阶段状态。

这些数据、checkpoint和日志继续Git忽略；可提交的是代码、测试、配置、可复现说明和如实记录结果的文档。旧pilot、旧2048 tokenizer/权重、作业仓库和CS336_note.md保持原样。
上游语料版本/许可声明可追溯，但不代表逐篇许可、事实或语义质量批准；M3收尾不改变原候选标志，不自动公开语料或权重。最终实际账单与云实例当前状态未核实，不把程序耗时折算当实际账单，不再次询问费用或自动操作实例。

## 本轮检查

本轮重新运行已有环境的完整CPU pytest与Ruff，CUDA_VISIBLE_DEVICES=-1、OMP/MKL线程1；不是复述历史结果：

| 环境 | 本轮pytest | 本轮Ruff |
| --- | --- | --- |
| Windows项目.venv | 736 passed / 10 skipped，31.66秒 | 通过 |
| WSL项目环境 | 741 passed / 5 skipped，20.02秒 | 通过 |

跳过项为隐藏CUDA后的GPU测试以及Windows符号链接权限测试。Windows首次默认沙箱临时目录无法创建，327 passed/3 skipped/416 fixture errors；完整失败日志保留，没有修改源码或测试。仅将重跑的basetemp指定到本轮新工作区目录，确认目标不存在且位于本轮目录后运行，完整通过。WSL调用因沙箱E_ACCESSDENIED在获得执行升级后使用原环境运行，没有安装依赖。
原始命令、退出码、耗时和stdout保存在logs/m3_closeout_20261015_v1的windows_checks.json、windows_workspace_checks.json、wsl_checks.json与对应日志；pytest显示时间与外层命令墙钟时间不同。
没有重跑模型训练、valid/test或生成；上面的GPU结果来自已验收实验。CPU回归中的合成小模型训练/恢复测试不等于重新训练用户33M权重。
另外核对原始回执与关键产物摘要、活动文档链接、提交白名单、Git忽略规则和暂存区保持情况。用户实际暂存后发现新报告末尾多余空行；先前git diff --check未覆盖尚未跟踪的报告，现补充检查全部33份文件的拟提交内容。
原报告完整字节已归档，SHA-256仍为`dfb21c08006423644df8173ce7a36c725bb1480eccd8ab01a688b1a447461ded`；发布文档仅去掉末尾1个LF，19255字节，SHA-256=`341f5a1dc3bc817383c0c3ca731f5a67fd64b5fbc63ea9a386ccb11bd0968b1b`。旧协议/回执不改，原报告摘要可对照归档文件，不能拿发布文档的新摘要冒称原始字节未变。修正记录在同目录staged_whitespace_fix.json。

## 用户执行Git收尾

开始收尾时分支为`m2/tokenizer-data`、HEAD为`7f4cc52`、暂存区为空。助手没有执行Git写操作或远程操作，未知GitHub当前状态；M2已合并的历史记录不代表本批M3已经提交。
本轮已准备`logs/m3_closeout_20261015_v1/git-steps-v2.ps1`、`pr-description.md`与`milestone-closeout.md`供用户执行/粘贴。v2补充UTF-8 BOM，经PowerShell5.1和7语法检查，原脚本保留。脚本使用明确的文件白名单，不包含日志、数据、权重、学习笔记和无关未跟踪文件。用户现已创建`codex/m3-closeout`并暂存33份文件；格式修正后只需重新暂存这份报告和两份说明，再检查diff，不重跑要求空暂存区的脚本。不使用`git add .`、清空暂存区、强推或改写main。
用户推送后创建PR，核对GitHub M3对应的issue与验收项，CI通过后合并并关闭已完成事项及里程碑。未完成的展示工作进入M4，结论写清生成质量限制，不把失败的连贯性对照改成通过。远端动作完成之前，本项目只声称本地M3收尾完成。

## 下一阶段

下一项是M4的模型卡与本地复现说明：定位为33M中文文本生成实验，展示真实训练曲线、固定输出及限制；沿用原12208步final作为已完整评估的基线，续训权重保留为对照，不替换原质量结论。
本轮不启动M4实现或新的训练。独立0.6B LoRA仍在M4之后，使用其自身基座/tokenizer另行开展。
