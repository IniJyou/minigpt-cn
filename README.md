# MiniGPT-CN

从零实现约33M参数的中文 Decoder-only Transformer，完成训练、生成和评测；
33M主线完成后，再推进独立0.6B基座的 LoRA-SFT 与多轮服务。

**继续任务先读 [当前工作与交接](docs/current-work.md)。这是唯一的下一任务入口。**
参考文档说明接口，实验日志保存历史结论，不把历史“下一步”当成当前计划。

## 当前状态

- 已验收：模型核心、权重共享、dropout、BPE编解码/增量训练、训练与完整评估、
  配置化checkpoint/恢复、生成CLI和KV Cache。
- 已具备：正文清洗/去重/划分、仅train文本导出，以及可复用的Wikimedia数据工具。
- 已验收：16000词表tokenizer保存/重载、同版本pilot三集合uint16编码与全部1000篇磁盘读回。
- 已验收：33M本地100步短训练、50步保存/恢复与连续训练状态完全一致、独立重载完整验证。
- 已验收：AutoDL/RTX4080 SUPER的500步资源验证与本地原始产物；已记录累计费用3.09元及用户保留实例运行的决定。
- 已落实：50元总预算下固定来源的有界中文训练输入与运行范围；来源许可声明已记录，不能替代逐篇质量/事实/许可保证。
- 已完成：首个中文Parquet分片的来源核对与24篇摘录审阅；发现质量问题后，已完成有界候选筛选和编码。
- 已完成：有界候选补筛、3篇确切排除、6529篇同版本16K编码与实际规模验收；train约445万tokens，本机30项报告核对通过。
- 已验收：有界33M首轮2000步训练与六份原始产物，完整valid loss=9.7774→5.6211；27项核对、final摘要/CPU重载与优化器状态通过。
- 已完成：独立valid/test与12条固定生成原始产物验收，25项核对通过，valid精确复现；生成仍明显重复/离题，展示质量未通过。
- 已完成：首分片有界扩充批次1，实际train约1187万token（新增742万）；31项核对通过，6篇明确缺项需在后续合并时暂缓，候选未直接批准训练。
- 已完成：批次2五份原始文件及24篇全文验收，41项核对通过；train约2032万token、新增约845万，4篇确切缺项将在后续合并时暂缓。
- 已完成：批次3五份原始文件与24篇全文验收，46项核对通过；train约4820万token、六个固定分片已耗尽，5000万规划目标尚差约180万；8篇确切缺项待输入冻结时剔除。
- 已验收：输入冻结/GPU预检查四份原始报告，本机40项核对与Ruff通过；8条暂缓实际剔除，train=48187994/73520篇，valid/test字节不变。
- 已验收：固定12208步的有界33M预训练、六份原始产物和CPU重载，31项核对/Ruff通过；处理100007936 tokens、49.52分钟，完整valid最终loss=3.9629/PPL=52.61，最终权重已完整保存在PC。
- 已验收：最终模型本地完整valid/test与12条固定续写四文件，31项核对/Ruff通过；valid/test loss=3.962893/3.995780，PPL=52.6093/54.3682，跨环境valid差值约9.67e-6。
- 质量结论：12条全部审阅，6条greedy明显重复，sampling离题/句意混乱或循环；生成质量未通过展示要求，不自动追加训练或调采样。
- 已整理：[M3实验结果报告](docs/m3-experiment-report.md)，汇总真实输入/训练/独立评估、资源与版本证据，原样收录全部12条续写，明确生成质量未达标。
- 已完成：按用户授权诊断生成重复，新增可选--repetition-penalty（默认1.0）；固定6个valid前缀的唯一候选1.1平均重复4-gram由63.93%降到42.00%，4篇下降/2篇上升，连贯性仍不足。95项CPU定向/3跳过与Ruff通过，见[重复诊断与解码改动](docs/generation-repetition-guide.md)。
- 已完成：同6篇valid的同位置16/64/128-token上下文诊断，FP32实跑18条并通过19项独立读回核对。真实前文NLL由4.337降至4.174，但长前缀自由续写仍循环或提前收尾；没有继续训练或改生产源码，结果见[生成重复诊断](docs/generation-repetition-guide.md#后续同位上下文诊断2026-10-15)。
- 已完成：按用户选择，固定采样温度0.8/top-p0.9、种子42/43/44，共36条valid惩罚对照；1.1平均重复比例28.41%→7.78%，共同长度控制后仍下降，但连贯性未通过。22项独立核对与Ruff通过，结果见[固定采样对照](docs/generation-repetition-guide.md#固定采样与重复惩罚对照2026-10-15)。
- 已落实：[有界续训对照方案](docs/bounded-continuation-plan.md)，仅将12208步终点延至14208，沿用3e-5与原数据，预先限定valid/连贯性判据。实际续训在原云端兼容环境执行，Windows环境不绕过原resume契约。
- 已验收：原云端只读核对回传，8项文件摘要、50项Python检查（含37份源码）通过；Torch2.8.0+cu128/CUDA12.8、原12208步恢复契约匹配。本机15项回传一致性核对通过；驱动与更早报告不同，不声称整个环境逐位等价，旧回执保留历史快照。
- 已交付并由用户执行：[固定2000步执行包与评估协议](docs/bounded-continuation-execution.md)，900秒总限时、固定14208终点。准备轮17项Windows定向检查、Ruff及ZIP读回通过；协议不因新结果改写。
- 已验收：固定续训2000步到14208，控制总耗时7.68分钟；新final、metrics和run在PC完成35项摘要/原始日志/CPU重载核对，回执为logs/m3_contiune_2000_report/artifact_acceptance.json。
- 已完成：[续训本地独立对照](docs/bounded-continuation-execution.md#本地独立对照结果2026-10-15)。同环境完整valid loss=3.962893→3.945884（下降0.017009），达到预设0.01门槛；18对完整续写审阅为1改善/12无明确变化/5变差，连贯性门槛未通过。31项独立产物核对/Ruff通过，原基线18条ID精确重现；按协议停止，不自动加步数或接纳新权重为生成质量改善。
- 已收尾：[M3阶段结论](docs/m3-closeout.md)。训练、独立评估与有界优化实验完成，生成质量限制保留，Git提交与远端里程碑关闭由用户执行。下一阶段为M4模型卡、复现说明与项目展示。
- 数据工具本轮收尾并冻结扩建；不再追求完整维基渲染、全站扫描或通用审批平台。
  未审批的候选不是训练语料，旧有损正文不会因升级源码自动修好。

## 主线顺序

1. 复用已验收pilot的train文本，保存并验收16000词表tokenizer。
2. 使用同一tokenizer编码固定train/valid/test，完成33M本地短训练和恢复检查。
3. 在实际AutoDL设备跑约500步，测显存、吞吐和稳定性，再确定费用与训练预算。
4. 准备预算明确的正式语料，正式预训练、生成、评测和展示。
5. 最后推进独立0.6B LoRA阶段。

pilot小语料只用于验证闭环。约445万train位置的首轮语料只跑有界2000步；
后续实际冻结48187994个train位置，完成100007936处理tokens，约2.075倍算术曝光。
两者不能混算，也不等于1亿唯一语料token。M3本批已停止，不自动扩展训练。

## 正式模型配置

| 配置 | 值 |
| --- | ---: |
| vocabulary / context | 16000 / 512 |
| layers / heads | 8 / 8 |
| d_model / d_ff | 512 / 1344 |
| norm / position / activation | RMSNorm / RoPE / SwiGLU |
| tied embeddings / dropout / bias | true / 0.1 / false |

配置见 [minigpt_33m.yaml](configs/minigpt_33m.yaml)，参数约33M。
2048词表的旧pilot权重不能与新16K tokenizer混用。
33M模型定位为中文文本生成实验，不承诺可靠的聊天或指令遵循能力。

## 里程碑

| 阶段 | 状态 |
| --- | --- |
| M0 Bootstrap | 基础搭建已完成 |
| M1 Transformer Core | 实现与本地验收已完成 |
| M2 Tokenizer and Training | 核心工具、16K pilot数据与33M本地短训练/恢复闭环已验收 |
| M3 33M Pretraining | 本地实验与文档收尾完成，生成质量限制已记录；Git/远端关闭由用户执行 |
| M4 Portfolio Release | 下一阶段：模型卡、真实曲线/样例、本地复现与项目展示 |
| M5 LoRA Chat and Evaluation | 独立0.6B基座与其自带tokenizer，33M主线完成后开展 |

GitHub状态、提交和合并由用户操作；此表描述本地实现，不代替远端验收。
位置编码、tokenizer和KV Cache消融属于后续评测，不阻挡首个33M闭环。

## 文档导航

- [当前工作与交接](docs/current-work.md)：范围、环境、下一任务与提交收尾。
- [Tokenizer操作](docs/tokenizer-guide.md)：已验收16K版本与保存/重载接口。
- [Token编码验收](docs/token-encoding-guide.md)：同版本三集合uint16产物、复现与独立读回。
- [33M短验证](docs/33m-short-validation.md)：本地100步、精确恢复、完整验证与资源统计范围。
- [AutoDL资源验证](docs/autodl-resource-guide.md)：500步云端实测结果、统计范围与产物归档收尾。
- [正式预训练方案](docs/formal-pretraining-plan.md)：50元预算、实测推算、正文来源候选与逐项验收范围。
- [正式来源验收](docs/formal-source-guide.md)：固定中文Parquet版本、用户获取单分片与有限样本审阅步骤。
- [正式候选三集合](docs/formal-corpus-guide.md)：复用现有规则，有界筛选、固定划分与读回，等待样本验收。
- [候选16K编码](docs/candidate-encoding-guide.md)：绑定暂缓名单与版本，磁盘全文读回及真实token规模报告。
- [首轮训练准备](docs/first-training-guide.md)：固定2000步、完整valid与周期保存、云端预检查及后续启动步骤。
- [首轮模型评估](docs/first-evaluation-guide.md)：独立valid/test、固定12条续写、原始输出与质量审阅。
- [正式语料有界扩充](docs/formal-corpus-expansion-guide.md)：首分片继续位置、旧heldout隔离、新三集合/16K编码与实际新增规模。
- [固定第2分片与批次2](docs/formal-corpus-batch2-guide.md)：用户一次执行续传、准确暂缓、累计heldout隔离、合并/编码与返回验收。
- [剩余固定分片与批次3](docs/formal-corpus-batch3-guide.md)：一次有界处理剩余4个已上传分片，同版编码/全文读回与报告ZIP取回。
- [100M运行准备](docs/pretraining-100m-guide.md)：按实际约4819万位置冻结输入、保持valid/test、固定12208步与GPU预检查，不自动开训。
- [最终模型评估](docs/final-evaluation-guide.md)：重载12208步final，完整冻结valid/test与12条固定中文续写，一次打包返回。
- [M3实验结果报告](docs/m3-experiment-report.md)：真实训练规模、资源/指标/版本、完整12条输出与质量限制，便于审阅和阶段收尾。
- [M3阶段收尾](docs/m3-closeout.md)：最终训练与优化结论、阶段完成范围、检查、用户Git步骤及M4交接。
- [生成重复诊断](docs/generation-repetition-guide.md)：原始预测/生成/缓存/因果对照，可选重复惩罚与固定valid唯一候选结果。
- [模型与运行接口](docs/runtime-reference.md)：张量、checkpoint、训练/评估/生成入口。
- [数据工具参考](docs/data-reference.md)：已实现的工具与边界，按需查询，不是待办清单。
- [数据来源与产物状态](data/README.md)：哪些能复用，哪些不能训练。
- [阶段路线](docs/learning-roadmap.md)：主线验收顺序，不再按旧十周教程重做。
- [实验结论](docs/experiment-log.md)：真实历史结果和已知限制，不是当前操作指令。

## 环境与检查

本地设备为RTX 4060 Laptop 8GB、约31GB内存；本地用于开发和短验证，正式训练在云端。
已有WSL项目环境：/home/luoyu/.venvs/minigpt-cn；作业环境单独保留。
缺依赖时由用户安装，助手不自行下载或重建环境。

Windows已有环境检查：

~~~powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q --tb=short
~~~

WSL已有环境检查：

~~~bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --tb=short
python -m ruff check .
~~~

## 数据与发布原则

原始语料、token分片、checkpoint、日志、密钥和缓存不提交Git。
保留来源、版本、许可、划分和校验和；公开权重/数据/adapter前核对适用许可。
代码的MIT许可不替代第三方语料或模型许可。项目学习笔记CS336_note.md独立维护，不修改。

## License

MIT
