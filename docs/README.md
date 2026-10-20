# 项目目录与文档导航

**下一任务以 [current-work.md](current-work.md) 为准。** 本页整理目录职责与文档用途；参考资料和历史操作说明不构成待办清单。

## M5已收尾

2026-10-20：工程流程完成，模型质量未通过接纳门槛，test继续封存，当前没有自动后续实验。先读[最终结论与手动体验](m5-closeout.md)，或直接打开[GitHub交付与本地查看说明](m5-github-closeout.md)。

## M5规范与过程记录

以下表格保留各轮交付时的状态；“待运行”“下一项”等历史表述不作为当前任务。

| 入口 | 用途 |
| --- | --- |
| [GitHub交付说明](m5-github-closeout.md) | 源码公开范围、仅本地的查看器和固定检查点使用边界 |
| [M5收尾与手动体验](m5-closeout.md) | 完成60步续训、53次评测和实际CLI体验；质量失败如实报告，停止追加实验 |
| [当前工作与交接](current-work.md) | M5已收尾，等待用户选择新工作 |
| [主线验收路线](learning-roadmap.md) | M0—M5阶段关系与工作节奏 |
| [M5任务与样本规范v1.2](m5-task-spec-v1.2.md) | 当前规范：增加Examples，列表、缩写、证明请求规则沿用v1.1 |
| [样本schema v1.2](../configs/m5_sample_v1_2.schema.json) | 当前样本机器格式；旧版本产物保留，不覆盖 |
| [v1.2数据候选补齐](m5-data-v1.2.md) | train36/valid15/test9条；评测两侧八类及最低家族覆盖通过，tokenizer/监督掩码已核对；本次训练的数据依据 |
| [最小LoRA实验协议v1](m5-lora-experiment-v1.md) | 冻结rank8、180步、3个预定检查点及valid/test接纳线；保留定义时的依赖盘点和用户安装命令 |
| [PEFT环境验收与运行入口](m5-lora-runner.md) | 43项依赖/PEFT导入通过；CPU定向84项及独立环境4项集成通过，inspect通过；保留入口冻结时状态 |
| [真实GPU预检结果](m5-gpu-preflight.md) | 预检时最长train序列1961 tokens，两次前后向/0优化步；参数/梯度/冻结检查通过，reserved约5.34 GiB；完整训练结果另列 |
| [首次LoRA训练与valid结果](m5-train-valid-results.md) | 180步/720次microbatch/48,920监督曝光及60次模型valid生成完成；三个检查点均未通过质量门槛，test未运行，保留显存异常统计与完整回执 |
| [valid错误复核与最小对照定义](m5-valid-error-review.md) | 分清整条结构失败及title/reason/角色错误；保留零训练greedy对照的冻结定义 |
| [greedy入口中断与调用边界修复](m5-greedy-call-fix.md) | 保留v1解码覆盖中断及v2修复/CPU验收的历史现场；追加授权后的结果另列 |
| [30次greedy valid对照结果](m5-greedy-results.md) | 基座15+step45十五次完成；step45完整覆盖14/15、块全字段35/62、macro-F1=0.637，未达门槛，test仍封存 |
| [成对训练示范修订定义](m5-lora-revision-plan.md) | 冻结定义：六类边界、48条新增train/24条开发检查、内容家族隔离及45步/87次对照上限；实际执行结果另列 |
| [六组成对说明示例](examples/m5_lora_revision_pairs_v1.json) | 12条原创spec_demo，排除于训练/评测；标签、标题和真实tokenizer/监督布局通过 |
| [成对修订候选数据验收](m5-lora-revision-data.md) | 数据轮验收：48条88块新增train、24条50块开发检查，旧36条保留后合并84条183块；真实tokenizer通过，45步监督曝光9657 |
| [修订对照运行入口](m5-lora-revision-runner.md) | 冻结入口及交付时验收：默认inspect；Windows153 passed/1 skipped、实际M5 CPU及123条真实inspect通过 |
| [修订对照独立GPU预检](m5-lora-revision-preflight.md) | 预检轮记录：最长1961 tokens完成2次前后向，reserved峰值5.338 GiB，0优化步/0生成；423项只读核对通过 |
| [45步修订训练与87次对照结果](m5-lora-revision-results.md) | 新检查块全字段23→26/50、旧valid35→47/62，但两组macro-F1下降，方向/绝对质量未通过；647项验收及410份历史摘要通过，test封存 |
| [修订失败样本与训练覆盖复核](m5-lora-revision-error-review.md) | 39条112块/78份输出及84条183块训练完整复核；具体形态缺口、已有示范未稳定迁移、原回放180→77条次并存，规范/标签不改；525项核对及453份历史摘要通过，0训练/0生成 |
| [固定训练样本回放诊断定义](m5-train-replay-plan.md) | 冻结14条17块、共同训练4条与仅修订训练10条；旧新step45各14次、28次上限，588项定义验收及476份前项摘要通过 |
| [固定训练原例回放入口](m5-train-replay-runner.md) | 50项新CPU测试全部通过，M5回归203 passed / 5可选依赖skip；实际默认inspect核对14条17块、43项环境与有效greedy，未加载权重/初始化CUDA/生成，入口交付轮未执行回放 |
| [固定训练原例回放结果](m5-train-replay-results.md) | 28次全部EOS；共同训练4→1/4、新增3→5/10，总样本全对7→6/14、块10→7/17；677项独立复算与28份CPU token记录核对通过，523份前项不变，0新训练/0test |
| [固定数据训练拟合诊断定义](m5-train-fit-plan.md) | 只改变同轨迹终点45→105步；计划从修订step45完整状态续行60步/13353监督tokens，使84条各呈现5次；拟定53次新评测，2128项定义核对、556份前项不变，当前0训练/0生成，精确恢复和资源门槛待验收 |
| [固定数据训练拟合诊断入口](m5-train-fit-runner.md) | 新增54项CPU测试全过，M5回归257 passed / 5可选依赖skip；实际极小Qwen/PEFT非零dropout的105步与45+60步精确一致，默认inspect核对84/14/24/15条及392参数身份；1212项独立核对、585份前项不变，未加载真实基座/初始化CUDA/训练/生成 |
| [同轨迹续行独立GPU预检](m5-train-fit-preflight.md) | 从修订step45完整状态恢复，最长1961 tokens完成2次前后向，0优化步/0生成；权重/Adam/scheduler不变、全部RNG恢复，reserved峰值5.375GiB低于6.996GiB上限，646项核对及619份前项摘要通过 |
| [同轨迹续行资源停止](m5-train-fit-results.md) | 新增6步至step51，step52首条反向后触发资源校验，25次前后向账本、0生成，无step105或成功回执；670项现场核对与641份前项摘要通过，失败瞬间读数未保存、具体子条件未确定 |
| [资源复核与最小观测定义](m5-memory-observability-plan.md) | 700项只读核对、663份前项不变；明确缺失的失败值、条件与更新边界；拟定6步/25次有界观测但未授权执行，下一项仅实现入口及CPU/默认inspect验收 |
| [独立资源观测入口及CPU验收](m5-memory-observe-runner.md) | 新42项测试、M5回归299 passed/5 skipped；25条/417分母下参数、Adam、scheduler、RNG和梯度精确一致，双端受保护默认inspect通过；727项验收与679份前项摘要通过，GPU尚未运行 |
| [有界GPU观测结果](m5-memory-observe-results.md) | 6更新/25前后向/69采样，在n08反向后确认仅reserved余量越界：7.807GiB > 6.996GiB；0生成/无checkpoint/无重试，966项现场核对与712份前项不变，底层allocator原因尚待验证 |
| [隔离分配器候选入口及CPU验收](m5-allocator-runner.md) | 新49项、M5回归348 passed/5 skipped，父环境不变且冲突即停；双端受保护默认inspect及768项验收通过，738份前项不变；GPU支持与效果尚未验证 |
| [合成规范示例v1.1](examples/m5_spec_examples_v1_1.json) | 16条规范示范，不能作为真实采集或盲测结果 |
| [完整提示与规则基线](m5-rule-baseline.md) | 固定提示、摘要协议、严格评分与规则54块校准复现 |
| [M5环境与基座就绪盘点](m5-base-readiness.md) | Windows/WSL实际缺项、固定模型revision与未执行的准备命令 |
| [M5环境与D盘准备](m5-environment-setup.md) | 用户执行的独立WSL环境/固定模型准备脚本，模型与HF/uv缓存放D盘 |
| [M5环境与模型只读验收](m5-base-acceptance.md) | 10月17日75项只读检查，作为首次推理前的准备记录 |
| [Qwen本地运行协议](m5-qwen-baseline.md) | Transformers入口、固定模型/模板/采样/token预算与复现命令 |
| [Qwen首次校准结果](m5-qwen-results.md) | GPU链路已跑通；严格类别0/54、完整覆盖1/16及错误复核 |
| [M5有界诊断定义](m5-diagnostic-plan.md) | 三种格式例×两种输入方式，配对合成12块、六份冻结协议与48次调用上限 |
| [六条件诊断结果](m5-diagnostic-results.md) | 48次全部完成；格式例敏感，去例后能输出多块但字段/语义仍失败 |
| [最小SFT候选与评测清单](m5-sft-candidate.md) | 36条95块、固定提示/loss接口与接纳线；初版评测预留的历史起点 |
| [评测候选与来源核对](m5-evaluation-data.md) | 11份来源核对、16条69块候选；隔离共享例题，保留类别/家族缺口；3本PDF的有限样页探测 |
| [Amann第一卷局部转录](m5-amann-excerpt.md) | 指定PDF页31/32/35，5条22块及页图/目标对照；整套三卷同组，合并评测候选v2为21条91块 |
| [历史规范v1](m5-task-spec.md) | 原规则、schema及12条原样例保留，用于版本对照 |
| [历史规范v1.1](m5-task-spec-v1.1.md) | 列表、缩写及证明请求修订；旧校准、基座与候选版本的依据 |

M5使用独立基座、tokenizer与环境，不混用33M的16K token ID。Note2Lecture暂不推进；具体采集与训练范围按当前工作入口确定。

## 33M成果、展示与复现

| 入口 | 用途 |
| --- | --- |
| [模型卡](../MODEL_CARD.md) | 模型版本、指标、能力限制与产物绑定 |
| [项目展示](showcase.md) | 训练曲线、完整固定生成与结果解读 |
| [本地复现](local-reproduction.md) | 同版权重与tokenizer核对、本地生成命令 |
| [M4验收说明](m4-acceptance.md) | 展示交付时的检查与发布步骤记录 |
| [M3实验报告](m3-experiment-report.md) | 实际训练、独立评估与原始样例 |
| [M3收尾](m3-closeout.md) | 最终结论、保留权重、检查与Git收尾记录 |

原12208步final为展示基线，续训权重保留作对照；生成重复和连贯性限制继续保留。

## 按需查询的实现与接口

| 文档 | 查询内容 |
| --- | --- |
| [运行接口](runtime-reference.md) | 模型、训练、评估、生成与缓存入口 |
| [Tokenizer操作与验收](tokenizer-guide.md) | 16K BPE保存、重载、编解码与摘要 |
| [同版本编码验收](token-encoding-guide.md) | uint16编码、manifest和磁盘读回 |
| [数据工具参考](data-reference.md) | 已有工具接口；扩建冻结 |
| [数据来源与产物状态](../data/README.md) | 本地pilot、来源清单、版本与状态 |

## 已完成实验的过程文档

以下文件保留原路径，供追查当时的范围、命令和证据。使用其中命令前，需要确认当前目的、依赖、路径与已有输出；查阅记录不表示授权重跑历史实验。

| 阶段 | 过程文档 |
| --- | --- |
| 本地短验证与云端资源 | [33M短训练/恢复](33m-short-validation.md)、[AutoDL 500步资源验证](autodl-resource-guide.md) |
| M3来源与早期方案 | [初始预训练方案](formal-pretraining-plan.md)、[固定来源验收](formal-source-guide.md)、[正文候选](formal-corpus-guide.md) |
| 候选编码与扩充 | [候选编码](candidate-encoding-guide.md)、[扩充批次1](formal-corpus-expansion-guide.md)、[批次2](formal-corpus-batch2-guide.md)、[批次3](formal-corpus-batch3-guide.md) |
| 首轮2000步 | [首轮训练](first-training-guide.md)、[首轮独立评估](first-evaluation-guide.md) |
| 固定100M处理量实验 | [输入冻结与训练准备](pretraining-100m-guide.md)、[最终模型独立评估](final-evaluation-guide.md) |
| 生成诊断与有界续训 | [重复诊断](generation-repetition-guide.md)、[续训对照方案](bounded-continuation-plan.md)、[续训执行与结果](bounded-continuation-execution.md) |
| 早期阶段结论 | [已验收实验记录](experiment-log.md) |

旧文档备份logs/docs_closeout_20261012_v1仅供用户追查历史，不作为后续任务计划。

## 目录职责与保留原则

| 路径 | 内容与整理方式 |
| --- | --- |
| src/minigpt_cn/ | 项目源码，按模型、tokenizer、数据、训练、评估、推理分模块 |
| tests/ | 实现与契约测试，fixtures保留在测试目录 |
| configs/ | 模型、实验配置、数据策略与M5 schema |
| scripts/ | 已有离线展示构建等脚本 |
| docs/、docs/assets/ | 活动入口、分类参考、派生展示图；docs/examples/保存可公开示例 |
| data/raw/、data/processed/ | Git忽略的来源与处理产物，保留版本目录与manifest，不按体积自动删除 |
| checkpoints/ | Git忽略的本地训练权重，旧pilot和短验证产物保留 |
| logs/ | Git忽略的运行记录、诊断、云端回传包及部分权重；保留实际路径和历史拼写 |
| .venv/ | 已验收33M项目环境，M5环境另行准备 |
| .pytest_cache/、.ruff_cache/、__pycache__/ | 可重建缓存；清理时限定到这些明确路径，检查链接与占用，不扩展到环境、数据或日志 |
| tmp/ | 当前本地临时目录；仅凭名称不能判定未来内容可删除 |

关键产物的准确位置与SHA见[当前工作](current-work.md)和对应验收记录。部分checkpoint实际保存在logs中；保持这些路径有助于继续使用已有命令、摘要和复现材料。

仓库根目录的[协作约定](../AGENTS.md)、[项目介绍](../README.md)、[模型卡](../MODEL_CARD.md)、[许可证](../LICENSE)、[项目配置](../pyproject.toml)保持原位。CS336_note.md是独立学习笔记，不参与项目清理；用户无关文件与暂存区保持原状。
