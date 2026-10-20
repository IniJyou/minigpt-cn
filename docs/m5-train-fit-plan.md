# M5 训练拟合诊断：固定数据、延长同一训练轨迹

2026-10-19。本轮交付定义与只读核对证据，0模型构造、0权重加载、0tokenizer构造、0前后向、0训练、0生成、0test。机器定义为configs/m5_train_fit_v1/protocol.json；未来入口、精确状态恢复与资源预检尚未实现或通过。Note2Lecture继续暂停。

## 问题与唯一变量

[固定原例回放](m5-train-replay-results.md)确认修订step45对共同原例仅1/4全对、新增原例5/10全对，总计6/14、块全字段7/17；Proof标题、引用Example、uncertain原因等完整已见原例仍失败。当前没有证据将原因唯一归结为LoRA容量、实现或训练曝光。

本方案检验：**保持84条训练记录、初始化、样本顺序生成规则、LoRA、损失、优化器和学习率函数不变，将同一轨迹的终点从45步延至105步，所选训练原例能否改善？** 对照A为修订数据训练得到的现有step45，不是最早36条数据的旧step45。处理B从A的完整状态有计划地续行第46–105步，新增60步；不从随机初始化重训，不只加载adapter重新启动优化器。

105=84×5÷4，即五整轮、每条记录恰好呈现5次；步数依据既有曝光事实预先确定，不按后续分数挑选。它使原36条各呈现5次，但新旧数据组成、交错顺序及优化路径仍不同，不能借此与最早36条实验作因果比较。

唯一人为改变的训练因素是终止步数。更多样本呈现、监督tokens及后续学习率位置均随这段固定轨迹自然变化，因此本方案不能分别隔离“重复次数”“总token量”或“学习率阶段”的作用，也不是等token/等时间对照。单种子与已查看的开发材料不支持盲测或统计显著性结论。

## 监督与损失核对

本轮静态核对冻结training_item/collate/accumulate/train_loop，并只读安装版PEFT.forward、Qwen3.forward、ForCausalLMLoss和fixed_cross_entropy；43项包版本一致。未发现已确证的监督mask、二次shift或重复除以4问题。既有实际token布局与极小CPU模型检查是历史证据，本轮未重跑神经计算，也不把这些证据表述为所有真实BF16梯度均已证明正确。

- input_ids、labels、attention_mask形状为[B=1,L]；prefix和padding的label为-100，监督部分是紧凑target JSON及一个最终EOS。
- CausalLM内部移位一次：最后一个prefix位置预测第一个target token，target内部逐token预测，最后预测EOS；调用方不预先移位。
- 四条组成累积组G，N为四条labels[:,1:]中非-100的总数。损失为`L_G = Σ_i Σ_监督位置 CE_i / N`。安装版先求和、再除N；四次backward后不再除4。
- 这是监督token加权的组均值，不是四条记录均值的平均；没有额外类别、title或reason权重。不以更换loss权重作为本实验的第二变量。
- 每完整组只裁剪一次梯度，随后optimizer.step、scheduler.step、清梯度。原45条实际日志的顺序、组token分母和学习率均与拟定日程前缀吻合。

原checkpoint已由前项CPU验收确认：392个FP32 adapter张量、5,046,272参数，392份Adam状态均为45次更新，scheduler.last_epoch=45，epoch=3/已消费12条，Python/NumPy/Torch/CUDA RNG状态已保存。本轮只读取该封存验收，不重新加载张量。仍须在未来入口证明参数名与optimizer位置映射、RNG及中途epoch恢复等价，不能据“状态文件存在”宣称已支持精确续行。

## 固定训练日程与预算

| 项目 | 定义 |
| --- | --- |
| 基座/环境 | 原固定Qwen3-0.6B revision、43项M5包、独立WSL环境与D盘模型/缓存，不安装下载 |
| 规则/数据 | v1.2完整提示；原84条183块文件原字节不变，36条前缀和48条新增顺序不变；无重标、补样或权重采样 |
| LoRA/精度 | rank8/alpha16/dropout0.05，196处/5,046,272个FP32 trainable；BF16基座、SDPA、原checkpointing及TF32设置 |
| 优化 | 原AdamW lr=1e-4、betas/eps/weight_decay/foreach/fused、组归一化与clip=1.0不变；microbatch1、累积4 |
| 学习率 | 原10步warmup、180步线性轨迹；scheduler总步数仍180，不改为105，不重新warmup |
| 状态起点 | 修订step45的adapter、optimizer、scheduler、全部RNG、样本游标；原目录只读，写入新独占目录 |
| 顺序 | 每轮从同一84条顺序复制，以random.Random(42+零起始epoch)打乱；先完成第三轮剩余72条，再跑第四/五轮 |
| 终点 | 绝对step105，累计420条次/5整轮；本次仅60优化步、240条次，唯一新保存checkpoint为step105 |
| 总监督token曝光 | step45为9657；step105为23010；新增13353，均含每条最终EOS |
| 原36条曝光 | 当前77条次/5161 tokens；终点180条次/12230 tokens |
| 新48条曝光 | 当前103条次/4496 tokens；终点240条次/10780 tokens |
| 输入预算 | 输入4096、target含EOS1024、完整序列5120，不截断/packing；实际训练最长1961 |
| 新训练墙钟 | 1800秒，microbatch边界检查；不能中断在途CUDA操作，不因超时扩大预算 |

拟定105组、420个sample_id及每组监督分母/学习率已另存planned_schedule.json，前45组与冻结日程和实际training.jsonl逐组吻合。第46步使用lr=7.941176470588235e-5，第105步使用4.470588235294118e-5，完成后scheduler.lr=4.411764705882353e-5。终点记录已完成5轮，下一游标为零起始epoch=5、offset=0。

## 精确状态恢复与资源执行门槛

未来实现先用同版本构造路径核对相同392个trainable名称、顺序、形状、dtype及adapter摘要，再恢复完整Adam与scheduler；不得只按相同shape猜测optimizer映射。原state没有显式param_names表，需由冻结构造路径、参数枚举及CPU身份测试建立位置映射；不能证实时阻塞执行，不回退到新优化器或重训。

必须验证首次新更新仍为绝对step46、392份Adam计数均45、scheduler.last_epoch=45、lr为上述值；optimizer/scheduler构造和加载后不得额外scheduler.step。样本序列前180项保持原样，第181项起来自冻结后缀。全部模型/optimizer构造、预检与数据准备完成后，最后恢复保存的RNG，防止重新初始化adapter或前后向预检消耗续行RNG。

CPU验收须比较随机极小模型的不中断执行与“保存→重建→恢复→继续”，覆盖非零dropout、Adam动量、学习率、部分epoch及checkpointing RNG；逐参数与optimizer状态、RNG和样本序列须一致。不得为通过测试只关闭dropout。真实状态身份校验及GPU加载后/首个更新前的权重与状态摘要也须记录；失败不得训练或评测。

未来独立GPU预检最多2次最长1961-token完整训练样本前后向，0优化步；检查基座冻结、adapter名称/dtype、有限loss/梯度，随后清梯度、复核权重未变并完整恢复续行状态/RNG。预检与训练全过程均要求reserved≤设备总量减1GiB；每microbatch及optimizer更新后同步观测allocated/reserved和GPU总量。历史17.75GiB reserved记录的成因未明，不能由预检通过推断完整训练在8GB内，也不放宽限制。若越界或读数无法可靠解释，停止并保留，候选不进入评分。

planned_output为logs/m5_train_fit_experiment_20261019_v1，当前不存在。未来入口需默认CPU inspect，独立preflight与train-evaluate阶段；inspect不加载基座或初始化CUDA。只有新runtime绑定实现/CPU/实际inspect、随后独立预检验收后，才能执行有计划的60步续行。出错后的retry/resume一律不提供，不混同于本定义明确计划的一次checkpoint续行。

## 未来评测：固定53次新生成

只在60步完整完成、step105状态/权重锁定及全部完整性检查通过后，顺序执行下表。A的三组输出全部复用已封存修订step45结果，不重跑A、最早旧step45或基座。新的53次只评B；没有按中间分数提前结束或追加调用。

| 次序 | 固定集合 | B新调用 | A基准 |
| --- | --- | ---: | --- |
| 1–14 | 同一14条完整训练原例17块 | 14 | 回放candidate_step45：6/14条、7/17块、macro-F1=0.394345 |
| 15–38 | 原24条新开发检查50块 | 24 | 修订candidate_diagnostic：14/24条、26/50块、macro-F1=0.614194 |
| 39–53 | 原15条valid62块 | 15 | 修订candidate_original_valid：4/15条、47/62块、macro-F1=0.571067 |

三组绝不合并分母。greedy、seed42、eval/参数冻结/inference_mode、do_sample=False/use_model_defaults=False、1024新token上限与原调用配置保持一致；新token总上限54272。账本在每次前向前flush/fsync，固定checkpoint/集合/sample_id/有效配置与顺序，第54次全局停止。普通生成异常保留空回复与完整分母；OOM、配置/摘要/预算/账本异常全局停止，失败不出成功回执、不重试。

训练原例报告总分、common_train与revision_only_train、五完整pair、全14条字段/结构错误、标题漏取/多取及reason混淆；保留0支持的definition/heading，14条全对只表示该选集拟合。24条新检查报告12完整pair及P1–P6，15条原valid逐样本报告。所有schema/覆盖失败整条归零，不修复非法kind或reason。

## 结论规则与边界

训练拟合改善要求相对A的样本全对数与块全字段正确数都严格增加，并且两个训练分组各自的样本/块正确数均不下降；否则报告混合变化或未改善。另单独报告是否14/14、17/17全对，不把它设为部署门槛。样本、块、kind、非空title/reason及完整pair的全部变化都必须呈现。

开发保持情况要求两套开发数据各自的JSON/schema/覆盖率、块全字段率、样本全对率、macro-F1、非空title/reason八项均不低于修订step45，不以多数类增量掩盖少数类退步。沿用既有九项绝对质量门槛分别报告，基座增益只复用各自同提示/同greedy基座结果；训练原例不套开发门槛。

- 训练拟合改善且开发均不退步：支持这段固定延长轨迹有帮助，仍不证明训练量是唯一原因。
- 训练改善但开发退步：报告拟合与开发表现的取舍，不选择新checkpoint或加训。
- 训练未改善/仍错：保留具体失败，不直接断言LoRA容量不足或实现错误。
- 未完整执行或资源/状态校验失败：无有效处理组，不作完成的因果比较，也不重新解释为质量失败。

即使拟合与质量均通过，test仍封存，原selection不改，不自动部署或推进Note2Lecture。该定义没有授予执行新实验的权限；唯一下一项是实现与CPU/默认inspect及精确状态恢复边界验收，真实训练/生成留待后续。

证据在logs/m5_train_fit_definition_20261019_v1：supervision_review.json、installed_loss_api.json、schedule_summary.json、planned_schedule.json及拟定53次评测清单。既有模型/数据/源码/测试/检查点/结果和旧协议不回写。
