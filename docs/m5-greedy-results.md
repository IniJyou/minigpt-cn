# M5 greedy valid对照结果

2026-10-19，用户明确同意追加30次后，已用冻结v2入口完成基座15次与固定step45适配器15次。**这次改用greedy未改善主要错误，step45仍未通过质量门槛；test继续封存。**

## 执行与比较口径

沿用任务规范v1.2、完整提示、valid15条62块、Qwen3-0.6B固定revision及原step45权重。推理为BF16/CUDA/SDPA，输入未截断，最多新增1024 tokens。相对原采样对照，仅采用预先定义的greedy解码；temperature/top_p/top_k/min_p均为null，num_beams=1。没有训练、额外检查点、重试或test调用。

入口为[scripts/m5_greedy_v2_wsl.sh](../scripts/m5_greedy_v2_wsl.sh)，运行时清单为[configs/m5_greedy_diagnostic_v2/runtime.json](../configs/m5_greedy_diagnostic_v2/runtime.json)，SHA256为be4db8e3f0fb3648bd6decf4cebafde198d50b8d6e466a9b370b6c5d121286eb。实际generate显式传入do_sample=False、use_model_defaults=False；每次前向前核对合并配置并持久记录attempt，实际模式均为greedy_search。

原v1中断现场保留6条完整非greedy回复，可能另有1次在途，不纳入评分。这次新增调用精确为30，连同历史尝试累计36–37次，按上界37计。[修复说明](m5-greedy-call-fix.md)及冻结协议中“追加授权待确认”是冻结时历史状态；本次授权另记在logs/m5_greedy_results_20261019_v2/authorization.json，没有修改旧配置或回执。

## 严格结果

同一15条valid配对比较，结构失败整条计错，不从无效回复捞取部分正确块。历史采样列来自[首次训练/valid](m5-train-valid-results.md)，不含中断v1回复。

| 指标 | 基座采样 | 基座greedy | step45采样 | step45 greedy |
| --- | ---: | ---: | ---: | ---: |
| JSON有效 | 8/15 | 8/15 | 15/15 | 15/15 |
| schema有效 | 3/15 | 2/15 | 15/15 | 15/15 |
| 编号完整有序 | 1/15 | 1/15 | 14/15 | 14/15 |
| 块全部字段正确 | 1/62 | 0/62 | 39/62 | 35/62 |
| 整片段完全匹配 | 0/15 | 0/15 | 2/15 | 2/15 |
| 八类macro-F1 | 0.140625 | 0.000000 | 0.656981 | 0.636535 |
| 非空标题正确 | 1/16 | 1/16 | 9/16 | 9/16 |
| 非空reason正确 | 0/7 | 0/7 | 4/7 | 4/7 |

新step45相对**同次greedy基座**的macro-F1增益为0.636535。原9项门槛中只通过JSON、schema及相对基座增益3项；其余6项失败：覆盖93.33%未达100%，块全字段56.45%未达80%，整片段13.33%未达50%，macro-F1未达0.75，非空标题56.25%和非空reason57.14%均未达80%。没有改变旧selection.json的未接纳结论。

按每条样本“块全部字段正确数”比较采样与greedy：基座0条增加、1条减少、14条不变；step45为0条增加、2条减少、13条不变。“不变”只指该计数，不表示回复字节或其他指标相同。单次固定对照不足以概括其他采样种子或解码策略。

## 仍然存在的错误

- m5-eval-v1-v08仍漏掉b0005，整条覆盖失败，使5块按严格口径全部计错。没有修补输出后重新计分。
- Amann II的5.7 Examples仍被标成heading并附加标题，参考为example/title=null。编号Proposition、正文前的Proof仍出现不合规范的标题。
- 结构有效的57块中仍有13处kind、16处title、10处reason错误，这些字段错误可重叠。16处标题错误包括5处遗漏及11处无依据添加；另5块因整条覆盖失败计错，不混入上述字段错误统计。
- Smith片段仍混淆body/uncertain与ambiguous/insufficient_context；正文中提及Proposition也被误判成theorem。

因此，这次观测支持“仅切换greedy不足以解决当前结构、标题和角色错误”。它不能单独证明数据稀缺、模型规模或某个训练参数就是原因。

## 证据与验收

完整原始结果在logs/m5_greedy_diagnostic_20261019_v2：call_attempts.jsonl、两条件predictions.jsonl/runtime.json/metrics.json/errors.jsonl、paired_changes.json、result.json及成功receipt.json。每个条件均15条EOS结束，运行异常0、达到输出上限0；新增tokens分别2422和1485，生成时间合计约242.76秒，不含摘要核对和加载时间。

只读复算在logs/m5_greedy_results_20261019_v2：acceptance.json记录392项核对通过，301份历史产物摘要未变；CPU tokenizer独立核对30条输入及生成token的原样解码，通过且没有构造模型或初始化CUDA。实际完整GenerationConfig、43项环境版本、固定adapter摘要、严格评分、逐条配对、九项门槛及调用账本均一致。新结果目录已封存，后续验收证据放在独立目录。

本轮未修改运行源码，无需重跑冻结入口的CPU测试；前项114 passed/1 skipped及实际M5配置合并/PEFT边界回归是历史检查。本轮另做Ruff、文档链接及git diff --check，最终记录为logs/m5_greedy_results_20261019_v2/final_qa.json。

这仍是反复查看过的开发valid，参考标签由助手核定，无独立人工金标，不能称作盲测或泛化结果。即便诊断门槛通过也不自动解封test。下一项建议先定义小规模训练修订对照，补足标题边界、角色词与上下文、完整编号覆盖的训练示范，并固定新的预算与评测边界；本轮不扩样、不启动新训练。唯一下一任务以[current-work.md](current-work.md)为准，Note2Lecture继续暂停。
