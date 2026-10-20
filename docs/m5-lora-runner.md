# M5 LoRA 环境验收与运行入口

日期：2026-10-18。用户完成PEFT安装后，本轮交付独立运行入口及CPU边界测试，执行已冻结的[实验协议v1](m5-lora-experiment-v1.md)。新[执行绑定](../configs/m5_lora_v1/runtime.json)另存，旧协议、数据、提示和规则/基座运行器不改。

## 实际验收

独立环境 `/home/luoyu/.venvs/minigpt-cn-m5` 的43项包与计划完全一致，只有PEFT0.17.1这一项新增；`uv pip check`通过，PEFT实际导入通过，其167个带摘要的安装RECORD条目校验通过。uv硬链接警告与D盘缓存、Linux环境跨文件系统的布局一致，已退回复制；这是[uv支持的链接模式](https://docs.astral.sh/uv/reference/settings/#link-mode)，不需要重装或重建环境。本轮未改任何安装配置。

Windows CPU定向回归为84 passed / 4 skipped，包含新增24项边界测试及原60项结构/Qwen测试；4个skip是Windows未安装可选PEFT的集成测试。随后在实际独立M5环境，用stdlib unittest执行这4项：随机极小Qwen3/PEFT的零B初始化、整组token均值梯度、冻结基座与gradient checkpointing、adapter safetensors保存/重载，全部通过。没有为测试给M5环境增加pytest。

这些集成测试只构造随机2层小模型，未读取真实Qwen权重。CPU合成优化步不属于本项目真实LoRA训练。本轮只执行真实tokenizer的inspect，不运行真实模型GPU预检、生成或训练；CPU通过不等于8GB GPU训练显存已经通过。

## 入口与阶段

新增[src/minigpt_cn/m5_lora_run.py](../src/minigpt_cn/m5_lora_run.py)与[WSL启动脚本](../scripts/m5_lora_wsl.sh)。新执行清单绑定完整新环境、旧实验协议、三集合/提示/模型文件，以及新源码、测试、启动脚本和本文摘要。未使用旧校准CLI绕过依赖或版本检查。

默认阶段是inspect，仅读取协议、完整依赖、模型侧文件清单、tokenizer及train/valid记录，核对输入预算、目标EOS、前缀掩码与2446个训练监督token。inspect不加载权重、不创建实验输出目录、不读取test标注用于选择，也不联网：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase inspect
```

后三个阶段属于后续实际模型执行；本轮没有运行以下命令：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase preflight
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase train-valid
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase test
```

- preflight：只取最长完整train序列，至多2次GPU前后向、0优化步。检查196个注入位置/5,046,272个可训练参数、初始B为0、基座冻结、梯度有限、reserved≤总量减1GiB；恢复RNG、清梯度并验证adapter未变。失败保留日志，不能据此直接训练。
- train-valid：先验预检回执，再用新版提示完整评估未适配基座valid；从与预检相同的adapter初始化训练20轮/180步。只保存step45/90/180，训练结束再各自完整评valid一次，按原九项门槛及预定排序选择。未完成训练、没有候选通过时不进入test。
- test：在读取test记录、tokenize或构造backend之前，检查完整训练与已接纳选择、阶段回执及选中checkpoint全部文件摘要。只运行基座/选中adapter/冻结规则各一次；失败不换检查点。

运行目录固定为 `logs/m5_lora_experiment_20261018_v1`，本轮尚未创建。preflight首次独占创建整个实验目录，后续每阶段也独占创建子目录；已存在的阶段、失败目录或不匹配的回执都不会覆盖/重跑。不提供CLI参数改rank、步数、数据或输出目录；变更需另存版本及执行绑定。

## 关键实现与保留输出

训练labels保留原位置，仅target JSON+EOS受监督，前缀与右侧padding为-100，CausalLM内部移位一次。每4条累积组用总监督token数N作损失分母，逐条backward，不再除以4。训练使用完整logits，生成仅保留最后位置。实际库中该分母转发与批量均值的loss/梯度等价已由CPU集成验证。

训练仅优化FP32 adapter；[PEFT的保存/加载接口](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/peft_model)用于独立adapter文件，不合并回基座，保存时明确关闭embedding保存。另存optimizer/scheduler/RNG/样本顺序和计数以追查；本版不提供自动resume。

各阶段保留started、failure或完整receipt。训练逐步保存loss/lr/监督token/梯度范数/耗时/显存；各评测保留全部原始响应、新token ID、metrics及错误，格式失败计入全部分母。OOM直接停止；其他逐条生成异常留下空响应按失败评分，不重试或修复。完成阶段前再次核对冻结摘要，再封存回执。后续阶段校验回执及整个文件树，修改选择JSON、检查点、附加侧文件均会阻止test。

环境验收、CPU实测结果和tokenizer inspect记录在 `logs/m5_lora_runner_20261018_v1`。当前环境与CPU入口已验收，真实GPU预检仍待执行，`training_ready=false`；下一项是按原协议完成GPU预检，再决定是否进入固定训练。Note2Lecture仍暂停。
