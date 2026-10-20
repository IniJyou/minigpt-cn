# M5 修订对照运行入口

2026-10-19。本轮实现并验收独立入口，默认只读inspect；没有加载本地Qwen权重、执行GPU预检、真实训练或生成。数据依据为[已验收候选包](m5-lora-revision-data.md)，范围沿用[修订定义](m5-lora-revision-plan.md)。Note2Lecture继续暂停。

## 入口与阶段

代码为[src/minigpt_cn/m5_lora_revision_run.py](../src/minigpt_cn/m5_lora_revision_run.py)，定向测试为[tests/test_m5_lora_revision_run.py](../tests/test_m5_lora_revision_run.py)，WSL脚本为[scripts/m5_lora_revision_wsl.sh](../scripts/m5_lora_revision_wsl.sh)，新执行清单为[runtime.json](../configs/m5_lora_revision_v1/runtime.json)。旧运行器、任务规范、提示、数据、回执及权重全部保持原字节。

```bash
# 默认只检查：使用现有独立M5环境，本地离线，无下载或安装。
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_revision_wsl.sh
```

只有inspect、preflight、train-compare三个阶段；test、resume、单独挑选checkpoint或额外推理均无入口。后两个阶段是后续执行能力，本轮仅验收实现，没有调用：

```bash
# 后续有相应执行授权时，先独立完成0优化步的GPU预检。
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_revision_wsl.sh --phase preflight
# 再在预检成功且回执完整的前提下，执行一次45步训练及87次开发对照。
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_revision_wsl.sh --phase train-compare
```

inspect核对完整摘要链、独立数据验收回执、全部文件树、43项M5包、安装版生成API源码、tokenizer/模板和预算。读取84条train、24条开发检查与15条旧valid，核对输入/目标预算及监督布局；只有train的84条布局传给训练阶段。旧test只核对绑定摘要，不解析、tokenize或运行。inspect不创建实验目录、不构造模型、不初始化CUDA，清理也不会因empty_cache触发CUDA初始化。

实际输出目录固定为logs/m5_lora_revision_experiment_20261019_v1。preflight独占创建目录并记录execution快照；train-compare先验证同执行摘要的成功预检回执，再独占创建自己的阶段目录。目录已存在、旧预检不通过或执行不匹配时，在模型加载前拒绝，不覆盖、恢复或重试。冻结定义/数据配置中的未执行状态保留为历史，不靠修改旧文件启动新阶段。

## 45步训练及预检

新循环复用冻结的目标/EOS监督、组内token归一化、梯度有限检查和裁剪，按已封存training_schedule逐组执行。84条顺序、每轮seed42+epoch_index打乱、两整轮168条后取第三轮12条，共45组×4=180条次。每组监督分母和总曝光必须与实际labels一致，总数固定9657；不能用3个完整epoch替代这段前缀。

模型、rank8/alpha16/dropout0.05、196个注入模块/5,046,272参数、BF16基座/FP32 adapter、AdamW和原10步warmup/180步线性衰减轨迹保持原设置。新adapter从原零B初始化开始，摘要必须与定义和预检相同；只保存step45，不续训旧adapter或另存可供挑选的检查点。状态记录第三轮偏移12、完整180条顺序、optimizer/scheduler/RNG和数据顺序摘要。

preflight复用原已验收的最长train序列两次前后向，0优化步；恢复RNG、清梯度、核对初始摘要、注入/冻结/有限梯度以及reserved≤设备总量减1GiB。训练有原1800秒墙钟检查，不量化、缩短或降级重试。实际allocated/reserved分别记录；CPU长度检查及这份入口实现不能证明完整训练显存必在8GB内。

## 87次开发对照

只有45步完整完成、9657曝光和唯一step45文件树一致，才锁定candidate_lock并开始新生成。每个条件重新加载冻结基座；适配器只读eval加载，全部参数不可训练；再次核对输入token完全一致。

| 顺序 | 条件 | 样本 | 调用编号 |
| --- | --- | ---: | --- |
| 1 | 基座，新开发检查 | 24 | 1–24 |
| 2 | 固定旧step45，新开发检查 | 24 | 25–48 |
| 3 | 新step45，新开发检查 | 24 | 49–72 |
| 4 | 新step45，旧valid回归 | 15 | 73–87 |

每次实际generate之前核对合并配置为greedy_search，并显式传do_sample=False/use_model_defaults=False/logits_to_keep=1。call_attempts.jsonl在前向前flush+fsync，记录condition、sample_id、顺序和有效模式；第88次、条件/编号错序、账本损坏或不能落盘都全局停止。配置错误继承BaseException，不能被逐条异常处理误当成一条空回复后继续。

普通逐条生成异常计空回复并保留分母；OOM/内存错误、摘要/配置/预算不符、文件树变化全局停止。失败现场、账本和原始输出保留，没有成功receipt，不重试。仅剥离终止EOS，JSON/schema/编号覆盖仍用原严格评分，不能修补、重排或捞取坏整条中的部分分数。

基座和旧step45的旧valid结果引用已封存greedy记录，重新核对编号和严格评分，不重跑或混入采样得分。新检查A/B的增益用新检查基座，新adapter旧valid的增益用旧valid基座。原九项门槛、八项旧valid不退步检查、新检查macro-F1与块全字段同时增加，以及六类完整pair计数分别报告；方向改善不能替代绝对质量通过。即便两组开发数据都过门槛，仍是诊断结果，不改旧selection、不解封test或部署。

## 本轮验证证据

Windows CPU定向检查为153 passed/1 skipped，包含39项新增入口测试与冻结M5回归。首次替身流程测试漏填context_limit，导致1失败；已在合成夹具修复，原JUnit保留，复跑结果作为最终依据。skip为Windows缺少Transformers的可选配置合并回归，不等于实际M5环境未验证。

实际M5环境另用随机极小Qwen/PEFT在CPU核对新循环：45次合成优化、180次microbatch、360个合成监督位置，冻结基座保持、adapter更新、学习率前缀一致；安装版配置合并复现旧sample行为并确认修复后greedy_search。该测试没有读取本地0.6B权重或真实候选正文，没有调用generate或初始化CUDA，不能当成实际M5训练。

证据位于logs/m5_lora_revision_runner_20261019_v1：两次Windows JUnit、cpu_checks.json、m5_cpu_integration.json、runtime绑定、实际inspect及最终摘要/链接/Ruff检查。文档测试数修正前的绑定和inspect保留，最终真实inspect单独记在inspect_final.json，才是当前本地候选的实际预算验证；不把CPU替身结果冒称Qwen生成成功。下一任务以[current-work.md](current-work.md)为准。
