# M5 同轨迹续行诊断：资源校验停止

2026-10-19。按[冻结方案](m5-train-fit-plan.md)、[入口](m5-train-fit-runner.md)及[已通过的独立预检](m5-train-fit-preflight.md)执行一次train-evaluate。**新增6个优化步完成至绝对step51，随后在step52第一个microbatch反向完成后的资源检查中停止。60步续行未完成，53次评测未启动，没有step105候选或成功回执。** 未重试、恢复失败运行或扩大预算；test仍封存，Note2Lecture暂停。

## 实际完成范围

| 项目 | 记录 |
| --- | --- |
| 固定runtime SHA | 4e563ea07b0811bcb5563b4e83f6b15a1b7b6d4b0cd88b121b5ae83fb1843c4f |
| 恢复来源 | 修订84条训练的原step45完整adapter/Adam/scheduler/RNG和第三轮游标 |
| 新优化步 | 6：绝对step46–51；step52未执行optimizer.step |
| 新前后向账本 | 25条：前24条组成6个完整组，第25条为step52第一条 |
| 已完成更新的监督token曝光 | 新增1108，累计9657→10765；不将最后未更新microbatch计入该累计 |
| 最后成功更新loss/lr | step51：0.028530076087918133 / 7.647058823529411e-5 |
| 出错样本 | m5-sft-v1-real-n08：输入1630、目标含EOS161、完整序列1791 tokens |
| 所在组 | step52，四条固定记录的监督分母417；其余三条尚未开始 |
| 评测/生成/test | 0 / 0 / 0 |

实际样本顺序、逐组token分母和学习率与冻结后缀吻合；已记录六步的loss及梯度范数均有限。首次新更新step46使用7.941176470588235e-5，原warmup和scheduler未重启。源step45和原预检回执保持原样。

## 停止原因与现有证据

保存的failure.json为：

```text
MemoryError: reserved headroom or memory reading integrity failed
```

终端堆栈指向accumulate_checked中的loss.backward返回、删除loss/batch后的guard()。此处尚未完成四条组，也未到step52的optimizer.step，所以本轮新更新数确定为6；第25次前后向未产生新的优化步。

ResourceGuard将三种条件归入同一个错误：设备总量读数变化、allocated/reserved峰值关系不合法、reserved峰值超过设备总量减1GiB。**当前失败记录没有保存触发时的observed值，无法确认本次是哪一项成立，也无法报告失败瞬间的显存峰值。** 该异常来自项目资源校验，终端没有CUDA OutOfMemoryError；不能据此认定发生了CUDA分配OOM。

| 已保存的显存观察 | allocated峰值 | reserved峰值 |
| --- | ---: | ---: |
| 独立最长样本预检 | 5,017,400,320 bytes | 5,771,362,304 bytes（5.375GiB） |
| 本轮已记录六步中的最大值 | 4,435,415,040 bytes | 5,110,759,424 bytes（约4.760GiB） |

设备总量为8,585,216,000 bytes，门槛为7,511,474,176 bytes（约6.996GiB）。上表第二行是已完成更新的日志峰值，不是失败瞬间的读数；不能用它证明后续资源始终在门槛内。n08在冻结输入预算内，且比预检的1961-token样本短，预检通过仍不能保证变长样本序列下全过程资源检查通过。

allocator缓存/碎片、WSL/驱动记账或采样关系等可作为后续核查方向，现有记录不足以区分，均不作根因结论。没有调整分配器、清缓存后重跑、换精度、改样本或放宽门槛。

## 实验结论

本轮没有完成的step105处理组，不能比较原例拟合、开发指标或LoRA容量，也不能把资源停止写成质量门槛失败。原step45回放和开发分数继续保持原结论；本轮没有新增预测或分数。

失败目录logs/m5_train_fit_experiment_20261019_v1/train-evaluate仅保留五份文件：started.json、input_preflight.json、training.jsonl、training_attempts.jsonl、failure.json。没有training_result.json、candidate_lock、checkpoint、生成账本或成功receipt；未保存step51权重，不存在可直接继续的step51检查点。

独立只读验收在logs/m5_train_fit_train_evaluate_20261019_v1：670项核对通过，641份前项产物原字节不变。这里“通过”指现场与停止规则的验收，execution_complete=false；不表示60步实验成功。terminal_capture.json保留单次终端退出和栈位置；验收没有加载模型、追加前后向或生成，也没有下载、安装或Git写操作。

下一项只读复核资源采样与失败样本布局，并定义最小观测补充方案：将触发条件、当时读数、阈值、阶段和样本身份完整保存，以便后续独立诊断。暂不修改旧门槛/运行器、不恢复失败目录、不安排新训练或生成；唯一入口见[current-work.md](current-work.md)。
