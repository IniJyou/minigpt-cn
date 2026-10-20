# M5首次LoRA训练与valid结果

2026-10-19。按冻结入口单次完成180步训练及60次模型valid生成，工程执行完成；三个预定检查点均未通过全部质量门槛，`accepted=false、selected_step=null`。test未运行，按协议停止，不追加轮次或挑选未通过的检查点。Note2Lecture继续暂停。

## 实际训练

沿用[实验协议v1](m5-lora-experiment-v1.md)、[运行入口](m5-lora-runner.md)及已通过的[GPU预检](m5-gpu-preflight.md)。独立WSL环境、本地Qwen/Qwen3-0.6B、v1.2完整提示及train36条95块保持不变。运行没有下载、安装、修改旧环境或合并adapter回基座。

| 项目 | 实测 |
| --- | --- |
| 参数 | rank8、alpha16、dropout0.05；196处LoRA、5,046,272个FP32可训练参数，基座冻结 |
| 初始化 | 与预检相同，adapter SHA256为`7716c1bddde7d596bf59dc7c8fb87d15f1553042c0e91ed1eb0917ea140aa3e8` |
| 训练范围 | 单个种子42，20轮，每轮36条；microbatch1、梯度累积4，共720次microbatch/180优化步 |
| 监督 | 每轮2446个target/EOS token，累计48,920次曝光；不packing或截断 |
| 优化 | AdamW lr1e-4，10步warmup后线性衰减，梯度裁剪1.0；有限loss/梯度检查通过 |
| 用时 | 422.912675917秒，约7分03秒；低于1800秒停止线 |
| 保存 | step45/90/180，独立adapter及optimizer/scheduler/RNG/样本顺序状态 |
| 在线loss | 按监督token加权，第1轮0.28094604，第20轮0.00028472；训练中权重持续变化，不是固定模型的验证loss |
| 训练峰值allocated | 5,038,476,800 bytes，约4.69 GiB |
| 训练峰值reserved | 15,294,529,536 bytes，约14.24 GiB；超过GPU报告的总量，见下文 |

训练期间PyTorch报告的reserved峰值高于GPU报告的8,585,216,000 bytes总量。第32步旁的独立`nvidia-smi`瞬时读数为total8188/used7936/free21 MiB，原样保存在`logs/m5_train_valid_acceptance_20261019_v1/memory_observation.json`。进程分配器峰值与驱动瞬时读数范围不同，现有证据没有确定这一差异的成因；不能声称完整训练显存始终在8GB以内，也不据此虚构共享内存或迁移机制结论。本次没有OOM，冻结协议的reserved≤总量减1GiB门槛适用于预检，未临时增加训练停止线。

## valid与接纳

valid固定15条62块，八类支持数依次为body38、heading9、definition1、theorem3、proof3、example1、other6、uncertain1。先用新版提示运行未适配基座15次，再完整训练，最后三个检查点各生成15次；冻结规则另算15条。所有原始回复与失败均计入分母，不修JSON或补块，不读取test用于选择。候选由助手标注，非独立人工金标；definition/example/uncertain各仅1块，不据此声称稳定泛化能力。

| 指标与原门槛 | 同提示基座 | step45 | step90 | step180 | 冻结规则参照 |
| --- | ---: | ---: | ---: | ---: | ---: |
| JSON，100% | 8/15 | 15/15 | 15/15 | 15/15 | 15/15 |
| schema，100% | 3/15 | 15/15 | 15/15 | 12/15 | 15/15 |
| 编号完整覆盖，100% | 1/15 | 14/15 | 14/15 | 11/15 | 15/15 |
| 块全部字段，≥80% | 1/62 | 39/62 | 43/62 | 35/62 | 53/62 |
| 整条完全匹配，≥50% | 0/15 | 2/15 | 3/15 | 3/15 | 10/15 |
| 固定八类macro-F1，≥0.75 | 0.140625 | 0.656981 | 0.560833 | 0.465873 | 0.522851 |
| 较同提示基座macro-F1增益，≥0.15 | — | +0.516356 | +0.420208 | +0.325248 | — |
| 非空title精确匹配，≥80% | 1/16 | 9/16 | 9/16 | 8/16 | 16/16 |
| 非空reason精确匹配，≥80% | 0/7 | 4/7 | 4/7 | 3/7 | 5/7 |

step45/90都未通过覆盖、块全部字段、整条匹配、macro-F1、非空title及reason共6项；step180另有schema失败，共7项。三者macro-F1增益均达标，但增益不是唯一接纳条件；所有门槛及选择均用未四舍五入值复算。规则只是冻结参照，其macro-F1与非空reason也低于接纳线，不据此把它替代为已接纳模型。

| 类别 | 支持块数 | step45 F1 | step90 F1 | step180 F1 |
| --- | ---: | ---: | ---: | ---: |
| body | 38 | 0.828571 | 0.853333 | 0.722222 |
| heading | 9 | 0.700000 | 0.666667 | 0.571429 |
| definition | 1 | 1.000000 | 0.666667 | 0.666667 |
| theorem | 3 | 0.666667 | 0.500000 | 0.500000 |
| proof | 3 | 1.000000 | 1.000000 | 0.666667 |
| example | 1 | 0.000000 | 0.000000 | 0.000000 |
| other | 6 | 0.727273 | 0.800000 | 0.600000 |
| uncertain | 1 | 0.333333 | 0.000000 | 0.000000 |

上述F1包含整条格式/覆盖失败产生的FN/FP，不是仅对有效回复计算的语义分数。三个检查点都在`m5-eval-v1-v08`只输出b0001–b0004，遗漏b0005；严格评分因此将该条全部5块计为失败。Amann第二卷的example块三者均预测为heading。step180另有3条schema失败：other的reason复制正文、uncertain的reason为非法值`"uncertain"`，以及uncertain的reason为null；JSON能解析并不代表schema合格。

模型格式和部分分类相对同提示基座有改善，但title、reason、完整覆盖及薄弱类别仍未达到目标。训练在线loss持续下降，三个保存点的macro-F1却依次下降；本轮结果不支持把更低训练loss当作质量通过，也没有通过归因实验确定下降原因。

本轮60次模型生成均无运行异常；基座有1条触及1024-token上限，三个adapter均为0条。新增token总数分别为基座3444、step45为1505、step90为1449、step180为1480；没有重试或修复输出。

## 产物与验收

本轮实际调用：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase train-valid
```

执行清单仍为[原runtime](../configs/m5_lora_v1/runtime.json)，SHA256为`e87b1e10dca896c7d5c9bd60eb019c04314e98e266993b422ef4f718fe7a74e4`。模型、提示、数据、训练源码和质量门槛均未修改；原协议及旧入口文档的未执行状态是冻结时记录。

训练逐步记录、完整计数、三个检查点、各条件预测/token/错误/metrics及选择结果位于`logs/m5_lora_experiment_20261018_v1/train-valid`，其中[training_result.json](../logs/m5_lora_experiment_20261018_v1/train-valid/training_result.json)记录完整训练，[selection.json](../logs/m5_lora_experiment_20261018_v1/train-valid/selection.json)锁定未通过结论，[成功阶段回执](../logs/m5_lora_experiment_20261018_v1/train-valid/receipt.json)绑定全部文件摘要。成功回执表示阶段完成及文件完整，不表示模型质量通过。

独立只读验收位于`logs/m5_train_valid_acceptance_20261019_v1`。CPU读取三个adapter及本机生成的训练状态，核对392个FP32张量/5,046,272参数、张量有限、B已非零、392项AdamW状态、step/epoch、scheduler、RNG和预定样本顺序；本地tokenizer核对60条输入与原始/去EOS回复，未构造模型或初始化CUDA。再重算180步lr/计数、48,920监督曝光、全部valid严格指标/错误和九项选择门槛，校验既有冻结产物。本轮Ruff/链接/差异检查通过，没有重跑CPU pytest；不冒称完整项目回归。

下一步是只读复核valid错误并定义下一版最小改动方案；保留原数据、提示、评分和本轮检查点，不直接追加训练或启用test。唯一下一任务以[当前工作](current-work.md)为准。
