# M5收尾：流程完成，模型质量未达标

2026-10-20。按用户“无论微调的好还是差先完成M5”的要求，本阶段正式收尾。任务定义、样本规范、本地基座推理、LoRA训练、恢复、评测与手动体验均已交付；质量接纳失败如实保留。当前没有自动追加训练、调参或评测任务，Note2Lecture继续暂停。

## 交付的能力与结论

这是Qwen3-0.6B的局部笔记结构标注实验：输入1–8个已划分文本块，输出对应的kind、title、reason JSON。八类与边界见[规范v1.2](m5-task-spec-v1.2.md)。训练仅监督目标JSON与末尾EOS，原正文、公式、代码保留。

最终候选在训练原例和新开发检查上改善，旧valid的块全字段及整条全对退化；两组开发集都没有通过全部质量门槛。**工程闭环完成；尚不能作为可靠的自动标注模型接纳。** 最后step105保留用于体验和审阅，没有按分数重新挑选模型或宣布正式发布。

## 本轮实际执行

1. 一次隔离进程的分配器对照：启动前设置`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，核实native后端、有效设置及恢复后device0实际segment。从原修订step45重新恢复，完成6更新/25次前后向/69次资源检查，在原n08失败位置未复现；reserved峰值5.049 GiB。
2. 从原修订step45再次完整恢复，执行冻结的60更新/240次前后向，终点step105。沿用原Adam、scheduler、RNG与游标，新增13,353个监督token曝光，累计23,010；84条训练样本各呈现5次，沿用原180步学习率曲线，没有重置warmup。
3. 恰好53次greedy评测：14条训练原例、24条新开发检查、15条旧valid；共3,140个新增tokens。step45参照全部复用已有输出，保留完整计数和严格评分。未执行test。
4. 单独一次手动CLI冒烟推理，验证用户入口能实际加载本地基座与step105 adapter。此调用不计入53次正式对照，也不用于选模型。

本轮共66次真实优化更新、265次前后向、54次生成；前6步属于独立诊断，不被算入最后step105的训练轨迹。没有重复尝试、搜索其他参数、安装依赖、下载模型或Git写操作。

最后60步的allocated峰值5,036,883,456 bytes，reserved峰值5,779,750,912 bytes（5.383 GiB）；设备总量8,585,216,000，保持原1 GiB余量后的上限7,511,474,176 bytes（6.996 GiB）。661次原位置资源采样均通过，60次optimizer返回已逐条留账，检查点和成功回执已封存。这证明本次有界训练通过资源门槛；历史失败现场仍保留，不据此认定碎片、泄漏或驱动的唯一根因，也不声称任意规模训练都能通过。

## 最终质量对照

对照均为修订84条数据训练的step45→step105，同提示、输入与greedy设置。整条全对要求所有块的全部字段都正确；schema或覆盖失败按整条计错，不修补输出。

| 数据组 | 整条全字段正确 | 块全字段正确 | 类别macro-F1 |
| --- | --- | --- | --- |
| 训练原例回放 | 6→13/14 | 7→16/17 | 0.394345→0.750000 |
| 新开发检查 | 14→19/24 | 26→31/50 | 0.614194→0.696353 |
| 旧valid | 4→2/15 | 47→45/62 | 0.571067→0.599542 |

训练回放类别17/17正确，但一处标题缺失，整条全对仍为13/14；回放宏平均对固定八类计算，无支持类别也计入，不能与独立测试准确率混同。新开发检查的12对边界样本有8对两成员全对，较step45的5对增加；仍有两条含非法类别的结构失败，schema与完整覆盖均为22/24。

| 开发门槛 | 预定要求 | 新开发检查step105 | 旧valid step105 |
| --- | --- | --- | --- |
| JSON有效 | 100% | 24/24，通过 | 15/15，通过 |
| schema / 完整覆盖 | 均100% | 均22/24，失败 | 均15/15，通过 |
| 块全字段 | ≥80% | 31/50（62.0%），失败 | 45/62（72.6%），失败 |
| 整条全字段 | ≥50% | 19/24（79.2%），通过 | 2/15（13.3%），失败 |
| 类别macro-F1 | ≥0.75 | 0.696353，失败 | 0.599542，失败 |
| 非空标题正确 | ≥80% | 2/6，失败 | 11/16，失败 |
| 非空reason正确 | ≥80% | 5/7，失败 | 4/7，失败 |

相对同提示裸基座的macro-F1增益门槛两组均通过，但无法抵消其他失败项。冻结的方向判定为`training_fit_improved=true`、`development_both_nonregression=false`；支持“这段续训改善已见原例拟合，同时存在开发表现取舍”。它不能单独证明LoRA容量不足、某种算法错误或唯一数据成因。

参考标注由助手核定，开发材料已多次查看；本轮不是盲测或独立人工金标。test的9条36块继续封存，`automatic_test_allowed=false`，没有独立test成绩或可靠泛化结论。

## 亲手体验

### 查看完整真实结果

打开[离线结果查看器](../logs/m5_closeout_20261020_v1/results_viewer.html)：三组数据、53个输入、step45/105共106份结果，含参考标注、结构检查、错误记录和原始回复。可切换样本和版本；打开页面不运行模型。查看器保留全部正确及失败样本，JSON格式化展示不修改保存内容或评分。

### 在WSL中实际标注一段文本

现有本地部署使用Transformers加载基座，再由PEFT加载LoRA adapter。每次CLI启动加载一次、生成一次并退出。基座与adapter保持独立文件，服务端API和Ollama不属于本阶段交付。

先检查入口与固定adapter位置（不加载真实模型，不生成）：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_demo_wsl.sh
```

实际GPU推理：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_demo_wsl.sh --text 'Definition: A group is a set with an associative operation, identity and inverses.' --run
```

本轮上述命令已实际完成一次，原回复为：

```json
{"annotations":[{"block_id":"b0001","kind":"definition","title":null,"reason":null}]}
```

终端会注明“质量未通过接纳门槛的实验候选”，并打印完整记录目录。`--text`原样构成一个块，不自动切段；模型可区分该简单定义，不代表复杂列表、标题和混合角色已可靠解决。

多块输入用UTF-8文件（1–8块，示例可保存为项目logs目录中的`my_m5_input.json`）：

```json
{"blocks":[{"block_id":"b0001","text":"Definition: A metric is a function."},{"block_id":"b0002","text":"Prove that the sequence converges."}]}
```

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_demo_wsl.sh --input-json /mnt/d/aiworkspace/transformer/logs/my_m5_input.json --run
```

该多块命令是使用说明，本轮没有额外执行。输入上限4096 tokens，超限直接停止，不截断；每次最多生成1024 tokens，默认greedy、不自动重试、不修补预测。新手动输出另建`logs/m5_manual_demo_*`目录，不修改固定评测。未来每次显式`--run`都会真实调用模型。

### 文件、环境与流程

| 产物 | 位置 |
| --- | --- |
| 原始基座 | `data/raw/m5_models/qwen3_0_6b_c1899de` |
| 最后step105 adapter与训练状态 | `logs/m5_finish_experiment_20261020_v1/checkpoints/step_000105` |
| 体验的固定文件摘要 | [configs/m5_demo_v1.json](../configs/m5_demo_v1.json) |
| 正式续训与53次输出 | `logs/m5_finish_experiment_20261020_v1` |
| 本轮只读验收、106份对照与查看器 | `logs/m5_closeout_20261020_v1` |
| 固定分配器对照 | `logs/m5_allocator_experiment_20261019_v1`（目录日期沿用旧定义，实际本轮运行） |

独立WSL环境为`/home/luoyu/.venvs/minigpt-cn-m5`，Python3.12.3、Torch2.8.0+cu126、Transformers4.57.6、PEFT0.17.1。固定Qwen revision为`c1899de289a04d12100db370d81485cdf75e47ca`。基座约596M参数冻结，LoRA rank8/alpha16/dropout0.05，196处投影共5,046,272个可训练参数；使用该基座自己的tokenizer。

实际完成的路径是：用户准备独立环境与D盘模型 → 默认inspect → GPU预检 → 监督JSON的LoRA训练 → 保存adapter及完整训练状态 → 固定开发评测 → 单次有界恢复续训 → 最后评测 → 加载基座与adapter手动推理。训练和评测主要在后台CLI执行，已有完整日志；没有常驻聊天窗口。手动入口见[m5_demo.py](../src/minigpt_cn/m5_demo.py)和[WSL脚本](../scripts/m5_demo_wsl.sh)。

最后adapter SHA256为`a1c424a3ef263f25ba8013229bfaeff7a627073ead6effcf31807b0616826a8f`，训练状态SHA256为`999f9eee574a2387bc738819666b978d5f8c4377bdd37116614132243302b75d`。模型、数据、缓存、日志与检查点均保持Git忽略；源码和配置单独交付，没有提交权重。

## 验收与停止状态

Windows现有环境M5 CPU回归：**371 passed / 5 skipped**，跳过为可选依赖；其中本轮续训与体验入口新增23项CPU测试。独立M5环境没有pytest，未安装；另以实际Torch CPU断言验证新增optimizer留账不改变参数、Adam或RNG，未初始化CUDA。默认体验inspect在禁止模型库导入的保护下通过，实际WSL手动推理成功。

最终53次结果的1,573项独立只读核对通过，774份前项产物原字节不变；包含240条前后向顺序、60条更新返回、661条资源读数、checkpoint、53次调用、全部严格指标、成对/分层报告和worker回执。查看器通过JS编译、106份内容和全部选择器/前后导航的DOM替身验证，未声称浏览器像素或截图验收。项目Ruff及本轮根层验收脚本检查通过。

旧协议、原始数据、历史失败现场和权重全部保留。活动入口已改为“M5已收尾，等待用户选择新工作”；不再把历史建议接成待办。用户以后若希望改进质量或做服务化部署，可另定目标与预算，本轮M5交付已经完成。
