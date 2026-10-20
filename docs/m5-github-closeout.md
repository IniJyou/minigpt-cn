# M5 GitHub交付说明

2026-10-20。M5工程实验已完成，模型质量未达接纳门槛；完整结果和停止结论见[M5收尾报告](m5-closeout.md)。GitHub交付分支为[`codex/m5-closeout`](https://github.com/IniJyou/minigpt-cn/tree/codex/m5-closeout)。

## 仓库包含什么

本次提交包含结构标注规范v1–v1.2、JSON schema、提示词、原创合成说明示例、冻结实验配置、运行代码、CPU测试、WSL入口和各轮结果报告。配置中的本地产物路径与摘要用于核验原实验，不表示相关文件已公开提供。

原始笔记、电子书、训练与评测数据、模型权重、adapter、缓存、运行日志及完整预测查看器仍留在本地，受Git忽略规则保护。实验报告中的`logs/`、`data/raw/`、`data/processed/`链接属于原工作区的产物定位，在仅含源码的新克隆中不可用。无需为消除这些历史定位而上传原始材料。

## 本地体验

原工作区保存着全部53组step45/105对照和参考标注，共106份输出。直接用浏览器打开：

```text
D:\aiworkspace\transformer\logs\m5_closeout_20261020_v1\results_viewer.html
```

使用原机器已经准备好的独立WSL环境、本地Qwen基座与step105 adapter，可以执行：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_demo_wsl.sh --text 'Definition: A metric is a function.' --run
```

不加`--run`只检查输入和固定adapter，不调用模型。每次显式运行最多生成一次；原始回复和输入另存到本地`logs/m5_manual_demo_*`。这是实验标注入口，复杂列表、标题和混合角色仍存在错误。

**只有GitHub源码的新克隆不能直接运行这个固定检查点。** 最后adapter与完整训练状态位于原工作区`logs/m5_finish_experiment_20261020_v1/checkpoints/step_000105`；基座位于`data/raw/m5_models/qwen3_0_6b_c1899de`。同版本产物缺失时按原校验停止，不会自动下载或用其他模型替代。环境准备历史见[独立M5与D盘准备](m5-environment-setup.md)。

## 最终实验结论

最后从修订step45完整状态续行60次更新至step105，完成240条次、13,353个新增监督token曝光和53次固定评测；训练reserved峰值5.383 GiB，低于原6.996 GiB门槛。另有一次独立CLI冒烟推理成功，不计入正式评测。

| 数据组 | 整条全字段正确，step45→105 | 块全字段正确，step45→105 |
| --- | --- | --- |
| 训练原例回放 | 6→13/14 | 7→16/17 |
| 新开发检查 | 14→19/24 | 26→31/50 |
| 旧valid | 4→2/15 | 47→45/62 |

两个开发集均未通过全部质量门槛，test的9条36块继续封存，没有独立test成绩。训练拟合改善与旧valid退化并存，不宣称可靠自动标注或泛化提升。当前没有自动追加训练、调参或评测任务；Note2Lecture继续暂停。

## Git交付核验

本轮按用户明确要求提交和推送M5，在独立任务分支操作，保留远端main已有的M4说明。运行代码、配置、冻结报告和旧权重保持原身份；新增的本说明用于区分源码交付与本地产物。

源码隔离副本的完整CPU检查为**1049 passed / 79 skipped**；其中64项需要未公开的本地冻结清单，明确跳过，其余15项为GPU、可选依赖或Windows符号链接权限。新增6项检查确认只跳过对应集成fixture、普通合成测试仍执行；已有文件即使损坏也不会被当作缺失跳过，仍由原测试校验。原机器本轮M5为**377 passed / 5 skipped**，64项本地集成检查正常执行，原冻结测试未修改。Ruff通过。Git操作没有触发新的模型训练或生成。

本次共交付133份文件，包含M5源码、配置、测试、文档及换行约定；JSON、JSONL、TXT与Shell脚本默认固定LF，三份已冻结的历史JSON显式保留原始CRLF字节，避免Windows检出后破坏WSL入口或字节摘要。每个Git暂存blob与原文件SHA256逐一核对一致。具体提交和远端核验回执保存在原工作区`logs/m5_git_closeout_20261020_v1`，不提交Git。
