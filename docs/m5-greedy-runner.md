# M5 greedy诊断入口

2026-10-19。入口执行[冻结的零训练诊断定义](m5-valid-error-review.md)及[机器协议](../configs/m5_greedy_diagnostic_v1/protocol.json)。旧三个检查点未接纳的结论及旧test门槛不变。

运行器为[src/minigpt_cn/m5_greedy_run.py](../src/minigpt_cn/m5_greedy_run.py)，[WSL脚本](../scripts/m5_greedy_wsl.sh)使用已有独立M5环境。新runtime另行绑定本入口、CPU测试、本文和原96份定义产物；不修改冻结旧源码。默认inspect仅核对父级回执、固定step45文件树、环境、生成API与15条valid输入预算，不加载权重或创建实验目录。

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_greedy_wsl.sh --phase inspect
bash /mnt/d/aiworkspace/transformer/scripts/m5_greedy_wsl.sh --phase run
```

run独占创建`logs/m5_greedy_diagnostic_20261019_v1`，顺序执行基座15次、固定step45十五次。所有valid输入在权重加载前完成token预算检查；两个条件的输入ID再次比较。基座和adapter均为eval且全部参数冻结，生成使用inference_mode；无训练方法、优化器或test选项。每个条件独立加载/释放模型。

OOM保存失败记录后停止，非OOM逐条异常按空回复评分，失败保留全分母、不修复或重试。既有目录拒绝覆盖，不提供参数改模型、检查点或解码。原始回复、新tokens、逐条输入长度、错误、实际环境/配置/显存及strict metrics全部保存。

新greedy adapter仅与新greedy基座计算原九项接纳线。另将各条件与原同权重采样回复按sample_id配对，报告每条严格完全匹配块数的增减，整条schema/覆盖失败不取部分正确。结果始终标为diagnostic_only，即使接纳线通过也不自动启用test、部署或改写旧selection。

结束前再次核对父级/新执行清单及全部文件摘要，再封存成功回执；失败不写成功回执。回执完成表示此次有界诊断和文件树完整，不等同M5质量通过。实际运行、检查结果与下一任务以[current-work.md](current-work.md)为准。
