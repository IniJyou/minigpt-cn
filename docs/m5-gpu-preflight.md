# M5真实GPU预检结果

2026-10-18，冻结入口首次执行`preflight`并通过。现有RTX4060 Laptop可以承载本协议最长训练样本的前后向；这还不是完整训练或标注质量通过。下一任务为同一配置的单次180步训练及valid评估，test继续封存。

本轮只运行最长train完整序列`m5-sft-v1-real-r05`，共1961 tokens，其中53个target/EOS监督tokens。输入未截断；2次前后向、0次优化步，没有模型生成或valid/test评分。

| 检查 | 实测结果 |
| --- | --- |
| 运行环境 | 独立WSL M5环境；Python3.12.3、Torch2.8.0+cu126、Transformers4.57.6、PEFT0.17.1 |
| 模型 | 本地Qwen/Qwen3-0.6B，revision c1899de289a04d12100db370d81485cdf75e47ca |
| LoRA注入 | 28层×7个线性层，共196处 |
| 可训练参数 | 5,046,272，全部为FP32 LoRA A/B；基座冻结 |
| 前后向 | 2次，每次loss均为0.43106546998023987；梯度有限且存在非零梯度 |
| 初始化完整性 | B初始为零；预检结束adapter摘要与初始化一致，梯度清空、RNG恢复 |
| 峰值allocated | 4,977,030,144 bytes，约4.64 GiB |
| 峰值reserved | 5,731,516,416 bytes，约5.34 GiB |
| reserved门槛 | ≤7,511,474,176 bytes，即GPU总量8,585,216,000 bytes减1 GiB；通过 |
| 显存余量 | GPU总量减本进程峰值reserved约2.66 GiB；不是系统实时空闲显存读数 |
| 进程/产物 | exit 0；成功回执绑定输出摘要，无失败记录或训练/test目录 |

两次loss相同不代表学习效果：预检没有更新参数，且B为零时初始LoRA输出为零。完整训练还会创建AdamW状态并累积4条样本的梯度，本轮没有验证完整训练的实际峰值或180步稳定性；冻结的OOM、非有限和墙钟停止条件继续生效。

本轮实际调用：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_lora_wsl.sh --phase preflight
```

调用使用`/home/luoyu/.venvs/minigpt-cn-m5`，完全离线读取D盘已有模型；没有下载、安装或修改旧环境。执行清单仍为[原runtime](../configs/m5_lora_v1/runtime.json)，SHA256为`e87b1e10dca896c7d5c9bd60eb019c04314e98e266993b422ef4f718fe7a74e4`，72份绑定、源码、协议和数据均不改。原协议/入口文档中的“尚未预检”及布尔值是冻结时状态，当前执行结果以本页和回执为准。

原始结果在[GPU预检JSON](../logs/m5_lora_experiment_20261018_v1/preflight/gpu_preflight.json)，成功回执在[preflight/receipt.json](../logs/m5_lora_experiment_20261018_v1/preflight/receipt.json)，执行清单副本在[execution.json](../logs/m5_lora_experiment_20261018_v1/execution.json)。回执已封存的目录不再追加文件。

独立只读验收与新冻结回执位于`logs/m5_gpu_preflight_acceptance_20261018_v1`；验收核对成功文件树、原runtime绑定、既有冻结产物、最长train布局与显存门槛，不重新构造模型或补跑前后向。本轮仅结果/文档核对、Ruff和差异检查，不重跑CPU pytest；前轮84项定向回归及独立环境4项集成测试仍作为历史入口验证。

后续`--phase train-valid`须先验证本次预检回执，再进行新版提示的15条基座valid、固定180步训练及step45/90/180各15条valid。完整训练及所有接纳线通过后才允许单独进入test；本轮没有开始这些阶段。唯一下一任务仍以[当前工作](current-work.md)为准，Note2Lecture继续暂停。
