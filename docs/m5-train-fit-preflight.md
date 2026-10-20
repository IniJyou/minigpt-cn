# M5 同轨迹续行诊断：独立GPU预检

2026-10-19。按[冻结定义](m5-train-fit-plan.md)及[已验收入口](m5-train-fit-runner.md)完成独立GPU预检，**通过2次最长1961-token完整样本前后向，0优化步、0生成**。从修订step45恢复完整状态，未执行第46–105步或53次评测，test继续封存，Note2Lecture暂停。

| 项目 | 结果 |
| --- | --- |
| 固定runtime | configs/m5_train_fit_v1/runtime.json，SHA=4e563ea07b0811bcb5563b4e83f6b15a1b7b6d4b0cd88b121b5ae83fb1843c4f |
| 来源状态 | 修订84条训练的step45，第三轮已消费12条，Adam及scheduler均处于45步 |
| 真实GPU加载 | 固定Qwen3-0.6B、BF16基座，FP32 LoRA，原SDPA/checkpointing设置 |
| 参数身份 | 392个命名FP32张量、5,046,272参数，注册顺序与已绑定meta/CPU证据一致 |
| 预检样本 | m5-sft-v1-real-r05，完整序列1961 tokens |
| 前后向次数 | 恰好2次，均在前向前flush/fsync记账 |
| loss | 0.02277340553700924、0.022242678329348564 |
| 优化步/生成/test | 0 / 0 / 0 |
| 峰值allocated | 5,017,400,320 bytes，约4.673 GiB |
| 峰值reserved | 5,771,362,304 bytes，5.375 GiB |
| 设备总量/预检上限 | 8,585,216,000 / 7,511,474,176 bytes；上限为总量减1GiB，约6.996 GiB |

运行器先核对原初始adapter摘要、命名参数顺序和训练后权重摘要，再恢复392份Adam状态与scheduler。两次前后向的loss和梯度均有限且adapter有非零梯度；基座冻结。结束后清梯度，核对adapter、Adam、scheduler未改变，完整恢复Python/NumPy/Torch/CUDA RNG并核对摘要；全部断言通过才生成成功回执。

训练后adapter摘要保持0ccf14a95fa9db95c5d77af9d1437c3a27ecb8b3aae3808efdb2586b07c0f493。预检中的checkpoint_identity沿用原CPU身份描述；本次确实加载基座并初始化CUDA完成了前后向。

正式输出为logs/m5_train_fit_experiment_20261019_v1/preflight，receipt.json封存started.json、input_preflight.json、forward_backward_attempts.jsonl和gpu_preflight.json四份文件；实验根目录的execution.json与冻结runtime一致。train-evaluate目录、step105、训练和生成账本均未创建。

独立只读验收在logs/m5_train_fit_preflight_20261019_v1：646项核对通过，619份前项产物原字节不变，原检查点树/标签/提示/实现/运行绑定均未改。验收读取已封存报告和断言执行证据，不重新加载模型或追加GPU调用；未重跑前项CPU神经测试，无下载、安装、Git写操作或重试。

本结果只说明本次恢复路径和两个预检样本呈现满足资源门槛。历史完整训练出现过17.75GiB reserved，成因仍未核定；预检通过不证明完整60步续行始终符合8GB设备预算。后续训练继续逐microbatch及optimizer边界监测，越界或不可靠读数即停并保留现场。

唯一下一项以[current-work.md](current-work.md)为准：按冻结协议执行一次第46–105步续行，再完成固定53次评测与只读验收。两套开发与训练回放分别报告，不挑选新检查点，不触发test或追加调用。
