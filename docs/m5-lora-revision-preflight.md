# M5修订对照：独立GPU预检结果

2026-10-19，按current-work的唯一下一任务执行一次冻结入口的preflight，结果通过。用合并84条train中最长的完整序列做2次前向与反向，**0优化步、0生成**；尚未执行修订训练或质量对照。

## 实际检查

| 项目 | 结果 |
| --- | --- |
| 环境 | 独立WSL M5环境，43项包版本与冻结配置一致；未安装或下载 |
| 基座 | 本地Qwen3-0.6B，596,049,920参数，BF16 / cuda:0 / SDPA |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，设备总量8,585,216,000 bytes |
| 预检样本 | m5-sft-v1-real-r05；1908输入+53监督目标（含EOS）=1961 tokens，无截断 |
| LoRA | 196处注入，5,046,272个FP32可训练参数；基座冻结，初始化B全零 |
| 前后向 | 2次，均通过有限且非全零的梯度检查；每次53个监督目标 |
| 两次loss | 均为0.43106546998023987；没有参数更新，不能据此声称训练改善 |
| 峰值allocated | 4,977,030,144 bytes，约4.635 GiB |
| 峰值reserved | 5,731,516,416 bytes，约5.338 GiB |
| reserved上限 | 设备总量减1 GiB，即7,511,474,176 bytes，约6.996 GiB；通过 |
| 余量 | 设备总量减reserved峰值约2.658 GiB |

初始adapter SHA256为7716c1bddde7d596bf59dc7c8fb87d15f1553042c0e91ed1eb0917ea140aa3e8，与旧实验初始化一致，前后向后摘要不变。冻结代码在结束时清空梯度并恢复RNG；本次只读验收核对成功调用、源码绑定及输出回执，没有再次构造模型或重复前后向。

这是最长样本的前后向资源检查，未覆盖优化器状态、完整训练循环或生成的资源峰值，不能据此认定后续训练全程都在8GB内。旧180步实验的reserved统计异常及其未知成因仍保留。

## 回执与边界

- [冻结入口说明](m5-lora-revision-runner.md)和configs/m5_lora_revision_v1/runtime.json保持原样；runtime SHA256为06844ee745cf6d1803f1f2f6230d89f5c5563b51f52943163e8ad6a70b2bf230。
- 实验目录logs/m5_lora_revision_experiment_20261019_v1只含execution.json与已封存preflight；其中gpu_preflight.json、runtime.json、input_preflight.json、started.json均由receipt.json绑定。未创建train-compare、检查点或调用账本。
- 新验收在logs/m5_lora_revision_preflight_20261019_v1/acceptance.json：423项核对通过，397份前项冻结产物原字节未变，runtime的383项绑定一致。审计文件写在封存实验目录之外。
- 输入准备核对84条train、24条新开发检查及15条旧valid，共123条；GPU前后向只用上述一条train。test只参与文件摘要核对，未读取记录、tokenize或评测。

下一任务以[current-work](current-work.md)为准：使用同一冻结runtime执行一次train-compare，45优化步/180条次/9657监督token曝光，保存预定step45后执行24+24+24+15共87次开发对照。失败保留，不自动重试或增加预算；本轮未执行该阶段。Note2Lecture继续暂停。
