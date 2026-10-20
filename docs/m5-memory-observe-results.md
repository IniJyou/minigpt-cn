# M5 有界GPU观测：确认reserved越过余量门槛

2026-10-19。使用[冻结入口](m5-memory-observe-runner.md)单次执行probe，**在原位置复现资源停止，并保存了触发读数**：step52首条n08反向后，reserved峰值8,382,316,544 bytes（7.806640625 GiB），超过7,511,474,176 bytes（6.995605469 GiB）上限。设备总量和allocated/reserved峰值关系检查均通过，唯一命中条件是reserved余量不足。

完成6次新更新、25次前后向、69次资源采样，随后按门槛退出。未重试、调整设置或生成；没有模型检查点、有界成功回执或step105处理组。原60步续训与53次评测仍未完成，test封存、Note2Lecture暂停。

## 实际范围与直接触发条件

| 项目 | 现场记录 |
| --- | --- |
| runtime SHA | 521452966cc3ddcb8687ac673b526ed396b47c0ee9bf9ca0a5a3b2c9d494ee09 |
| 来源 | 原修订step45完整adapter/392份Adam/scheduler/RNG/游标，冻结身份核对通过 |
| 新优化步 | 6次：绝对step46–51，六次optimizer和scheduler均有返回事件 |
| 前后向 | 25次，含n08反向返回；账本为绝对microbatch181–205 |
| 资源检查 | 前68次通过，第69次失败；264条完整连续观测事件 |
| 停止位置 | event264，after_backward_delete，step52、microbatch205、m5-sft-v1-real-n08 |
| 布局及loss分母 | 输入1630、目标含EOS161、序列1791，整组监督分母417 |
| 更新边界 | step52 optimizer未进入，最后完成更新为51；本次未保存step51权重 |
| 已更新监督token曝光 | 新增1108，累计9657→10765；未更新的n08不计入该累计 |
| 时间 | 失败guard记录elapsed约42.032秒，未触发1800秒门槛；不是整个进程墙钟或费用 |
| 生成/额外预检/test | 0 / 0 / 0 |

本次停止仍是项目MemoryError，终端没有CUDA OutOfMemoryError。分配/反向已返回，项目在保留1GiB余量的门槛处停止；不能把它写成CUDA分配OOM或显存读数无效。

| 最后两次采样 | allocated峰值 | reserved峰值 | 设备总量 |
| --- | ---: | ---: | ---: |
| event258，n08 collate前，通过 | 4,435,415,040 | 5,110,759,424 | 8,585,216,000 |
| event264，n08反向并删除loss/batch后，失败 | 4,697,260,544 | 8,382,316,544 | 8,585,216,000 |

单位均为bytes。失败reserved比上限高870,842,368 bytes（约0.811 GiB）；跨这段计算，reserved峰值增加3,271,557,120 bytes，allocated峰值增加261,845,504 bytes。失败allocated峰值约4.375 GiB，设备总量约7.996 GiB。

三项判定完整保存为：

```json
{"device_total_changed": false,
 "peak_relation_invalid": false,
 "reserved_headroom_exceeded": true}
```

synchronize和全部读取均返回，partial_observed与observed完全一致，last_valid_event_id=258。它们是本轮同一采样用于判定的原值，未在失败后补查或重读。

## 能下什么结论

本次已把[原失败](m5-train-fit-results.md)中未区分的三项条件，定位为**n08计算后reserved峰值越过余量上限**。这是资源问题的直接触发机制；门槛和错误留证均按定义执行，966项现场核对通过，712份前项产物原字节不变。现场验收通过不表示probe成功完成，bounded_probe_completed=false、training_complete=false。

reserved和allocated为自基座加载前reset后累计的不同峰值，不是两份同一时刻活跃张量快照。差额不能直接当作可立即回收量，也不能仅凭差额证明碎片、泄漏、驱动异常或模型活跃张量必须占用7.807 GiB。没有保存allocator分块/当前占用或继承配置，当前资料不足以确认底层原因或原进程的具体allocator设置。

两次GPU运行的六步loss/norm也不是逐值一致：例如旧step47 loss=0.023901328561，本次0.024214774719；完整差异保留在observations_summary.json。复现的是冻结起点身份、样本/lr/监督分母和失败位置及条件，不宣称两次GPU权重轨迹逐位相同。先前CPU精确对照不能扩展为跨进程GPU逐位保证，差异的具体原因本轮未判定。

## 后续修复方向与下一验收点

下一项把**独立进程的单项allocator配置候选入口及CPU/默认inspect验收**合并交付。优先核对本地固定Torch2.8实际配置接口，候选仅为进程启动前的`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，保留数据、模型、精度、数学计算和1GiB余量门槛。

这是有依据的候选，尚非已验证修复：PyTorch2.8将expandable_segments列为实验选项，目的是改善分配大小频繁变化时的内存利用；是否适用于本次场景仍需后续对照验证。[PyTorch2.8内存管理说明](https://docs.pytorch.org/docs/2.8/notes/cuda.html#memory-management)

新入口须记录实际继承环境、最终配置及生效/不支持状态，不能假设本次原配置为空或后端一定是native；也不能把设置字符串存在当作已在平台上生效。Torch2.8源码包含平台不支持时的警告与关闭分支，应明确报错停止，不静默退回另一配置。[v2.8.0配置实现](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/cuda/CUDAAllocatorConfig.h)

本轮未修改任何allocator设置，也未执行此候选。下一轮另存源码/配置/启动脚本与目录，默认inspect依旧不加载真实权重或初始化CUDA；合并完成CPU环境隔离、配置冲突、旧产物拒绝覆盖及预算边界验收。未来GPU对照预算仍至多6更新/25前后向/69采样，0生成/test/额外预检，不在入口交付时自动执行，不接续60步训练或挑选检查点。

## 证据位置

实际失败目录logs/m5_memory_observe_experiment_20261019_v1仅有started.json、input_inspect.json、observations.jsonl、training_attempts.jsonl、training.jsonl、failure.json六份文件；无receipt/result/checkpoint。failure.json含完整堆栈，observations.jsonl保存全部原值和更新边界。

只读验收logs/m5_memory_observe_probe_20261019_v1含acceptance.json、observations_summary.json与terminal_capture.json；未追加任何模型调用、下载、安装或Git写操作。旧失败现场、原step45和所有旧入口/协议保持不变。唯一下一任务以[current-work.md](current-work.md)为准。
