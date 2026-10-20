# M5 greedy入口中断与调用边界修复

2026-10-19。用户授权的30次greedy对照**尚未完成**。v1启动后发现实际解码被Transformers改为采样，助手停止进程；保留6条完整基座回复，另可能有1次在途调用，按最多7次尝试计。adapter未运行，test未运行；这些回复不是greedy结果，不计入诊断评分。

## 发生原因与已有证据

v1在CPU检查和实际inspect中确认了`GenerationConfig.get_generation_mode() == greedy_search`，但没有检查`generate()`合并模型默认值后的有效配置。这是入口验证缺口。

实际Transformers4.57.6的`GenerationMixin._prepare_generation_config`在默认行为下，会把与库全局默认相同的自定义值回填为模型默认。`do_sample=False`等于库默认，本地Qwen的模型默认却是True，因此产生警告并改为采样。原API检查只检查配置对象，未覆盖这个调用阶段。

独立M5环境的CPU回归使用本地GenerationConfig与实际合并函数复现：v1最终为sample；修复后最终为greedy_search，并逐字段核对全部预定参数。没有构造模型、初始化CUDA或做真实前向。原Qwen/LoRA历史推理使用do_sample=True，不受此次False被覆盖问题影响，旧分数和检查点不改。

## 修复与运行边界

另存[src/minigpt_cn/m5_greedy_run_v2.py](../src/minigpt_cn/m5_greedy_run_v2.py)、[v2启动脚本](../scripts/m5_greedy_v2_wsl.sh)和新CPU测试，不覆盖v1已绑定源码。v2复用v1的固定valid/30次边界流程，在真实调用处显式传：

```python
model.generate(
    ...,
    generation_config=checked_config,
    do_sample=False,
    use_model_defaults=False,
)
```

模型加载后及每次调用前，都使用实际合并函数检查有效模式和全部冻结生成参数。配置错误直接全局停止，不能当作普通逐条生成异常吞掉。另对Transformers的generation/utils.py保存并核对摘要，补足原来只绑定configuration_utils.py的检查。

每次调用在前向前将attempt编号、条件及有效模式写入call_attempts.jsonl并flush/fsync，便于中断后准确核对已开始调用数。独占目录、原样回复/严格评分、OOM停止、非OOM异常全分母、无训练/test入口和固定step45规则不变。

v1原始现场在`logs/m5_greedy_diagnostic_20261019_v1`。中断时只有6条完整回复，未封存成功回执；v1未持久记录调用起点，因此无法精确确定中断时第7次是否已经进入generate，最多1次在途，不能伪造精确调用数。外部中断记录、原文件树摘要、CPU回归及修正版只读inspect保存在`logs/m5_greedy_acceptance_20261019_v1`。

## 调用预算与当前状态

原[冻结协议](../configs/m5_greedy_diagnostic_v1/protocol.json)规定`max_model_generation_calls=30`且`retry=false`。此前已消耗最多7次非greedy尝试；剩余预算无法容纳重新完整的基座15+adapter15。因此本轮只把修复做成可审阅、通过CPU及实际inspect的版本，**不自动重新生成**。

修正版定义另存在`configs/m5_greedy_diagnostic_v2`，有效实验参数仍只改变采样为greedy；新增调用边界强制措施、准确attempt日志和新独占输出目录`logs/m5_greedy_diagnostic_20261019_v2`。若用户明确同意追加最多30次，累计预算按最多37次计，再完整执行一次；修正版inspect不消耗真实模型调用。

当前test、所有旧数据/提示/评分及三个检查点继续封存，Note2Lecture暂停。下一任务及追加预算状态以[当前工作](current-work.md)为准。
