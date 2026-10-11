# 实验日志

每次实验都复制下面的模板。失败实验也要记录，因为失败原因是项目复盘的重要部分。

## 2026-10-09：中文 pilot 迁移一致性

- 分支：`m1/transformer-core`；本次工作未执行 Git 提交。
- 来源：作业 pilot 第5000步 checkpoint；项目配置 `configs/pilot_3m.yaml`。
- 参数：3,737,856；tokenizer：2048 BPE；完整验证：37,826 个目标。
- WSL PyTorch：2.11.0+cu130；GPU：RTX 4060 Laptop。
- CUDA/BF16：原实现与项目 loss 均为3.8759773192，PPL=48.2298111859。
- CPU/FP32：原实现与项目 loss 均为3.8759228132，PPL=48.2271824449。
- 两种模式均逐块检查完整 logits 与 loss，最大误差为0；共享 Parameter 重载后保持。
- 7篇验证文档重编码与原 uint16 文件逐 ID 相同；3条 prompt 的 greedy 输出逐 token 相同。
- 原始报告：`logs/pilot-migration-cuda-bf16.json`、`logs/pilot-migration-cpu-fp32.json`，不提交。
- 复现、SHA-256、接口和数学复习见 `docs/pilot-migration.md`。
- 结论：迁移行为一致；生成仍重复/不连贯，不证明正式33M资源或质量。
- 下一步：建立独立项目 WSL 环境，再接入配置 checkpoint、训练/评估/生成入口和 KV Cache。

## 2026-10-10：步骤2C 配置与 checkpoint

- 新增统一运行配置与版本1 checkpoint；本次未执行 Git 提交。
- 训练/数据/模型配置、实际优化器参数组、真实步数、RNG 和采样状态可保存重载。
- CPU/FP32 与 CUDA/BF16 恢复测试：下一批 ID、dropout 后 loss、下一步参数逐元素相同。
- 原作业 AdamW 未在本段迁移；测试用原生 AdamW 单独验收 checkpoint 机制。
- 旧 pilot 显式导入为推理格式，iteration=5000、共享 Parameter 保持，不伪造精确续训状态。
- 新格式完整验证：37,826 个目标，loss=3.8759773192，PPL=48.2298111859，logits/loss 差异为0。
- 原 checkpoint SHA-256 仍为 `505fafd02381b8a92dc359a6c7f0801db9f444d23a1e11c7f7123b7eacc61ce0`。
- 忽略的本地产物：`checkpoints/c_verification/pilot_3m_v1.pt`、`logs/checkpoint-c-verification.json`。
- 复跑命令见 `docs/step2-guide.md` C段；下一段接入训练与完整评估 CLI。

## 2026-10-10：步骤2D 训练与完整评估入口

- 接入原作业 AdamW、余弦调度、全局梯度裁剪和 get_batch，保留公式与中文注释。
- 原作业 AdamW 对照20次更新：参数和一/二阶矩逐元素完全一致。
- 训练参数/梯度/矩状态 FP32，CUDA BF16 autocast；完整验证不再只采一个 batch。
- Windows、独立 WSL：95 passed，Ruff通过；CPU-only CI 跳过2个CUDA测试。
- CPU/FP32 固定小 batch 60步过拟合：最终 loss 小于初始的10%。
- CPU/FP32、CUDA/BF16 的真实训练循环：含 dropout 时连续/恢复状态完全一致。
- 独立评估旧 pilot：全部37,826个目标，loss=3.8759773192044915，PPL=48.2298111859。
- 新中文短运行：原2048 tokenizer和数据，3,737,856参数、batch8/context256、100步。
  从随机权重开始，完整验证 loss：0步7.6813722416 -> 50步6.7380504432 -> 100步6.6604861100。
- 真实CUDA/BF16对照：连续100步 vs 50步保存+恢复到100步，权重/AdamW/RNG/采样状态完全相同；
  恢复后记录的训练loss和lr也相同。不是原5000步pilot的续训或长期质量实验。
- 新片段目录拒绝覆盖；恢复只允许改训练终点，语料哈希/PyTorch/其他配置必须相同。
- loss、梯度、参数、矩状态非有限时退出，无正常最终 checkpoint。
- 本地产物均忽略：`checkpoints/d_verification/`、`logs/d-pilot-evaluation.json`，
  完整对照摘要 `logs/d-training-verification.json`；原作业checkpoint校验和未变。
- 未实现生成CLI、KV Cache、正式配置测速、W&B与best checkpoint策略，未执行Git提交。

## 2026-10-11：步骤2E 中文生成 CLI

- 接入原generate.py逻辑，增加中文公式/形状注释、ModelOutput、greedy和独立采样Generator。
- checkpoint配置/tokenizer自动加载，单条、非流式、无KV Cache；BF16前向、FP32概率。
- 新增51个生成测试；覆盖原temperature/top-p/EOS/裁剪/no_grad以及CLI/UTF-8/seed/错误路径。
- 最终Windows/独立WSL：146 passed、Ruff通过；隐藏GPU后CPU测试143 passed、3 skipped。
- 三条prompt的CUDA/FP32：采样64token和greedy32token与原作业参考逐ID相同，原checkpoint哈希未变。
- 实际CLI读取原5000步pilot和项目100步checkpoint均可生成；不import作业包、不重训。
- 旧pilot greedy64token存在“他的数学家”等重复；100步模型greedy64token主要输出逗号。
- 旧pilot FP32采样128token（T=0.8、p=0.9、seed42）出现英文碎片和非法UTF-8，保留1个替换字符。
  明确记录这些失败样例，不把接口验收当作中文语义质量验收。
- 生成库推理阶段eval/no_grad，结束恢复原模式；NaN/+Inf/全屏蔽候选明确退出。
- 原始样例忽略目录 `logs/e_verification/`，生成迁移摘要 `logs/e-generation-verification.json`。
- 使用独立项目环境，未改作业源码/学习笔记/模型权重，未执行Git写操作。下一步F实现KV Cache。

## 2026-10-11：步骤2F KV Cache 与窗口重建

- 在现有模型上增加可选缓存路径，保留原作业/项目注释与公开forward接口，不改权重键名。
- 每层保存已做RoPE的K与未旋转的V；新位置offset=历史长度，mask为[new,past+new]。
- 缓存只用于eval推理，由调用者持有；不进入checkpoint，不跨请求复用、不原地修改旧缓存。
- 生成默认仍无缓存；`--use-cache` 开启prefill/单token decode，满窗口重建并重置位置。
- Windows/独立WSL有CUDA：181 passed，Ruff通过；隐藏GPU后CPU：176 passed、5 skipped。
  本段新增35个用例（33个缓存/基准用例与2个生成入口回归用例）。
- 真实旧pilot与用户项目100步checkpoint：三条prompt，FP32/BF16，各新增64token，
  共12组cached/uncached greedy逐ID一致。两份缓存CLI报告也与用户E阶段报告逐ID一致。
- 旧pilot超长prompt300token/窗口256，在FP32/BF16下各新增8token，重建结果逐ID一致。
  满窗口基准记录incremental_calls=0、window_rebuilds=7，持久KV=1,048,576 bytes。
  此时两种模式都计算整个窗口，没有算法上的增量加速，短测量波动不能解释为持续缓存收益。
- 无缓存完整迁移回归：37,826目标，loss=3.8759773192044915、PPL=48.22981118588135，
  与作业完整logits/loss差异仍为0，共享参数保持，原作业checkpoint校验和未变。
- 独立短prompt基准：RTX4060 Laptop、PyTorch2.11.0+cu130、BF16、prompt3token、输出64token，
  每种模式预热2次/测5次，计时仅覆盖greedy生成、禁用EOS，不含加载/文本编解码。
  无缓存/缓存总时间中位数0.641453/0.596017秒，约1.076倍；后续token约99.53/107.42 tokens/s，
  首token约9.09/9.01ms，CUDA allocated峰值31,783,936/31,625,216 bytes；持久KV270,336 bytes。
  笔记本未锁频，原始trials全部保留；小pilot提升有限，不代表33M吞吐、长期速度或训练预算。
- 单条非流式生成仍会重复/不连贯；100步模型仍主要输出逗号，缓存不改变语言质量。
- 报告均在忽略目录 `logs/f_verification/`：`pilot_bf16_isolated.json`、
  `pilot_bf16_full_window.json`、`uncached_regression.json`、两份cached CLI报告。
  `pilot_bf16_short.json` 是与测试并行的早期smoke，不用于上述独立基准结论。
- 用户逐步命令见 `docs/step2-guide.md` F段。未执行Git写操作，未改作业源码/学习笔记/原权重。
- 下一步由用户复核F，再讨论正式16K tokenizer及33M/512的资源与稳定性试跑；未开始正式训练。

## 实验模板

### 基本信息

- 实验名称：
- 日期：
- Git commit：
- 运行设备：
- 配置文件：
- 随机种子：

### 数据

- 数据版本与校验和：
- 训练 / 验证 token 数：
- tokenizer 版本与校验和：

### 训练

- 模型参数量：
- precision：
- batch size / gradient accumulation：
- 学习率与 scheduler：
- tokens/s：
- 运行时长与费用：

### 结果

- 最佳 validation loss：
- BPB / 其他指标：
- 生成样例位置：

### 观察与下一步

- 哪些设置有效：
- 哪些设置失败：
- 下一次只改变哪个变量：
