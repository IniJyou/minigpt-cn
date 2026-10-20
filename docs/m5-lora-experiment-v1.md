# M5 最小 LoRA 实验协议 v1

日期：2026-10-18。本轮只交付实验定义、依赖盘点和用户安装命令。机器协议为 [protocol.json](../configs/m5_lora_v1/protocol.json)，下一任务仍以 [current-work.md](current-work.md) 为准。尚未安装 PEFT、加载模型、运行 LoRA 注入或训练。

## 数据与运行范围

绑定 [v1.2 数据候选](m5-data-v1.2.md) 的 [评测协议 v3](../configs/m5_eval_candidate_v3.json)、完整提示及 train36条95块、valid15条62块、test9条36块。候选最低覆盖已通过，仍由助手标注，薄弱类别各仅1块；不借本实验扩样、改标签或改质量门槛。

基座继续为本地 Qwen/Qwen3-0.6B，revision `c1899de289a04d12100db370d81485cdf75e47ca`，使用原版 tokenizer/模板及 `enable_thinking=False`。模型、数据、旧规范与旧运行器保持冻结。未来运行器须绑定本协议与自身源码/测试摘要，通过 CPU 边界检查和 GPU 预检后再运行；本文件本身不构成安装或训练的执行授权。

## 本轮只读盘点

WSL 独立环境为 `/home/luoyu/.venvs/minigpt-cn-m5`。实际 Python3.12.3、Torch2.8.0+cu126、Transformers4.57.6、Accelerate1.10.1，与旧基座完整包清单一致；`uv pip check` 通过。GPU 为 RTX4060 Laptop，CUDA/native BF16 查询通过，本轮 CUDA 张量分配为0。

当前方案只缺 `peft==0.17.1`。其10项基本依赖均已满足；版本约束核对依据 [官方发布依赖](https://github.com/huggingface/peft/blob/v0.17.1/setup.py)，wheel 摘要取自 [PyPI 对应版本](https://pypi.org/project/peft/0.17.1/)。还检查了该版本引用的5个 Transformers 顶层符号均存在。以上不能替代安装后的实际 PEFT 导入、LoRA 注入、前向和显存验收。

旧 `QwenBackend` 对完整依赖清单做等值比较，安装 PEFT 后会拒绝旧协议。新协议另存“原清单 + peft0.17.1”的预期环境；实际安装后必须逐项验收，不改旧协议消除这个限制。新训练入口可复用原序列化/严格评分函数，不能拿旧校准 CLI 直接跑新版数据。

## 固定训练方案

| 项目 | 本次单一配置 |
| --- | --- |
| LoRA | rank8、alpha16、dropout0.05、bias=none、默认零 B 初始化 |
| 注入位置 | 28层的 q/k/v/o_proj、gate/up/down_proj，共196个线性层 |
| 可训练参数 | 形状核算5,046,272；基座、embedding、lm_head和norm冻结 |
| 计算精度 | 基座BF16、adapter参数/梯度FP32，CUDA BF16 autocast；无GradScaler |
| 内存与前向 | cuda:0、SDPA、non-reentrant gradient checkpointing；训练use_cache=False、logits_to_keep=0 |
| 批次 | microbatch1、梯度累积4，有效batch4；不丢样本、不packing、不截断 |
| 数据顺序 | 每epoch用独立 `random.Random(42 + epoch_index)` 打乱36条，epoch_index从0起 |
| 优化器 | AdamW，lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、foreach=False、fused=False |
| 学习率与梯度 | 10步warmup后线性衰减；采用同版Transformers scheduler；梯度范数上限1.0 |
| 随机种子 | Python/NumPy/Torch/CUDA均42；TF32关闭、cudnn_benchmark=False；不承诺跨平台逐位复现 |
| 训练边界 | 单一配置、单个种子、20 epochs、每epoch9个优化步，共180步/720次训练microbatch |
| 保存点 | epoch5/10/20，即step45/90/180；保存adapter、optimizer/scheduler/RNG与计数状态 |

上述参数是本次有界实验的工程选择，没有声称最优。按36条每轮2446个目标含EOS token计，20轮监督位置为48,920次曝光；不是新增48,920个独立语料token。adapter数量由本地safetensors头计算，尚未通过真实PEFT实例确认。[PEFT的LoRA参数与初始化说明](https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora)支持这些选项；本项目具体取值由本协议固定。

对于原权重 `[d_out,d_in]`，LoRA A/B形状分别为 `[8,d_in]` / `[d_out,8]`，新增参数为 `8*(d_in+d_out)`。训练输入/labels为 `[1,L]`，完整 logits 为 `[1,L,151936]`；生成时才使用 `logits_to_keep=1`。这是 [Qwen3前向接口](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3) 中全序列训练与末位生成的区别，不能沿用旧推理参数只训练最后一位。

## 监督、预检与停止

沿用原 SFT 接口：模板前缀 + 紧凑 target JSON + 一个EOS151645；只监督 target/EOS，前缀及右侧pad151643的labels均为-100，labels不预移位，由CausalLM内部移位一次。

每4条累积组先计算总监督token数N；每条前向传 `num_items_in_batch=N`，所得loss为该条目标交叉熵和除以N，逐条backward，组末clip/optimizer.step/scheduler.step，再清梯度。不再除以4，避免变成逐样本均值或重复归一化。已用极小CPU logits验证安装版 `ForCausalLMLoss` 的移位与分母接口；[同版源码](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/loss/loss_utils.py)可追查。真实PEFT转发该参数仍须未来运行器验证。

未来GPU预检只取训练集中最长的完整序列，至多2次前向/反向，0次优化步；保存并恢复训练RNG，清空梯度，再从同一初始adapter开始正式训练。核对196个注入位置、5,046,272个trainable参数、冻结基座、loss/梯度有限，并实测峰值allocated/reserved。reserved须不超过GPU总量减1GiB；这是预设接纳线，本轮没有训练显存结论。

OOM、非有限loss/梯度、来源/摘要/环境/监督掩码不符、输出目录已存在、达到训练墙钟1800秒或预检不通过时停止并保留诊断。墙钟在首个训练microbatch开始，于microbatch边界检查；阻塞中的CUDA操作不能硬保证在1800秒内终止。不自动缩短输入、量化、换rank/学习率、重试、追加轮次或跨环境恢复；未完成180步视为本次运行不完整，不启动test。

## valid 选择与 test 一次使用

先用新版提示对15条valid运行一次未适配基座及冻结规则。旧基座0/54属于旧提示和校准池，不用来计算新版增益。训练期间只保存3个预定检查点；全部训练结束后，再各自完整评估valid一次，避免评测随机数改变训练轨迹。模型单独加载，每条seed42、BF16/SDPA及原生成设置不变，严格原样评分，不修复或重试输出。

各adapter须同时通过原接纳线：JSON/schema/覆盖率100%、块全部字段≥0.80、整条样本≥0.50、固定八类macro-F1≥0.75且较同提示基座提升≥0.15、非空title与reason精确率各≥0.80。全部候选均不通过则停止，不运行test。符合者按macro-F1、块全部字段、样本完全匹配率依次从高到低选择，完全相同时取较早step；使用未四舍五入的计数/分数，不能人工挑“看起来好”的样本。

选定后先锁定协议、实际环境、运行器源码和adapter文件摘要，再对9条test运行一次基座、选中adapter与冻结规则；不评测其他检查点，不根据test改标签/提示或追加训练。test独立应用相同全部接纳线及同提示test基座增益；失败时报告失败，不回头换checkpoint。

模型生成上限为valid60次（基座15+3×15）及有条件test18次（基座9+adapter9），总计78次；规则输出另外计数。原始响应/token、全量metrics/错误、训练loss/学习率/监督token/耗时/峰值显存、各checkpoint摘要和最终选择回执全部保留。冻结规则不新增Examples补丁，其对新版标记的限制如实记录。训练loss仅用于监测，不替代生成质量或选择检查点。

## 用户执行的依赖准备

仅新增PEFT wheel，使用 [带摘要的增量要求文件](../configs/m5_lora_v1/requirements.addon.txt)。下面是WSL命令，**助手本轮未执行安装**：

```bash
UV_CACHE_DIR=/mnt/d/aiworkspace/transformer/data/raw/m5_uv_cache \
  /home/luoyu/.local/bin/uv pip install \
  --python /home/luoyu/.venvs/minigpt-cn-m5/bin/python \
  --index-url https://pypi.org/simple \
  --no-deps --only-binary :all: --require-hashes \
  -r /mnt/d/aiworkspace/transformer/configs/m5_lora_v1/requirements.addon.txt

/home/luoyu/.local/bin/uv pip check \
  --python /home/luoyu/.venvs/minigpt-cn-m5/bin/python
```

该文件不是整个环境的wheel锁；已有环境仍以新协议完整版本清单验收。缓存继续在D盘，不重建环境或改变Torch/CUDA。命令未测试网络连通；若遇先前相同网络超时，按已有 [M5环境准备](m5-environment-setup.md) 的临时代理说明处理。

本轮证据在 `logs/m5_lora_plan_20261018_v1`，含只读盘点、CPU损失接口检查、官方发布元数据、依赖约束、形状计算与验收/冻结回执。当前 `training_ready=false`：PEFT安装验收、训练运行器及GPU预检仍待完成。下一项是用户完成依赖安装后，验收新环境并交付训练入口和CPU边界测试；不由这次文档直接启动训练。
