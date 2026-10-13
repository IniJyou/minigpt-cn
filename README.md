# MiniGPT-CN

从零实现约33M参数的中文 Decoder-only Transformer，完成训练、生成和评测；
33M主线完成后，再推进独立0.6B基座的 LoRA-SFT 与多轮服务。

**继续任务先读 [当前工作与交接](docs/current-work.md)。这是唯一的下一任务入口。**
参考文档说明接口，实验日志保存历史结论，不把历史“下一步”当成当前计划。

## 当前状态

- 已验收：模型核心、权重共享、dropout、BPE编解码/增量训练、训练与完整评估、
  配置化checkpoint/恢复、生成CLI和KV Cache。
- 已具备：正文清洗/去重/划分、仅train文本导出，以及可复用的Wikimedia数据工具。
- 已验收：16000词表tokenizer保存/重载、同版本pilot三集合uint16编码与全部1000篇磁盘读回。
- 已验收：33M本地100步短训练、50步保存/恢复与连续训练状态完全一致、独立重载完整验证。
- 已验收：AutoDL/RTX4080 SUPER的500步资源验证与本地原始产物；已记录累计费用3.09元及用户保留实例运行的决定。
- 已准备：50元总预算下的正式语料与首轮预训练方案，具体正式输入仍待验收。
- 尚未完成：正式语料准备、正式预训练与展示。
- 数据工具本轮收尾并冻结扩建；不再追求完整维基渲染、全站扫描或通用审批平台。
  未审批的候选不是训练语料，旧有损正文不会因升级源码自动修好。

## 主线顺序

1. 复用已验收pilot的train文本，保存并验收16000词表tokenizer。
2. 使用同一tokenizer编码固定train/valid/test，完成33M本地短训练和恢复检查。
3. 在实际AutoDL设备跑约500步，测显存、吞吐和稳定性，再确定费用与训练预算。
4. 准备预算明确的正式语料，正式预训练、生成、评测和展示。
5. 最后推进独立0.6B LoRA阶段。

pilot小语料只用于验证闭环，不能靠重复它凑出1亿token就宣称正式预训练效果。
正式首轮计划1亿tokens；只有验证仍改善、语料规模合适且预算允许时才扩到2亿。

## 正式模型配置

| 配置 | 值 |
| --- | ---: |
| vocabulary / context | 16000 / 512 |
| layers / heads | 8 / 8 |
| d_model / d_ff | 512 / 1344 |
| norm / position / activation | RMSNorm / RoPE / SwiGLU |
| tied embeddings / dropout / bias | true / 0.1 / false |

配置见 [minigpt_33m.yaml](configs/minigpt_33m.yaml)，参数约33M。
2048词表的旧pilot权重不能与新16K tokenizer混用。
33M模型定位为中文文本生成实验，不承诺可靠的聊天或指令遵循能力。

## 里程碑

| 阶段 | 状态 |
| --- | --- |
| M0 Bootstrap | 基础搭建已完成 |
| M1 Transformer Core | 实现与本地验收已完成 |
| M2 Tokenizer and Training | 核心工具、16K pilot数据与33M本地短训练/恢复闭环已验收 |
| M3 33M Pretraining | 云端资源验证已验收；50元正式准备方案已保存，正式语料/预训练待完成 |
| M4 Portfolio Release | 生成CLI已有；正式模型、评测与展示待完成 |
| M5 LoRA Chat and Evaluation | 独立0.6B基座与其自带tokenizer，33M主线完成后开展 |

GitHub状态、提交和合并由用户操作；此表描述本地实现，不代替远端验收。
位置编码、tokenizer和KV Cache消融属于后续评测，不阻挡首个33M闭环。

## 文档导航

- [当前工作与交接](docs/current-work.md)：范围、环境、下一任务与提交收尾。
- [Tokenizer操作](docs/tokenizer-guide.md)：已验收16K版本与保存/重载接口。
- [Token编码验收](docs/token-encoding-guide.md)：同版本三集合uint16产物、复现与独立读回。
- [33M短验证](docs/33m-short-validation.md)：本地100步、精确恢复、完整验证与资源统计范围。
- [AutoDL资源验证](docs/autodl-resource-guide.md)：500步云端实测结果、统计范围与产物归档收尾。
- [正式预训练方案](docs/formal-pretraining-plan.md)：50元预算、实测推算、正文来源候选与逐项验收范围。
- [模型与运行接口](docs/runtime-reference.md)：张量、checkpoint、训练/评估/生成入口。
- [数据工具参考](docs/data-reference.md)：已实现的工具与边界，按需查询，不是待办清单。
- [数据来源与产物状态](data/README.md)：哪些能复用，哪些不能训练。
- [阶段路线](docs/learning-roadmap.md)：主线验收顺序，不再按旧十周教程重做。
- [实验结论](docs/experiment-log.md)：真实历史结果和已知限制，不是当前操作指令。

## 环境与检查

本地设备为RTX 4060 Laptop 8GB、约31GB内存；本地用于开发和短验证，正式训练在云端。
已有WSL项目环境：/home/luoyu/.venvs/minigpt-cn；作业环境单独保留。
缺依赖时由用户安装，助手不自行下载或重建环境。

Windows已有环境检查：

~~~powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest -q --tb=short
~~~

WSL已有环境检查：

~~~bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q --tb=short
python -m ruff check .
~~~

## 数据与发布原则

原始语料、token分片、checkpoint、日志、密钥和缓存不提交Git。
保留来源、版本、许可、划分和校验和；公开权重/数据/adapter前核对适用许可。
代码的MIT许可不替代第三方语料或模型许可。项目学习笔记CS336_note.md独立维护，不修改。

## License

MIT
