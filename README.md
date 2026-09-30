# MiniGPT-CN

从零实现约 33M 参数的中文 Decoder-only Transformer，完成 BPE Tokenizer、Causal
Self-Attention、RoPE、RMSNorm、SwiGLU、训练循环和 KV Cache；随后基于支持中文的
0.6B 基础模型进行 LoRA 指令微调，构建多轮对话服务并评估微调效果。

> 当前状态：M0 仓库骨架和配置系统已经就绪。Transformer、正式预训练、LoRA 微调和
> 对话服务均尚未实现。下文描述的是项目计划，不代表已经完成的成果。

## English summary

MiniGPT-CN plans to implement a ~33M Chinese Decoder-only Transformer from scratch,
then fine-tune a separate 0.6B base model with LoRA for multi-turn chat. The project
focuses on readable implementations, reproducible training, and controlled
evaluation of instruction following, repetition, and inference performance.
Currently, only the repository scaffold and configuration system are implemented.

## 项目目标

- 从零理解并实现现代 Decoder-only Transformer 的主要组件。
- 在本地完成 smoke test 和小模型 pilot，在云端训练约 33M 参数的正式模型。
- 保存配置、数据来源、训练曲线、消融结果和成本，提供可复现的生成 CLI。
- 在同一个 0.6B 基座模型上实施 LoRA-SFT，提供支持会话历史和流式输出的对话服务。
- 比较 0.6B 基座微调前后的指令遵循率、重复率和推理性能，记录失败案例和能力边界。

## 两阶段技术路线

| 阶段 | 模型与方法 | 主要交付物 |
| --- | --- | --- |
| 从零预训练 | 自行实现约 33M Transformer 和 16K byte-level BPE | 组件测试、训练系统、checkpoint、生成 CLI、消融实验 |
| 指令微调与服务 | 独立的 0.6B 基座模型 + LoRA-SFT | adapter、多轮对话服务、微调前后评测和模型卡 |

两个阶段使用各自的模型权重和 tokenizer。33M 模型不会通过 LoRA 变成 0.6B 模型，
自制的 BPE 也不能直接替换 0.6B 基座的 tokenizer。33M 阶段用于展示原理实现和预训练；
0.6B 阶段用于展示指令微调、评测和部署。

## 计划中的正式模型

| 配置项 | 数值 |
| --- | ---: |
| Vocabulary | 16,000 |
| Context length | 512 |
| Layers | 8 |
| Attention heads | 8 |
| Hidden size | 512 |
| SwiGLU hidden size | 1,344 |
| Parameter target | 约 33M |

使用 RMSNorm、RoPE、SwiGLU，并共享输入 embedding 与输出 projection 的权重。
完整配置见 [configs/minigpt_33m.yaml](configs/minigpt_33m.yaml)。

首轮计划训练 1 亿 tokens；验证集仍持续改善且预算允许时，再扩展到 2 亿 tokens。
正式训练前必须通过组件测试、小 batch 过拟合、本地 pilot 和 checkpoint 恢复演练。
33M 模型定位为中文文本生成实验，不承诺可靠的指令遵循或聊天能力。

## 0.6B LoRA 指令微调计划

候选基座为 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)，
它是支持中文的多语言预训练模型，模型卡标注 Apache-2.0 许可。正式实验会固定模型
revision、tokenizer、chat template 和依赖版本，避免下载更新影响复现。

1. 准备来源和许可明确的中文指令数据，包含单轮和多轮会话。按完整会话划分
   train/validation/test，并进行去重，确保评测内容不进入训练集。
2. 使用基座自带的 tokenizer，统一训练与推理的 chat template；只对 assistant 回复
   计算训练 loss，并检查截断、结束 token 和多轮样本的标签是否正确。
3. 使用 [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)
   和 [TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)，冻结基座，
   先用少量样本跑通训练、保存和加载，再确定数据规模、LoRA rank 与训练预算。
4. 对话服务携带当前会话历史，按 token 预算管理上下文，支持流式输出，并提供简单
   的 Gradio 页面。该阶段使用基座推理框架的 KV Cache；33M 模型中的 KV Cache
   则由项目自行实现和测试。
5. 保存 adapter、微调配置、数据清单、评测结果和失败案例，不把训练完成等同于效果改善。

微调所需的 PEFT、TRL、Transformers 和演示依赖将在后续阶段配置；当前安装命令只覆盖
已有骨架的运行和开发依赖。

## 评测与消融设计

核心对照是同一个 0.6B 基座的 **Base vs LoRA-SFT**：固定测试集、tokenizer、
chat template、解码参数、硬件和精度。33M 模型可作为独立参考，但参数量、语料和
tokenizer 的差异意味着不能把它与 0.6B 的分数差距解释为 SFT 的收益。

| 实验 | 记录内容 | 对照条件 |
| --- | --- | --- |
| 指令遵循 | 冻结约 100～200 条测试指令，覆盖格式约束、信息提取和多轮上下文；报告通过率与失败案例 | 可自动判断的使用规则，其余按固定标准人工复核 |
| 重复率 | 重复 4-gram 比例、输出长度、生成样例 | 相同 prompts、解码参数和最大输出长度，预先固定指标定义 |
| 推理性能 | 首 token 延迟、后续生成 tokens/s、显存峰值 | 相同硬件、精度、batch 和输入/输出长度；预热后多次测量 |
| 33M tokenizer 消融 | Character vs 16K BPE 的 BPB、bytes/token 和处理速度 | 相同文档划分与训练预算，不直接比较不同 tokenizer 的 perplexity |
| 33M 位置编码消融 | Learned positional embedding vs RoPE 的验证 loss 和生成样例 | 相同数据、seed 和训练预算 |
| 33M KV Cache 消融 | cached/uncached greedy 输出一致性、生成速度和显存 | 同一 checkpoint、prompt 和生成长度 |

LoRA 的主要目标是改善指令遵循，吞吐量是否变化需要实测。性能实验应分别注明
adapter 挂载和合并权重的方式；验证 loss、训练 loss 或少数展示样例不能替代独立评测。

## 里程碑与当前进度

| Milestone | 范围 | 当前状态 |
| --- | --- | --- |
| M0 Repository Bootstrap | 包骨架、配置、测试、CPU CI 和文档 | 已完成基础搭建 |
| M1 Transformer Core | Attention、RMSNorm、RoPE、SwiGLU、完整模型与组件测试 | 待实现 |
| M2 Tokenizer and Training | BPE、数据管线、训练、checkpoint 和恢复 | 待实现 |
| M3 33M Pretraining | 本地验证、云端预训练、训练曲线和消融 | 待实现 |
| M4 Portfolio Release | 33M 生成工具、演示、模型卡、可复现发布 | 待实现 |
| M5 LoRA Chat and Evaluation | 0.6B 指令微调、多轮服务与受控评测 | 新增计划，GitHub milestone 待创建 |

先完成 M1～M4，再推进 M5。原十周安排对应 33M 主线，新增微调与服务需要额外时间，
按验收结果推进，不以固定日期代替测试和实验。

## 本地环境

- Python 3.10+
- Windows / Linux
- 本地开发设备：RTX 4060 Laptop 8GB、约 31GB 内存
- 正式训练计划：AutoDL 单张 RTX 4090 24GB

本地用于开发、33M 小规模验证以及 0.6B LoRA 的显存试跑。0.6B 的 BF16 权重本身
约为 1.2 GB，训练还需激活和框架开销；8GB 显存下的序列长度、micro batch 与
gradient checkpointing 设置必须实测。安装 PyTorch 时应根据用途选择 CPU 或 CUDA 构建，
不能仅凭依赖安装成功就认定 GPU 环境已经可用。

原计划的 50～80 元、最高 100 元预算仅针对 33M 主线，不作为扩展项目的总费用承诺。
各阶段先试跑，再根据实测吞吐量和租机价格估算：

```text
预训练小时数 = 目标 tokens / 实测训练 tokens/s / 3600
租机成本 = 实际运行小时数 × 实际单价
总成本另计数据处理、验证、微调、推理评测和存储等费用
```

费用与吞吐量均在实验记录中填写实际数字。

## 开发环境

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
```

Linux/macOS 下将虚拟环境解释器路径替换为 `.venv/bin/python`。

## 仓库结构

```text
configs/                模型和训练配置
data/                   数据说明（原始数据不会提交到 Git）
docs/                   学习路线与实验记录
src/minigpt_cn/         Python 源码
tests/                  CPU 单元测试
.github/workflows/      GitHub Actions CI
```

## 学习顺序

1. PyTorch 张量、autograd、`nn.Module` 与训练循环。
2. 字符级 bigram 语言模型。
3. Causal Self-Attention 与 Multi-Head Attention。
4. RMSNorm、RoPE、SwiGLU 和完整 Transformer Block。
5. BPE Tokenizer、数据管线、训练和生成。
6. KV Cache、生成策略和推理性能测量。
7. 指令数据、LoRA、SFT、chat template 与多轮对话评测。

33M 主线的原十周安排见 [docs/learning-roadmap.md](docs/learning-roadmap.md)，
实验记录见 [docs/experiment-log.md](docs/experiment-log.md)。LoRA 扩展以本 README 的
阶段计划为准；学习笔记的进展不代表项目实现已完成。

## 数据与发布原则

- 训练数据、checkpoint、日志和密钥永不提交到 Git。
- 正式数据必须在 `data/README.md` 中记录来源、日期、许可和校验和。
- 最终发布 tokenizer、配置、推理权重和校验和，不发布原始语料或优化器状态。
- LoRA 阶段记录基座 revision、数据许可和 adapter 配置；第三方模型、数据和
  adapter 的使用与发布遵循各自许可，仓库的 MIT 许可不替代这些要求。

## 已知限制

- 当前仓库尚未实现模型和训练功能。
- 33M 预训练模型用于中文文本生成实验；0.6B LoRA 阶段的对话能力须通过独立评测验证。
- 两个模型都可能产生重复、幻觉、不完整句子以及训练数据中存在的偏差。
- 多轮服务使用会话上下文，不提供长期记忆；微调不保证事实准确性或复杂推理能力。

## License

MIT
