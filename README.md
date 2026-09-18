# MiniGPT-CN

从零实现一个面向中文文本生成的 Decoder-only Transformer。本项目强调“理解并亲手实现”，
计划覆盖 BPE Tokenizer、Causal Self-Attention、RoPE、RMSNorm、SwiGLU、训练循环和 KV Cache。

> 当前状态：仓库骨架和配置系统已经就绪，Transformer 本体尚未实现。

## English summary

MiniGPT-CN is an educational, from-scratch implementation of a small Chinese
Decoder-only Transformer. The project focuses on reproducibility and clear
implementations rather than chat capabilities.

## 项目目标

- 从零理解并实现现代 Decoder-only Transformer 的主要组件。
- 在本地完成 smoke test 和小模型 pilot，在云端训练约 33M 参数的正式模型。
- 保存训练配置、曲线、评测结果和成本，形成可以展示在简历中的完整项目。
- 提供命令行生成工具和简单的 Gradio 演示（后续阶段实现）。

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

完整配置见 `configs/minigpt_33m.yaml`。

## 本地环境

- Python 3.10+
- Windows / Linux
- 本地开发设备：RTX 4060 Laptop 8GB、约 31GB 内存
- 正式训练计划：AutoDL 单张 RTX 4090 24GB

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

详细安排见 `docs/learning-roadmap.md`。

## 数据与发布原则

- 训练数据、checkpoint、日志和密钥永不提交到 Git。
- 正式数据必须在 `data/README.md` 中记录来源、日期、许可和校验和。
- 最终发布 tokenizer、配置、推理权重和校验和，不发布原始语料或优化器状态。

## 已知限制

- 当前仓库尚未实现模型和训练功能。
- 最终模型定位为小型中文文本生成模型，不承诺聊天、事实准确性或安全对齐能力。
- 小模型可能产生重复、幻觉、不完整句子以及训练数据中存在的偏差。

## License

MIT
