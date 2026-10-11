# 中文 pilot 迁移验收

日期：2026-10-09。范围是“迁移并复现”，不是重训或正式 33M 资源基准。

## 迁入内容与边界

- 模型：Linear、Embedding、RMSNorm、SwiGLU、RoPE、因果多头注意力、残差与 dropout。
- 完整 MiniGPT：接入不可变 `ModelConfig`，返回 `ModelOutput(logits, loss, cache)`。
- BPE：已训练词表的编解码、流式编码和 pilot JSON 保存加载。
- 验收工具：明确指定作业路径后加载参考模型；常规模型和 tokenizer 不跨仓库导入。

本轮不迁移 BPE 训练、下载/清洗流水线、AdamW 和训练入口，不实现生产生成 CLI 或
KV Cache。它们按下一阶段分别接入，避免把迁移一致性与新功能混在一起。
作业仓库、学习笔记、原权重和语料没有修改或复制进 Git。

## 原始样本

相对于 `D:\aiworkspace\cs336-assignment1-basics`：

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `out/pilot_3m_bpe2048/ckpt_final.pt` | 5000 次更新后的旧 checkpoint | `505fafd02381b8a92dc359a6c7f0801db9f444d23a1e11c7f7123b7eacc61ce0` |
| `data/processed/tokenizer_2048.json` | 2048 BPE、EOS ID 256 | `22089eb3002b9c24509c5bc37cef710994c9ace095f600bb26a71877b6e1ea12` |
| `data/processed/valid.bin` | 无文件头、小端 uint16，37,827 个 token | `eda11af1f8f1d2874697b0d4cfdd1f111dd6dcf735ed8bf578c677e840bb4adb` |
| `data/processed/valid.jsonl` | 全部 7 篇验证文档 | `d1522a0a4e3a07470b8164619fd05f9ef19c3dae5a601fa9e3f97f14ca78b11d` |

结构配置为 `configs/pilot_3m.yaml`：D=256、4 层、8 头、FFN=704、上下文256、
RoPE theta=10000、RMSNorm epsilon=1e-5、共享 embedding、dropout=0。
唯一可训练参数量为 3,737,856。旧 checkpoint 不含模型配置，因此此 YAML 是明确的恢复前提。

## 实测结果

WSL、PyTorch `2.11.0+cu130`；GPU 为 RTX 4060 Laptop。

| 精度/设备 | 原实现 loss | 项目 loss | 项目 PPL | 最大 logits 差异 |
| --- | ---: | ---: | ---: | ---: |
| CUDA / BF16 前向、FP32 参数 | 3.8759773192 | 3.8759773192 | 48.2298111859 | 0 |
| CPU / FP32 | 3.8759228132 | 3.8759228132 | 48.2271824449 | 0 |

两个模式分别对照同精度参考模型，均严格 `rtol=0, atol=0`。
FP32 与 BF16 之间的轻微差别是精度对照差异，不能要求跨精度 greedy 文本相同。

全部 37,826 个 next-token 目标逐一覆盖，包含最后不足256个目标的尾块。
损失按目标数加权：`总 NLL / 总目标数`，不是简单平均各块的 loss。
验证文档重新编码后与原 `valid.bin` 逐 ID 相同。
原模型和项目模型重载后均共享同一个 Parameter，`parameters()` 中只出现一次。

三个 prompt（“数学是一门”“在计算机科学中，”“中国的历史”）各最多续写32个
greedy token，同设备/精度下迁移前后逐 token 相同。
例如 CUDA/BF16 的第一条输出：

> 数学是一门的数学，在数学上，他的数学家，他的数学家，他的数学家，他的数学家，他的数学家，

这说明生成行为保留，并不说明质量已经达标；重复、语义不连贯和非法 UTF-8 替换字符
仍是原模型的已知限制。迁移没有尝试修饰生成结果。

## 公共模型接口

```python
from minigpt_cn.config import load_model_config
from minigpt_cn.model import MiniGPT

model = MiniGPT(load_model_config("configs/pilot_3m.yaml"))
output = model(input_ids, targets=None, use_cache=False, kv_cache=None)
logits = output.logits  # [batch, sequence, vocabulary]
```

`targets` 应与输入同形状，并由调用者事先右移一个 token；模型不再内部移动。
例如原始 tokens 为 `[a,b,c,d]`，则输入 `[a,b,c]`、目标 `[b,c,d]`。
有 targets 时返回 FP32 平均交叉熵，无 targets 时 `loss=None`。
步骤1验收时 `cache=None`，请求缓存会明确报错。后续步骤2F已实现可选缓存：
默认无缓存仍返回None，eval下开启use_cache返回每层K/V；详见 `docs/step2-guide.md` F段。
超过上下文的输入明确报错，生成端须先裁剪。learned 位置编码和 bias 尚未迁移，会明确报错。

旧权重键名（如 `token_embeddings.W`、`layers.0.attn.q_proj.W`）保持不变，
所以可以 `load_state_dict(ckpt["model_state_dict"], strict=True)`，不需要改写 checkpoint。

## 数学复习

| 模块 | 公式和形状 | 易错点 |
| --- | --- | --- |
| Attention | `softmax(QKᵀ/√Dh)V`；Q/K/V 为 `[B,H,S,Dh]` | 下三角允许访问，不能把未来位置留在 softmax 中 |
| RMSNorm | `gamma*x/sqrt(mean(x²)+eps)`；输入输出 `[B,S,D]` | FP32 计算均方，保留 pilot 的 epsilon |
| RoPE | 每对偶奇通道按位置角度旋转；Q/K 为 `[B,H,S,Dh]` | 只旋转 Q/K，offset 不能按每段从零重算 |
| SwiGLU | `(SiLU(xW1ᵀ) * xW3ᵀ)W2ᵀ`；中间 `[B,S,d_ff]` | 两条升维路径相乘，不是拼接 |
| Block | `x+dropout(attn(norm(x)))`，再经过 FFN 残差 | dropout 在 embedding 和残差支路，不新增 attention 概率 dropout |
| 权重共享 | embedding/LM head 共用 `[V,D]` Parameter | 梯度应为两条路径之和，优化器只能更新一次 |
| Loss | `mean(logsumexp(logits)-target_logit)`，标量 | 目标在调用端右移，不能 double shift |

RoPE offset已有组件测试；步骤2F已补齐完整模型的缓存/窗口重建测试。

## 复跑

步骤2独立环境搭建见 `docs/step2-guide.md`。安装项目后，在 WSL 项目目录执行：

```bash
cd /mnt/d/aiworkspace/transformer
PY=/home/luoyu/.venvs/minigpt-cn/bin/python
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m pytest -q
"$PY" -m ruff check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -m minigpt_cn.evaluation.verify_pilot \
  --assignment-root /mnt/d/aiworkspace/cs336-assignment1-basics \
  --device cuda --precision bfloat16 \
  --output logs/pilot-migration-cuda-bf16.json
```

将最后一条改为 `--device cpu --precision float32`、输出为
`logs/pilot-migration-cpu-fp32.json` 可复跑 CPU 对照。
报告含 checkpoint/tokenizer/验证数据/配置与参考代码 SHA-256、运行版本、结果及原始生成 ID。
`logs/` 被 Git 忽略，不提交实验产物。

本次验收为避免未经用户确认创建环境，使用了已有作业 WSL 解释器，并仅将项目 `src`
加入进程级 `PYTHONPATH`，没有安装/更改作业环境。
Windows 项目独立环境的 CPU 测试亦已通过；Windows 的 PyTorch 版本为 `2.14.0+cu126`。
跨版本结果不能当作逐 bit 相同的承诺。
