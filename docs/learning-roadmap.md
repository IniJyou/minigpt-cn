# MiniGPT-CN 十周学习路线

建议投入：每周 8～12 小时。时间不足时可以延长周期，不要跳过测试和小规模验证。

## 第 1 周：PyTorch 与语言模型基础

- Tensor、广播、矩阵乘法、autograd、`nn.Module`、Dataset。
- 手写线性回归和字符级 bigram 模型。
- 验收：训练 loss 能够稳定下降，并能解释每个张量的形状。

## 第 2 周：Self-Attention

- embedding、softmax、Q/K/V、scaled dot-product attention、causal mask。
- 先写循环参考实现，再写向量化实现并比较输出。
- 验收：未来 token 不会影响当前位置输出。

## 第 3 周：Transformer Block

- Multi-Head Attention、RMSNorm、RoPE、SwiGLU、残差和 dropout。
- 验收：模块形状正确、梯度有限、参考测试通过。

## 第 4 周：Byte-level BPE

- 实现 `train`、`encode`、`decode`、`save`、`load`。
- 验收：中文、emoji、空字符串和换行能够 UTF-8 round-trip。

## 第 5 周：数据管线

- 下载并记录数据许可与校验和。
- 清洗、文档去重、hash 分割、`uint16` 分片和 memory map。
- 验收：划分之间无文档交集，重复运行结果一致。

## 第 6 周：完整 MiniGPT 与训练系统

- 组合模型，实现 loss、优化器、scheduler、梯度裁剪和 checkpoint。
- 验收：模型输出为 `[batch, sequence, vocabulary]`，checkpoint 可独立恢复。

## 第 7 周：本地验证

- smoke 配置过拟合单个小 batch。
- RTX 4060 上训练 5M～15M 参数 pilot。
- 验收：无 NaN/Inf，中断恢复后的下一步 loss 近似一致。

## 第 8 周：AutoDL 正式训练

- 单张 RTX 4090 24GB，先运行 500 steps 测速和估价。
- 首轮 1 亿 tokens；验证集仍改善时扩展到 2 亿 tokens。
- 定期把最佳 checkpoint 备份到本地。

## 第 9 周：评测与消融

- Character tokenizer vs 16K BPE。
- Learned positional embedding vs RoPE。
- 无 KV Cache vs KV Cache。

## 第 10 周：作品集发布

- 流式 CLI、Gradio、训练曲线、模型卡、真实生成样例。
- 发布推理权重、tokenizer、配置和校验和。
- 准备演示视频和包含真实数字的简历描述。
