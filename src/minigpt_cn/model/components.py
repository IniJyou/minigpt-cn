"""从已验收作业迁移的基础层，保留 W/gain 参数名与运算顺序。"""

# 原作业来源：linear.py、embedding.py、rmsnorm.py、swiglu.py、dropout.py、utils.py。
# 这里把短组件合并到一个文件，没有把它们替换为 PyTorch 的完整 Transformer。
# 下文 B=batch、S=sequence、D=d_model、V=vocabulary；“项目适配”标出迁移差异。

import math

import torch
from torch import nn


class Linear(nn.Module):
    # 原作业：实现 y = x @ W^T。
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        # 1. 定义 W，形状为 [out_features, in_features]；注意传入 device 和 dtype。
        # nn.Parameter 告诉 PyTorch：这是需要通过训练学习的权重，而不是普通临时张量。
        self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        # 2. 按作业要求初始化：mean=0，sigma^2=2/(d_in+d_out)。
        std = math.sqrt(2.0 / (in_features + out_features))
        # 截断范围 [-3*sigma, 3*sigma]，避免初始化时出现过大的权重。
        nn.init.trunc_normal_(self.W, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        # y = x W^T：[..., in_features] -> [..., out_features]。
        # 项目适配：先校验最后一维，非法输入给出清楚的异常。
        if x.ndim == 0 or x.shape[-1] != self.W.shape[1]:
            raise ValueError("Linear 输入最后一维必须等于 in_features")
        # einsum 中 i 是输入维度，o 是输出维度；... 保留 batch/sequence 等前置维度。
        return torch.einsum("...i, oi -> ...o", x, self.W)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        # 原作业 num_embeddings 对应 vocab_size，embedding_dim 对应 d_model。
        # 1. 分配内存并包装为 Parameter：W 的形状为 [vocabulary, d_model]。
        self.W = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype))
        # 2. 保留作业层的初始化：mean=0、std=1.0，截断到 [-3,3]。
        # 项目适配：完整 MiniGPT 会按 config.embedding_init_std 再初始化这张矩阵；
        # pilot 的共享模式使用 std=0.02，具体逻辑在 transformer.py 中。
        nn.init.trunc_normal_(self.W, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        # token_ids：[B,S]；直接按整数 ID 从 W 取出对应的向量，而不是做矩阵乘法。
        return self.W[token_ids]  # [batch, sequence] -> [batch, sequence, d_model]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        # 学习参数 gain（gamma）：缩放参数，初始化为全1，形状为 [D]。
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x):
        # 原作业输入 x：[batch_size, seq_len, d_model]，输出保持同形状。
        if x.ndim == 0 or x.shape[-1] != self.gain.numel():
            raise ValueError("RMSNorm 输入最后一维必须等于 d_model")
        # y = gamma * x / sqrt(mean(x^2) + eps)，形状仍为 [..., d_model]。
        # 与 pilot 一致：均方和归一化用 FP32，最终转回输入精度。
        # 1. 转换为 float32，降低低精度平方/求均值溢出的风险。
        x_float = x.float()
        # 2. 沿最后一个 D 维计算均方；keepdim=True 保留 [B,S,1]，便于广播。
        # epsilon 在开平方之前相加：rms=sqrt(mean_square+eps)。
        rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        # 3. 归一化，再乘 gain；[D] 自动广播到 [B,S,D]，最后转回输入 dtype。
        return ((x_float / rms) * self.gain).to(x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        # 原作业：W1 和 W3 是并行升维层，d_model -> d_ff。
        # 输入 [B,S,D]，两条支路的输出都是 [B,S,d_ff]。
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        # W2 是降维层，d_ff -> d_model，输出回到残差流的 [B,S,D]。
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(self, x):
        # (SiLU(x W1^T) * x W3^T) W2^T；中间为 [batch, sequence, d_ff]。
        # 原作业 silu_fn(input)=input*sigmoid(input)，这里把同一公式内联了。
        # gate 此时是 W1 投影的结果，下一行才应用 SiLU；W3 提供 signal。
        gate = self.w1(x)
        # SiLU(gate) 与 signal 逐元素相乘，不是矩阵乘法，也不是拼接。
        return self.w2((gate * torch.sigmoid(gate)) * self.w3(x))


class Dropout(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p < 1:
            raise ValueError("dropout 必须满足 0 <= p < 1")
        self.p = float(p)

    def forward(self, x):
        # 原作业：默认 p=0 不消耗随机数，eval 时也保持原值。
        if not self.training or self.p == 0:
            return x
        # inverted dropout：除以保留率，让训练时输出的期望不变；p=0 不消耗 RNG。
        keep = torch.rand_like(x, dtype=torch.float32) >= self.p
        # keep 与 x 同形状；被保留的值除以 (1-p)，不是在 eval 时再缩放。
        return x * keep.to(x.dtype) / (1.0 - self.p)


def softmax(x, dim=-1):
    # 原作业：先减去这一行的最大值，再 exp 和归一化，避免大 logits 导致 exp 溢出。
    # Attention 的最后一维是 key 位置，模型输出的最后一维则是 vocabulary。
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(logits, targets):
    """保持 pilot 的 log-sum-exp 运算顺序，返回所有目标的平均 NLL。"""
    # logits：[batch, sequence, vocabulary]；targets：[batch, sequence]。
    logits = logits.float()
    # 1. 求 logits 最大值 M，形状 [B,S,1]，用于稳定计算 log-sum-exp。
    maximum = logits.max(dim=-1, keepdim=True).values
    # 2. 从词表维提取真实目标对应的 logit：targets [B,S] -> [B,S,1]。
    # gather 后 squeeze 回 [B,S]；不是对整行 logits 取最大值作为答案。
    selected = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    # 3. log(sum(exp(logits))) = M + log(sum(exp(logits-M)))，结果 [B,S]。
    log_sum_exp = maximum.squeeze(-1) + (logits - maximum).exp().sum(-1).log()
    # 4. 每个 token 的损失为 log_sum_exp-target_logit，再对 B、S 全部取平均。
    return (log_sum_exp - selected).mean()
