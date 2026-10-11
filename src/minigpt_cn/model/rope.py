"""成对旋转位置编码；频率、配对方式和 pilot 一致。"""

# 原作业来源：rope.py；类名仍为 RotaryPositionalEmbedding。
# 项目适配：原 d_k 改名为 head_dim，并增加可选位置与 offset、清晰的边界校验。

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, head_dim, max_seq_len, device=None):
        super().__init__()
        # 原作业：theta 是基准频率，d_k 是每个 head 的维度（必须为偶数），
        # max_seq_len 是最大序列长度。偶/奇数通道要成对旋转，所以不能是奇数维。
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim 必须为正偶数")
        self.head_dim = head_dim
        # omega_i = theta^(-2i/head_dim)，每对偶/奇数通道共用一个角度。
        # 原作业：只计算 d_k/2 个频率，因为旋转是成对进行的。
        # arange(0,Dh,2) 得到 [0,2,4,...,Dh-2]，对应公式里的 2i。
        powers = torch.arange(0, head_dim, 2, device=device).float() / head_dim
        frequencies = 1.0 / (theta**powers)
        # 创建位置序列 [0,1,...,max_seq_len-1]，与频率做外积。
        # angles：[max_seq_len, head_dim/2]，每一行是该位置所有通道对的旋转角度。
        angles = torch.outer(torch.arange(max_seq_len, device=device).float(), frequencies)
        # 可由配置重新计算，不写入 checkpoint，以兼容原 pilot 的 state_dict。
        # 原作业：预计算 cos 和 sin，persistent=False 使这些缓存不进入 state_dict。
        # 这里缓存的是“旋转角度表”，不是推理阶段各层保存的 K/V 缓存。
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x, token_positions=None, *, offset=0):
        # x：[batch, heads, sequence, head_dim]，实际只用于 Attention 的 Q/K。
        if x.ndim != 4 or x.shape[-1] != self.head_dim:
            raise ValueError("RoPE 输入必须为 [batch, heads, sequence, head_dim]")
        if token_positions is None:
            # 项目适配：原来必须传 token_positions，现在也能用 offset 生成位置。
            # 例如已有5个位置、这次输入2个 token，offset=5 对应位置 [5,6]。
            # F的增量 Attention 使用这个位置约定；重建滑动窗口时重新从0开始。
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise ValueError("RoPE offset 必须为非负整数")
            token_positions = torch.arange(x.shape[-2], device=x.device) + offset
        if token_positions.shape not in {(x.shape[-2],), (x.shape[0], x.shape[-2])}:
            raise ValueError("token_positions 必须为 [sequence] 或 [batch, sequence]")
        if token_positions.dtype != torch.long:
            raise TypeError("token_positions 必须为 torch.long")
        if (token_positions < 0).any() or (token_positions >= len(self.cos_cached)).any():
            raise ValueError("RoPE 位置超出 context_length")
        # 原作业：按 token_positions 提取 cos/sin，而不是固定取缓存表前 S 行。
        # 位置 [S] -> 表项 [S,Dh/2]；位置 [B,S] -> 表项 [B,S,Dh/2]。
        cos, sin = self.cos_cached[token_positions], self.sin_cached[token_positions]
        # 维度对齐：给 cos/sin 加上 batch/head 维度，以便和 x 的所有 head 广播。
        # 原注释的“x 是4维、cos 是3维”指这里的含 head 输入和按 batch 取出的表。
        while cos.ndim < x.ndim - 1:
            cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        # [B,S,Dh/2] -> [B,1,S,Dh/2]；单个位置序列则得到 [1,1,S,Dh/2]。
        # 确保类型一致：cos/sin 转为 x.dtype，保留 pilot 的低精度前向行为。
        cos, sin = cos.unsqueeze(-3).to(x.dtype), sin.unsqueeze(-3).to(x.dtype)
        # [batch, heads, sequence, head_dim/2]；旋转矩阵保持每对通道的长度。
        # 原作业：拆分偶数/奇数通道，原变量名 x_even/x_odd 现在简写为 even/odd。
        even, odd = x[..., 0::2], x[..., 1::2]
        result = torch.empty_like(x)
        # 对每对 (even,odd) 应用二维旋转矩阵：
        # new_even = even*cos - odd*sin；new_odd = even*sin + odd*cos。
        # 将结果交错写回原通道位置，不能把全部 even 通道与全部 odd 通道直接拼接。
        result[..., 0::2] = even * cos - odd * sin
        result[..., 1::2] = even * sin + odd * cos
        # 输出仍为 [batch, heads, sequence, head_dim]，形状和每对通道的长度都不变。
        return result
