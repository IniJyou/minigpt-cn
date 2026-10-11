"""循环参考 Attention 与向量化因果多头注意力。"""

# 原作业来源：attention.py；scaled_dot_product_attention 和 CausalSelfAttention 仍在一起。
# 项目适配：增加形状校验、接入 ModelConfig；拆头不用 einops，但张量顺序不变。

import math

import torch
from torch import nn

from minigpt_cn.model.components import Linear, softmax
from minigpt_cn.model.rope import RotaryPositionalEmbedding


def validate_layer_cache(cache, *, batch, heads, head_dim, device, dtype=None):
    """检查一层 (K,V)，返回历史长度；不接受形状错误或隐式设备/精度转换。"""
    if not isinstance(cache, tuple) or len(cache) != 2:
        raise ValueError("每层 KV Cache 必须是 (K,V) 二元 tuple")
    k, v = cache
    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("KV Cache 的 K/V 必须为张量")
    if k.ndim != 4 or v.ndim != 4 or k.shape != v.shape:
        raise ValueError("KV Cache 的 K/V 必须同形状 [batch,heads,sequence,head_dim]")
    if (k.shape[0], k.shape[1], k.shape[3]) != (batch, heads, head_dim) or k.shape[2] <= 0:
        raise ValueError("KV Cache 的 batch/heads/head_dim 不匹配，sequence 必须非空")
    if k.device != device or v.device != device:
        raise ValueError("KV Cache 与输入必须位于同一设备")
    if not k.is_floating_point() or k.dtype != v.dtype or (dtype is not None and k.dtype != dtype):
        raise TypeError("KV Cache dtype 必须与本次 Q/K/V 前向精度一致")
    return k.shape[2]


def _validate_qkv(q, k, v, mask):
    # 项目新增：在进入数学运算之前检查张量，避免稍后只得到难懂的 einsum 报错。
    if min(q.ndim, k.ndim, v.ndim) < 2:
        raise ValueError("Q/K/V 至少包含 [sequence, head_dim] 两维")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("Q/K/V 的 batch/head 维度必须一致")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("Q/K 的 head_dim、K/V 的 sequence 必须分别一致")
    if min(*q.shape, *k.shape, *v.shape) <= 0:
        raise ValueError("Q/K/V 的维度必须非空")
    if q.dtype != k.dtype or k.dtype != v.dtype or not q.is_floating_point():
        raise TypeError("Q/K/V 必须使用相同浮点 dtype")
    if q.device != k.device or k.device != v.device:
        raise ValueError("Q/K/V 必须位于同一设备")
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask 必须为 bool，True 表示允许访问")
        shape = (*q.shape[:-2], q.shape[-2], k.shape[-2])
        try:
            expanded = torch.broadcast_to(mask, shape)
        except RuntimeError as error:
            raise ValueError("attention mask 不能广播到 [batch, heads, query, key]") from error
        if not expanded.any(dim=-1).all():
            raise ValueError("每个 query 至少必须允许一个 key")


def scaled_dot_product_attention(q, k, v, mask=None):
    _validate_qkv(q, k, v, mask)
    # 原作业：Q [...,n,d_k]，K [...,m,d_k]，V [...,m,d_v]。
    # n 是查询序列长度，m 是键值序列长度；自注意力中通常 n=m=S。
    # mask：[n,m] 或可广播的 bool 张量，True 保留，False 掩盖。
    # Q/K/V：[batch, heads, sequence, head_dim]。
    # A = softmax(Q K^T / sqrt(head_dim))，scores：[batch, heads, query, key]。
    # 1. 计算相似度分数：沿 head_dim 做点积，再除以 sqrt(d_k)。
    scores = torch.einsum("...nk, ...mk -> ...nm", q, k) / math.sqrt(q.shape[-1])
    if mask is not None:
        # 2. 应用掩码：False 位置设为极大的负数，让 softmax 后概率为零。
        # 保留 pilot 的有限最小值掩码，而不是更换数值计算路径。
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    # 3. 沿 key 维做 softmax 得到注意力权重；4. 按这些权重对 V 加权求和。
    # 输出 [...,n,d_v]；在多头自注意力里为 [batch, heads, sequence, head_dim]。
    return torch.einsum("...nm, ...mk -> ...nk", softmax(scores), v)


def attention_reference(q, k, v, mask=None):
    """便于学习的循环实现，仅供小张量测试，不能用于正式训练。"""
    # 项目新增：原作业文件没有这个参考函数，用它核对上面向量化实现的数值。
    # 明确遍历 batch、head、query；每个 query 都与所有 key 算分数，再加权 V。
    _validate_qkv(q, k, v, mask)
    if q.ndim != 4:
        raise ValueError("循环参考实现要求 [batch, heads, sequence, head_dim]")
    mask = None if mask is None else torch.broadcast_to(mask, (*q.shape[:3], k.shape[-2]))
    batches = []
    for batch in range(q.shape[0]):
        heads = []
        for head in range(q.shape[1]):
            queries = []
            for query in range(q.shape[2]):
                scores = torch.stack(
                    [
                        (q[batch, head, query] * key).sum() / math.sqrt(q.shape[-1])
                        for key in k[batch, head]
                    ]
                )
                if mask is not None:
                    scores = scores.masked_fill(~mask[batch, head, query], torch.finfo(q.dtype).min)
                probabilities = softmax(scores)
                queries.append(
                    sum(probabilities[key] * v[batch, head, key] for key in range(k.shape[-2]))
                )
            heads.append(torch.stack(queries))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        # 项目适配：原来的 d_model/num_heads/max_seq_len/theta 参数从 config 统一取得。
        # config.n_heads 对应原 num_heads，head_dim 对应原 d_k。
        # d_model 必须被 n_heads 整除，这个约束现在由 ModelConfig 在构建前校验。
        self.d_model = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        # 原作业：Q/K/V 三个投影层，把同一份输入映射到三个不同的特征空间。
        self.q_proj = Linear(config.d_model, config.d_model, device, dtype)
        self.k_proj = Linear(config.d_model, config.d_model, device, dtype)
        self.v_proj = Linear(config.d_model, config.d_model, device, dtype)
        # 输出投影层：整合所有 head 的信息，仍然映射到 d_model。
        self.output_proj = Linear(config.d_model, config.d_model, device, dtype)
        # 项目当前固定使用 RoPE；theta 和最大上下文长度都从模型配置读取。
        self.rope = RotaryPositionalEmbedding(
            config.rope_theta, self.head_dim, config.context_length, device
        )

    def forward(self, x, token_positions=None, *, use_cache=False, kv_cache=None):
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError("Attention 输入必须为 [batch, sequence, d_model]")
        if kv_cache is not None and not use_cache:
            raise ValueError("传入 KV Cache 时必须开启 use_cache")
        if use_cache and self.training:
            raise ValueError("KV Cache 仅用于 eval 推理，请先调用 model.eval()")
        batch, sequence, _ = x.shape
        # 原作业：从倒数第2维取序列长度 s，最后一维是隐藏维度 d。
        # 线性投影并拆分多头：把长度 D 拆为 H*Dh，再将 H 移到 S 前面。
        # 项目适配：对应原 rearrange('... s (h d) -> ... h s d')：
        # reshape 是拆维度，transpose(1,2) 才是交换 sequence 和 heads 的位置。
        # [B,S,D] -> [B,S,H,Dh] -> [B,H,S,Dh]，与原 einops rearrange 同序。
        q, k, v = [
            projection(x).reshape(batch, sequence, self.num_heads, self.head_dim).transpose(1, 2)
            for projection in (self.q_proj, self.k_proj, self.v_proj)
        ]
        # 原作业：应用 RoPE，只旋转 Q 和 K，V 保持不动。
        # F新增：每层缓存自己的 K/V，精度在投影后检查（autocast 时可能为 BF16）。
        past_length = (
            0
            if kv_cache is None
            else validate_layer_cache(
                kv_cache,
                batch=batch,
                heads=self.num_heads,
                head_dim=self.head_dim,
                device=x.device,
                dtype=k.dtype,
            )
        )
        if past_length + sequence > len(self.rope.cos_cached):
            raise ValueError("缓存长度加新输入超过 context_length，请裁剪窗口并重建缓存")
        # token_positions=None 时从 offset 开始；历史 K 已旋转，不能再次应用 RoPE。
        q = self.rope(q, token_positions, offset=past_length)
        k = self.rope(k, token_positions, offset=past_length)
        if kv_cache is not None:
            # [B,H,past,Dh] + [B,H,new,Dh] -> [B,H,past+new,Dh]，不原地修改旧缓存。
            k = torch.cat((kv_cache[0], k), dim=-2)
            v = torch.cat((kv_cache[1], v), dim=-2)
        # 下三角：位置 i 只允许访问 j <= i，未来 token 的概率为零。
        # 原作业：生成因果 mask，确保 Query 只能看到当前位置以及之前的 Key。
        if use_cache:
            # 非方形 mask：[new_sequence,total_sequence]，按窗口内绝对位置判断。
            # 例如历史3个、新增2个，两个 query 分别允许 key 0..3 和 key 0..4。
            query_positions = torch.arange(sequence, device=x.device) + past_length
            key_positions = torch.arange(k.shape[-2], device=x.device)
            mask = key_positions[None, :] <= query_positions[:, None]
        else:
            # 不改变原无缓存计算路径，保证原训练和 pilot 的数值验收继续成立。
            mask = torch.ones(sequence, sequence, dtype=torch.bool, device=x.device).tril()
        result = scaled_dot_product_attention(q, k, v, mask)
        # 合并多头：原 rearrange('... h s d -> ... s (h d)')。
        # [B,H,S,Dh] -> [B,S,H,Dh] -> [B,S,D]，不能跳过 transpose 直接 reshape。
        result = result.transpose(1, 2).reshape(batch, sequence, self.d_model)
        # 返回输出投影，形状 [batch, sequence, d_model]，可与残差相加。
        result = self.output_proj(result)
        # 缓存是推理状态，detach 避免历史 autograd 图随序列增长；不写入 state_dict。
        return (result, (k.detach(), v.detach())) if use_cache else result
