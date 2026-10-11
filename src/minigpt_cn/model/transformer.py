"""接入 ModelConfig 的 decoder-only MiniGPT，兼容旧 pilot 权重键名。"""

# 原作业来源：transformer_block.py 和 transformer_lm.py，现在合并到这个文件。
# TransformerBlock 保留类名；TransformerLM 在项目里改名为 MiniGPT。
# 原作业的层名/权重名保持不变，所以旧 checkpoint 不需要重新映射权重键。

from dataclasses import dataclass

import torch
from torch import nn

from minigpt_cn.config import ModelConfig
from minigpt_cn.model.attention import CausalSelfAttention, validate_layer_cache
from minigpt_cn.model.components import Dropout, Embedding, Linear, RMSNorm, SwiGLU, cross_entropy

# F实现：每层是一对 (已做RoPE的K,V)，整体按层排列。
# 缓存由调用者持有，不保存在模型属性中，不同请求不能混用，也不进入 checkpoint。
KVCache = tuple[tuple[torch.Tensor, torch.Tensor], ...]


@dataclass
class ModelOutput:
    # 项目新增：原 TransformerLM 直接返回 logits Tensor，这里返回统一结果对象。
    # 调用者用 model(input_ids).logits 取出 [batch, sequence, vocabulary]。
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    cache: KVCache | None = None


class TransformerBlock(nn.Module):
    def __init__(self, config, device=None, dtype=None):
        super().__init__()
        # 项目适配：原来逐个传 d_model/num_heads/d_ff/theta，现在统一传 ModelConfig。
        # 原作业：初始化因果自注意力模块。
        self.attn = CausalSelfAttention(config, device, dtype)
        # 初始化两个 RMSNorm 层，分别用于 Attention 和 FFN。
        self.ln1 = RMSNorm(config.d_model, config.rms_norm_eps, device, dtype)
        self.ln2 = RMSNorm(config.d_model, config.rms_norm_eps, device, dtype)
        # 初始化前馈网络 SwiGLU，以及两条残差支路末尾的 dropout。
        self.ffn = SwiGLU(config.d_model, config.d_ff, device, dtype)
        self.attn_dropout = Dropout(config.dropout)
        self.ffn_dropout = Dropout(config.dropout)

    def forward(self, x, token_positions=None, *, use_cache=False, kv_cache=None):
        # 原作业：Attention 子层采用 Pre-norm 结构。
        # x 分为两路：一路不处理直接传送（残差），另一路进 Norm + Attention。
        # 两路形状都是 [batch, sequence, d_model]，所以可以直接相加。
        # Pre-norm：x + dropout(Attention(RMSNorm(x)))，残差均为 [B,S,D]。
        attention = self.attn(self.ln1(x), token_positions, use_cache=use_cache, kv_cache=kv_cache)
        if use_cache:
            attention, new_cache = attention
        x = x + self.attn_dropout(attention)
        # 原作业：FFN 子层再次分为两路，一路残差，另一路 Norm + FFN。
        # 注意先更新了 Attention 残差的 x，再对这份新 x 做 FFN。
        # 缓存时 x 只包含新位置，所以 FFN 也只计算新位置，不重新计算历史 FFN。
        x = x + self.ffn_dropout(self.ffn(self.ln2(x)))
        return (x, new_cache) if use_cache else x


class MiniGPT(nn.Module):
    def __init__(self, config: ModelConfig, device=None, dtype=None):
        super().__init__()
        # 项目适配：只接入已验收的无 bias、RoPE 架构，不静默忽略暂不支持的配置。
        if config.bias or config.position_encoding != "rope":
            raise NotImplementedError("本轮仅迁移已验收的 bias=false、RoPE 架构")
        self.config = config
        # config.context_length 对应原 max_seq_len，n_layers 对应原 num_layers，
        # n_heads 对应原 num_heads；这些命名在 configs/*.yaml 中统一。
        self.max_seq_len = config.context_length
        self.embedding_dropout = Dropout(config.dropout)
        # 原作业第1步：tokens 进入 embedding 层，[B,S] -> [B,S,D]。
        self.token_embeddings = Embedding(config.vocab_size, config.d_model, device, dtype)
        # 原作业第2步：堆叠 Transformer Blocks。
        # ModuleList 会注册各层参数，保留 layers.0、layers.1 等 checkpoint 键名。
        self.layers = nn.ModuleList(
            [TransformerBlock(config, device, dtype) for _ in range(config.n_layers)]
        )
        # 原作业第3步：最终归一化层。项目配置目前固定 RMSNorm，
        # 没有沿用作业 use_rms_norm=False 的 Identity 实验开关。
        self.ln_final = RMSNorm(config.d_model, config.rms_norm_eps, device, dtype)
        # 原作业第4步：最后的 Linear（LM head）映射回词表大小，[B,S,D] -> [B,S,V]。
        self.lm_head = Linear(config.d_model, config.vocab_size, device, dtype)
        # 原作业默认不共享时 std=1；共享实验默认 std=0.02。
        # 项目适配：现在统一从 config.embedding_init_std 读取，pilot 显式设为0.02。
        std = config.embedding_init_std
        nn.init.trunc_normal_(self.token_embeddings.W, std=std, a=-3 * std, b=3 * std)
        if config.tie_embeddings:
            # 原作业：同一个 Parameter，两条计算路径的梯度累加，优化器只更新一次。
            # 不能只复制数值：必须是同一个 Parameter，两条路径的梯度才能累加。
            # parameters() 自动去重，优化器只会对共享矩阵更新一次。
            self.lm_head.W = self.token_embeddings.W

    def _validate_ids(self, ids, name):
        # 项目新增：对 input_ids 和 targets 提前检查类型、形状、设备和 token 范围。
        # 这些是 token ID，只能用整数 long，不能随模型精度改成 BF16/FP16。
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
            raise TypeError(f"{name} 必须为 torch.long 张量")
        if ids.ndim != 2 or min(ids.shape) == 0:
            raise ValueError(f"{name} 必须为非空 [batch, sequence] 张量")
        if ids.device != self.token_embeddings.W.device:
            raise ValueError(f"{name} 与模型必须位于同一设备")
        if (ids < 0).any() or (ids >= self.config.vocab_size).any():
            raise ValueError(f"{name} 的 token ID 超出词表范围")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        use_cache: bool = False,
        kv_cache: KVCache | None = None,
    ) -> ModelOutput:
        # 项目适配：原 forward(token_ids) 增加可选 targets 和缓存接口，返回 ModelOutput。
        if not isinstance(use_cache, bool):
            raise TypeError("use_cache 必须为 bool")
        if kv_cache is not None and not use_cache:
            raise ValueError("传入 KV Cache 时必须开启 use_cache")
        if use_cache and self.training:
            raise ValueError("KV Cache 仅用于 eval 推理，请先调用 model.eval()")
        self._validate_ids(input_ids, "input_ids")
        batch, sequence = input_ids.shape
        past_length = 0
        if kv_cache is not None:
            if not isinstance(kv_cache, tuple) or len(kv_cache) != self.config.n_layers:
                raise ValueError("KV Cache 层数必须与 n_layers 一致，整体必须为 tuple")
            lengths = [
                validate_layer_cache(
                    cache,
                    batch=batch,
                    heads=self.config.n_heads,
                    head_dim=self.config.d_model // self.config.n_heads,
                    device=input_ids.device,
                )
                for cache in kv_cache
            ]
            if len(set(lengths)) != 1:
                raise ValueError("各层 KV Cache 的历史 sequence 长度必须一致")
            past_length = lengths[0]
        if past_length + sequence > self.config.context_length:
            raise ValueError("输入加缓存超过 context_length，请裁剪窗口并重建缓存")
        if targets is not None:
            self._validate_ids(targets, "targets")
            if targets.shape != input_ids.shape:
                raise ValueError("targets 形状必须与 input_ids 一致")
        # 原作业：准备位置信息用于 RoPE。[S] -> [1,S] -> [B,S]，所有样本共用顺序位置。
        # F新增：prefill 从0开始；decode 从历史长度开始。例如已有3个位置，新位置为3、4。
        positions = (
            (torch.arange(sequence, device=input_ids.device) + past_length)
            .unsqueeze(0)
            .expand(batch, -1)
        )
        # Embedding：从词表中查向量，再应用 embedding dropout，得到 [B,S,D]。
        x = self.embedding_dropout(self.token_embeddings(input_ids))
        # 逐层通过 Transformer Block，每层输入/输出保持 [B,S,D]。
        new_caches = []
        for index, layer in enumerate(self.layers):
            if use_cache:
                layer_cache = None if kv_cache is None else kv_cache[index]
                x, new_cache = layer(x, positions, use_cache=True, kv_cache=layer_cache)
                new_caches.append(new_cache)
            else:
                x = layer(x, positions)
        # 最终 RMSNorm，再投影到词表空间得到 logits；这里还不做 softmax。
        # logits 是每个位置对 V 个候选 token 的原始分数。
        logits = self.lm_head(self.ln_final(x))  # [batch, sequence, vocabulary]
        # targets 是调用者已右移一位的目标；这里不再移动，避免 double shift。
        # 项目新增：例如 input_ids=[a,b,c]，targets 应是 [b,c,d]；loss 为平均交叉熵。
        loss = None if targets is None else cross_entropy(logits, targets)
        # logits/loss 只对应这次输入的新位置，历史不重复输出；无缓存路径仍返回 cache=None。
        return ModelOutput(logits=logits, loss=loss, cache=tuple(new_caches) if use_cache else None)
