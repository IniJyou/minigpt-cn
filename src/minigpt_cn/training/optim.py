"""从作业 adamw.py / utils.py 接入的 AdamW、余弦调度与梯度裁剪。

保留作业的更新顺序和 epsilon 位置，不替换为 torch.optim.AdamW。
项目适配：参数/梯度/矩状态保持 FP32，BF16 只用于模型前向。
"""

import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        # 参数检测；项目适配：同时拒绝 NaN/Inf 和零 epsilon。
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("学习率必须为有限非负数")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas):
            raise ValueError("betas 必须包含两个 [0,1) 中的数")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps 必须为有限正数")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("weight_decay 必须为有限非负数")
        # 将超参数存入 defaults 字典，checkpoint 会保存真实参数组。
        super().__init__(
            params, dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self):
        """执行单步优化更新；共享 Parameter 在 model.parameters() 中只出现一次。"""
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps, lr, wd = group["eps"], group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if p.dtype != torch.float32 or grad.dtype != torch.float32:
                    raise ValueError("项目 AdamW 要求 FP32 参数和梯度，不能直接使用 FP16/BF16 参数")
                if grad.is_sparse:
                    raise ValueError("项目 AdamW 不支持稀疏梯度")
                state = self.state[p]
                # 状态初始化：m/v 与参数同形状，比如 embedding 为 [vocabulary,d_model]。
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                # 更新矩估计：m_t = beta1*m_(t-1)+(1-beta1)*g_t。
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t = beta2*v_(t-1)+(1-beta2)*g_t^2。
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # 原作业：把偏差修正合并进步长，epsilon 放在未修正的 sqrt(v_t) 之后。
                # alpha_t = lr*sqrt(1-beta2^t)/(1-beta1^t)。
                step_size = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                denom = exp_avg_sq.sqrt().add_(eps)
                # 应用解耦的权重衰减：theta <- (1-lr*lambda)*theta。
                if wd != 0:
                    p.mul_(1 - lr * wd)
                # addcdiv_: theta <- theta - alpha_t*m_t/(sqrt(v_t)+eps)。
                p.addcdiv_(exp_avg, denom, value=-step_size)
        return None


def lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    # 原作业：第 it 次（从0开始）更新前设置学习率，前 warmup_iters 步线性增长。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    # 退火进度：(it-warmup)/(decay_end-warmup)，训练终点不参与这个公式。
    decay_rate = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_rate))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def clip_gradient_norm(parameters, max_norm):
    # 原作业：过滤没有梯度的参数，计算所有参数梯度拼接后的全局 L2 范数。
    params = [p for p in parameters if p.grad is not None]
    total_norm = math.sqrt(sum(torch.norm(p.grad.detach(), p=2).item() ** 2 for p in params))
    if not math.isfinite(total_norm):
        raise FloatingPointError("梯度全局范数出现 NaN/Inf")
    if total_norm > max_norm:
        # g <- g * max_norm/(||g||_2+1e-6)，不是逐个参数单独裁剪。
        coefficient = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad.mul_(coefficient)
    return total_norm


def check_finite(named_tensors, context):
    """在训练 loss、梯度、参数、矩状态检查点发现异常，立即终止，不保存最终文件。"""
    for name, tensor in named_tensors:
        if tensor is not None and not torch.isfinite(tensor).all().item():
            raise FloatingPointError(f"{context}: {name} 出现 NaN/Inf")
