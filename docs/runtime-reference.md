# 模型与运行接口参考

已验收功能的查询页，不是当前待办；下一任务见 [工作交接](current-work.md)。
常规代码不导入作业仓库，迁移对照工具仅在显式复核时读取作业参考。

## 模型与张量

```python
from minigpt_cn.config import load_model_config
from minigpt_cn.model import MiniGPT
model = MiniGPT(load_model_config('configs/minigpt_33m.yaml'))
output = model(input_ids, targets=None, use_cache=False, kv_cache=None)
```

input_ids/targets为整数[B,S]；logits为[B,S,V]；loss为可选标量；cache为可选逐层K/V。
调用端将token右移：输入[a,b,c]，目标[b,c,d]；模型不再内部移动目标。

| 模块 | 公式/形状 | 注意点 |
| --- | --- | --- |
| Attention | softmax(QKᵀ/√Dh)V；Q/K/V [B,H,S,Dh] | mask屏蔽未来位置 |
| RMSNorm | gamma*x/sqrt(mean(x²)+eps)；[B,S,D] | FP32均方 |
| RoPE | 偶奇通道按位置角度旋转；[B,H,S,Dh] | 只转Q/K，offset=缓存历史长度 |
| SwiGLU | (SiLU(xW1ᵀ)*xW3ᵀ)W2ᵀ | 中间[B,S,d_ff]，相乘不是拼接 |
| Block | x+dropout(attn(norm(x)))，再FFN残差 | 保留已验收dropout位置 |
| tied embedding | 同一个[V,D] Parameter | 两路径梯度累加，优化器不重复更新 |
| Loss | mean(logsumexp(logits)-target_logit) | 目标调用端右移一次 |

代码：src/minigpt_cn/model/{components,attention,rope,transformer}.py。
正式模型配置为minigpt_33m.yaml；pilot配置V2048/D256/L4/context256，仅作回归。

## 运行入口

按需用--help查看实时参数，不默认重训、重建环境或重新导入旧checkpoint：

```bash
python -m minigpt_cn.training.train --help
python -m minigpt_cn.evaluation.evaluate --help
python -m minigpt_cn.inference.generate --help
python -m minigpt_cn.inference.benchmark --help
python -m minigpt_cn.training.checkpoint --help
```

- 训练：--config统一模型/训练/数据配置，--out-dir新输出目录；--resume明确checkpoint。
  --max-steps改变终点，不改变lr_decay_steps；实际超参/恢复约束以配置和源码为准。
- 精度：CPU FP32；CUDA BF16前向、FP32参数/优化器状态，不把纯FP16参数当混合精度。
  数值非有限时失败退出，不保存正常最终checkpoint。
- 完整验证：覆盖全部next-token目标，按目标数加权平均，不只采一个batch。
- 新checkpoint保存配置、真实步数、优化器/RNG/采样状态；旧pilot导入仅推理，不能精确续训。
  移动文件后可显式给同版本tokenizer，不能换词表/ID语义。
- 生成：单条、greedy或temperature/top-p、EOS停止、可选--use-cache；不承诺文本质量。
- KV Cache：每层保存旋转后的K与V，新token增量注意力；只用于eval，不跨请求复用。
  满窗口rebuild_on_slide重建缓存/位置，与无缓存裁剪语义一致，满窗口无持续增量加速。

CPU测试和cached/uncached greedy一致性已验收；数字见 [实验结论](experiment-log.md)。
33M本地batch2/context512/BF16短验证已实测：100步、精确恢复及独立重载验证均通过。
实际参数33104384、训练日志区间吞吐约7637 tokens/s、PyTorch最大分配显存1.175GiB；
环境、耗时/显存范围和复现见 [33M短验证](33m-short-validation.md)。
实际云端资源、长期质量与费用仍未实测，不从本地短验证数字外推正式预训练效果或云端预算。
