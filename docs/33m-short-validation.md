# 33M本地短训练与恢复验收

2026-10-13已实际运行并验收。本页是运行参考；下一任务只以
[current-work.md](current-work.md)为准，不自动重跑已有实验或启动云端。

## 本轮配置与范围

- 模型：configs/minigpt_33m.yaml，33104384个独立参数；V16000/context512/L8/H8/D512/FF1344。
- 运行：configs/minigpt_33m_short_run.yaml，batch2、100步、seed42、dropout0.1、共享embedding。
- 精度：CUDA BF16前向，参数/梯度/AdamW矩状态保持FP32。
- 数据：已验收pilot_tokens_16k_v1的train/valid及pilot_tokenizer_16k_v1；未使用test训练或调参。
- 环境：/home/luoyu/.venvs/minigpt-cn，PyTorch2.11.0+cu130、RTX4060 Laptop 8GB。
- 确定性：CUBLAS_WORKSPACE_CONFIG=:4096:8、deterministic_algorithms=true、TF32=false。
- 三个独立进程：0→50保存，50→100恢复，0→100连续对照；共计算200次更新。
  被验收的单条训练轨迹是100步、102400个训练tokens，仅验证pilot闭环与可学习性。

Windows环境PyTorch2.14.0+cu126与本轮WSL版本不同；精确恢复继续使用原WSL环境和数据路径。
不把旧2048权重当成33M续训起点，不因100步loss下降宣称正式预训练或生成质量。

## 实际结果

| 已完成步数 | 完整valid loss | valid targets |
| --- | ---: | ---: |
| 0 | 9.790885448663369 | 25271 |
| 50 | 8.151741689660346 | 25271 |
| 100 | 8.052401653122390 | 25271 |

所有更新经过loss、梯度、参数、优化器矩状态的有限数值检查，未出现NaN/Inf。
恢复100步与连续100步的model_state_dict、optimizer、sampler_state和rng_state逐项完全一致；
记录到日志的loss、学习率和裁剪前梯度范数也完全一致。共享参数保持同一Parameter，只更新一次。
新进程独立重载100步checkpoint后，完整valid loss仍为8.052401653122390，覆盖全部25271个目标。

| 运行段 | 本段更新 | 循环耗时（秒） | 循环有效tokens/s | 训练日志区间tokens/s |
| --- | ---: | ---: | ---: | ---: |
| first50 | 50 | 11.81 | 4336 | 7716 |
| resumed100 | 50 | 11.48 | 4458 | 7802 |
| continuous100 | 100 | 19.26 | 5318 | 7637 |

循环耗时包括训练、完整验证与checkpoint保存，排除启动；循环有效吞吐使用本段训练token数除以此耗时。
训练区间吞吐取相邻train日志间的区间，排除夹有验证的区间及初始热身区间；包括日志开销。
本轮save_interval=1000，100步内仅退出时保存，所以这些区间没有checkpoint写出。
累计run_state时间戳在checkpoint写出前捕获；分段实测耗时使用各process_seconds，不用该时间戳代替总耗时。

三段最大PyTorch分配显存1.175GiB、保留显存1.227GiB；范围包含启动、训练和验证，
只统计当前进程CUDA分配器，不包含其他进程与驱动占用。这些是本地batch2短验证数字，
AutoDL实际设备、batch、精度、500步稳定性与费用仍待独立验证。

## 已保存产物（Git忽略）

checkpoints/m3_33m_short_v1下保存：

- first50/、resumed100/、continuous100/：各含ckpt_final.pt、run.json、metrics.jsonl、resources.json。
- reload_valid.json：新进程重载后的完整验证报告。
- acceptance.json：配置/源码/数据/checkpoint摘要、状态一致性、验证曲线、资源统计及CPU测试结果。

本地实际运行封装和完整验收脚本在logs/m3_33m_short_v1，仅作复现实验记录，不提交Git。
目录已存在时先只读检查；不删除或覆盖旧产物，不为重复运行自动生成新训练。

## 只读检查已验收结果（WSL）

```bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
python - <<'PY'
import json
from pathlib import Path
import torch
from minigpt_cn.training.checkpoint import file_sha256

torch.set_num_threads(1)
base = Path('checkpoints/m3_33m_short_v1')
report = json.loads((base / 'acceptance.json').read_text(encoding='utf-8'))
assert report['acceptance'] == 'PASS' and report['completed_iteration'] == 100
assert file_sha256('configs/minigpt_33m_short_run.yaml') == report['config_sha256']
for name, digest in report['checkpoint_sha256'].items():
    assert file_sha256(base / name / 'ckpt_final.pt') == digest
resumed = torch.load(base / 'resumed100/ckpt_final.pt', map_location='cpu', weights_only=True)
continuous = torch.load(base / 'continuous100/ckpt_final.pt', map_location='cpu', weights_only=True)

def equal(a, b):
    if torch.is_tensor(b):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(b, dict):
        assert a.keys() == b.keys()
        for key in b:
            equal(a[key], b[key])
    elif isinstance(b, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b

for key in ('model_state_dict', 'optimizer', 'sampler_state', 'rng_state'):
    equal(resumed[key], continuous[key])
print('PASS：100步checkpoint摘要一致，恢复与连续训练的权重/优化器/采样器/RNG完全一致')
PY
```

## 训练命令参考（复现前明确选择新目录）

现有目录已验收，以下展示本轮训练/恢复参数；直接重复会被不覆盖保护拒绝。
本轮实际使用logs中的封装另外保存resources.json；下面函数演示相同确定性设置与训练调用。

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
short_run() {
  python -u -c 'import sys, torch; torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; from minigpt_cn.training.train import main; main(sys.argv[1:])' "$@"
}
short_run --config configs/minigpt_33m_short_run.yaml \
  --out-dir checkpoints/m3_33m_short_v1/first50 --stop-after 50
short_run --config configs/minigpt_33m_short_run.yaml \
  --out-dir checkpoints/m3_33m_short_v1/resumed100 \
  --resume checkpoints/m3_33m_short_v1/first50/ckpt_final.pt
short_run --config configs/minigpt_33m_short_run.yaml \
  --out-dir checkpoints/m3_33m_short_v1/continuous100
```

stop-after保持原max_steps=100与lr_decay_steps=100；50步退出不会重设调度曲线。
续训使用新目录，从已完成50步开始，不重复前50步，也不更改batch、精度或其他超参数。

## 本轮代码检查

实际定向CPU测试：tests/test_training.py与tests/test_checkpoint.py，
Windows34通过/2跳过（9.08秒），WSL34通过/2跳过（5.58秒）；跳过项为两个CUDA小模型测试。
本轮真实33M三段GPU训练与独立重载验证已执行，不把CPU跳过项算成GPU测试通过。
两边Ruff通过；未重跑完整CPU回归。源码仅纠正训练run.json的耗时范围文字，未改模型或优化器计算。

本轮停在33M本地短训练/恢复验收，正式语料、AutoDL作业、预训练和生成展示没有在本轮启动。
