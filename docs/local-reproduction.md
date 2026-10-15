# MiniGPT-CN 33M 本地复现

本页对应[模型卡](../MODEL_CARD.md)中的原12208步final，使用现有生成CLI。目标是确认同版权重/tokenizer可在本地加载、生成并保存原始ID与设置；不启动训练或读取test。

## 必需产物与环境

- 项目已有Windows `.venv`，含项目及pyproject.toml所列依赖；本轮不下载、安装或重建环境。
- 原权重：`logs/m3_training_report/m3_pretrain_100m_return_v1/ckpt_final.pt`，397351864字节。
- 同版tokenizer：`data/processed/pilot_tokenizer_16k_v1.json`，1142324字节。
- CPU用float32；CUDA/BF16需要实际设备和PyTorch支持。已验收的本机为Python3.12.4/PyTorch2.14.0+cu126、RTX4060 Laptop。

代码新克隆不含这些Git忽略产物；缺失时先取得准确版本，不用旧2048 tokenizer、随机权重或重新训练的16K词表替代。推理不依赖train/valid/test bin或原云端绝对路径。若文件已移动，只修改下面本地路径并维持摘要一致。

## Windows核对

在PowerShell执行；以下为直接粘贴的命令，不依赖旧收尾ps1脚本：

```powershell
Set-Location -LiteralPath 'D:\aiworkspace\transformer'
$m4Python = '.\.venv\Scripts\python.exe'
$m4Checkpoint = '.\logs\m3_training_report\m3_pretrain_100m_return_v1\ckpt_final.pt'
$m4Tokenizer = '.\data\processed\pilot_tokenizer_16k_v1.json'
foreach ($m4Path in @($m4Python, $m4Checkpoint, $m4Tokenizer)) {
    if (-not (Test-Path -LiteralPath $m4Path -PathType Leaf)) { throw "缺少本地文件：$m4Path" }
}
& $m4Python -B -c "import sys; from minigpt_cn.training.checkpoint import file_sha256; expected=('f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d','21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470'); actual=tuple(file_sha256(p) for p in sys.argv[1:]); assert actual==expected, actual; print('SHA256 PASS')" $m4Checkpoint $m4Tokenizer
if ($LASTEXITCODE -ne 0) { throw '产物摘要核对失败，停止' }
& $m4Python -B -c "import platform,torch; print(platform.python_version()); print(torch.__version__); print('CUDA:',torch.cuda.is_available())"
```

摘要不符或缺文件即停止，不改旧数据。生成输出已存在时CLI会拒绝覆盖；已有结果先核对，不反复运行同一命令。

## CPU最小复现

先用8个greedy新token验证加载、推理和输出链路。它用于命令复现检查，不代表完整评估或生成质量通过：

```powershell
& $m4Python -B -m minigpt_cn.inference.generate `
  --checkpoint $m4Checkpoint --tokenizer $m4Tokenizer `
  --prompt '北京是中国的' --max-new-tokens 8 --greedy `
  --device cpu --precision float32 --seed 42 --repetition-penalty 1.0 `
  --output 'logs/m4_local_reproduction_v1/cpu_smoke.json'
```

应显示checkpoint iteration=12208、模式greedy、缓存关闭；JSON中checkpoint_sha256应与上述摘要相同，generated_ids不超过8个、EOS ID256时提前停止。
默认生产行为仍为repetition_penalty=1.0；本例显式写出。CPU FP32的短输出与M3 CUDA/BF16的128-token固定样例不作为逐字一致性对照。

## CUDA可选续写

在已有CUDA环境，先检查BF16支持；前述Windows变量在同一PowerShell会话保留：

```powershell
& $m4Python -B -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'; print(torch.cuda.get_device_name(0))"
& $m4Python -B -m minigpt_cn.inference.generate `
  --checkpoint $m4Checkpoint --tokenizer $m4Tokenizer `
  --prompt '北京是中国的' --max-new-tokens 128 `
  --device cuda --precision bfloat16 --seed 42 `
  --temperature 0.8 --top-p 0.9 --repetition-penalty 1.0 `
  --output 'logs/m4_local_reproduction_v1/cuda_sampling.json'
```

它沿用原评估中该提示词的采样设置，不挑种子或尝试更好参数。没有--use-cache即无缓存；可选缓存及滑窗策略见[运行参考](runtime-reference.md)。默认示例保持原评估方式，不新增缓存性能结论。
已有相同环境的原12条样例见[M3报告](m3-experiment-report.md#固定续写与质量结论)；报告保存全部6个提示词及greedy/sampling结果。用户自由输入的新前缀只属于本地试用，不能替代固定评估或挑成总体质量证明。

## 读回结果与复现边界

```powershell
$m4Result = Get-Content -LiteralPath 'logs/m4_local_reproduction_v1/cpu_smoke.json' -Raw -Encoding UTF8 | ConvertFrom-Json
$m4Result | Select-Object iteration, device, precision, decoding, use_cache, generated_tokens, stop_reason, checkpoint_sha256
$m4Result.continuation
```

JSON同时记录prompt_ids、generated_ids、full_ids、EOS/停止原因、解码参数和UTF-8状态；保留这些字段比只截图文本更便于核对。输出在Git忽略的logs目录，不提交权重、数据或运行日志。
相同seed仅固定当前采样发生器；跨框架版本、精度、设备或内核不承诺逐字相同。checkpoint可以跨环境安全读取做推理，训练精确恢复仍要求原框架版本、配置、数据与路径契约。
本页没有运行完整valid/test，也不把CPU烟测新增输出混入旧12条评估。已有完整指标来自模型卡引用的原始报告；M3质量限制与本地命令是否可执行分别记录。

## 本轮文档验收

本轮核对模型/词表摘要、CLI参数和文档链接，并执行上面的CPU最小复现；检查记录在`logs/m4_model_card_v1/acceptance.json`。Windows PowerShell5.1子进程首次核对因Get-FileHash命令不可用而在推理前停止，现统一使用已有项目Python核对摘要，不依赖该cmdlet；失败日志保留。CUDA长续写命令在本轮仅核对参数和语法，不自动执行。首次复现已有输出时不覆盖，先查看已有结果。
M3收尾轮完整CPU回归Windows736 passed/10 skipped、WSL741 passed/5 skipped及两边Ruff通过是历史结果，不冒称本轮重跑。随后用户已授权完成M4整体展示，见[项目展示](showcase.md)和[用户验收](m4-acceptance.md)。本页命令块保持原样，M4终验只读回已有CPU结果，不重复生成或执行CUDA长续写。
