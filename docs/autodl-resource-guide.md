# AutoDL单卡500步资源验证

本轮只验证实际云端设备的显存、吞吐和500步稳定性，使用已验收pilot数据。
500步资源报告与下载原始产物均已独立核对通过，用户累计费用3.09元及保留实例运行决定已记录。
本轮已收尾；正式准备见[正式预训练方案](formal-pretraining-plan.md)。
下一任务以 [current-work.md](current-work.md) 为准，本页保留完整操作步骤。

## 1. 本地已准备的内容

- 配置：configs/minigpt_33m_autodl_500.yaml，引用唯一33M模型配置，context512、batch16、BF16/FP32、500步。
- 调度：warmup20、lr_decay500；每100步完整valid，退出时保存final。
- 输入：同版本16000词表tokenizer及固定train/valid/test uint16，来源/摘要保持不变。
- 新训练从随机权重开始；上传包不包含本地checkpoint，不跨PyTorch版本假装精确续训。
- 上传包：logs/autodl_500_prepare_v2/minigpt-cn-autodl-500-v2.tar.gz。
- 交付清单：同目录delivery_manifest.json，记录压缩包字节数、SHA-256和包内逐文件摘要。
- 包内入口：run_autodl_500.py；--preflight-only只检查文件、数据及CUDA/BF16环境。

上传包由当前工作区源码生成，包含必要源码/配置/CPU测试、token数据和验收记录。
两份pilot YAML仅作为既有CPU路径解析测试的配置夹具，500步入口固定使用33M/16K配置。
不包含.git、虚拟环境、旧权重、原始下载、历史教程备份或无关未跟踪文件。产物Git忽略。
batch16已在本次RTX4080 SUPER实例完成500步；结果仅适用于报告记录的设备、配置与框架。
v2压缩包保持原始字节和摘要不变。包内说明保留打包时的4090候选信息，当前设备/价格以本页为准；
运行入口按实际GPU记录环境，不要求型号必须是4090，不需改包内配置。

## 2. 已租用实例与硬件核对

用户2026-10-13已租用按量计费单卡vGPU-32GB实例，页面报价1.58元/时。
用户回传：实际GPU为NVIDIA GeForce RTX4080 SUPER、31.48GiB、计算能力8.9；
Python3.12.3、PyTorch2.8.0+cu128、CUDA runtime12.8，CUDA/BF16支持与小矩阵运算通过。
驱动595.71.05、nvidia-smi显示CUDA13.2为驱动兼容上限，不替代镜像的CUDA runtime12.8。
不能将本机测量记成4090结果；本次预算上限和实际账单另由用户记录。
保留默认磁盘容量即可容纳本批小包和checkpoint，无需为这次验证扩大数据盘。
本轮不选择长租套餐或多卡。

优先使用现成PyTorch镜像。官方当前列有PyTorch2.8.0/Python3.12/CUDA12.8，
实际主机镜像列表为准；不要把主机的CUDA兼容上限当成镜像中torch.version.cuda的实际值。
出处：[官方镜像/环境列表](https://www.autodl.com/docs/base_config/)。

实例进入运行中即开始计费，按量费用按开关机时间计算，并非仅算GPU训练时间；
即使训练程序已经结束，仍需在控制台关机停止实例GPU计费。
按页面1.58元/时，单纯开机30分钟对应0.79元；这是时间乘单价的例子，不是本次费用承诺。
出处：[官方计费说明](https://api.autodl.com/docs/price/)。

## 3. 上传并解压到数据盘

创建好实例后，打开JupyterLab，进入autodl-tmp目录，上传压缩包。
打开Terminal执行下列命令。先把SHA-256替换为本地delivery_manifest.json中的archive_sha256；
核对成功后才解压。项目目录必须是新目录，已有目录先检查，不覆盖。

```bash
cd /root/autodl-tmp
sha256sum minigpt-cn-autodl-500-v2.tar.gz
# 与本地交付清单逐字比对；不一致就停止，重新上传，不继续解压。
test ! -e minigpt-cn-autodl-500-v2 && test ! -L minigpt-cn-autodl-500-v2 && \
  tar --keep-old-files -xzf minigpt-cn-autodl-500-v2.tar.gz
cd /root/autodl-tmp/minigpt-cn-autodl-500-v2
```

JupyterLab默认工作目录为/root，数据盘为/root/autodl-tmp。
出处：[官方上传/终端入口](https://www.autodl.com/docs/quick_start/)、
[目录说明](https://api.autodl.com/docs/env/)。

## 4. 用户创建项目环境并安装缺少的依赖

用镜像已有的Python创建项目虚拟环境，复用预装PyTorch；不要先重装Torch或单独下载CUDA。
下载/安装由用户执行。若.venv已经存在，先检查它，跳过创建命令，不重建已有环境。

```bash
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
# 仅首次、.venv不存在时执行：
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --disable-pip-version-check \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30 --retries 1 \
  'numpy>=1.26' 'PyYAML>=6.0' 'regex>=2024.0' 'OpenCC==1.4.2' 'pytest>=8.0' 'ruff>=0.6'
export PYTHONPATH="$PWD/src"
```

这些是当前训练入口和所选CPU测试需要的依赖；不运行冻结的数据流水线。
预装Torch会从镜像继承，依赖满足时pip不会主动升级它。实际torch/CUDA/Python版本仍需记录。
本机官方源访问等待过长，用户改用清华镜像后安装成功；下载仍在增长时继续等待，
不再套用曾截断Ruff下载的180秒总上限。镜像只在本次命令指定，不修改全局pip配置。
出处：[清华PyPI镜像使用说明](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)。

## 5. 先检查上传和环境，再跑CPU检查

```bash
python run_autodl_500.py --preflight-only
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_training.py tests/test_checkpoint.py tests/test_tokenizer.py -q --tb=short
python -m ruff check src tests
```

预检查会核对包内逐文件摘要、16K词表/merges/EOS、三集合摘要、实际CUDA/BF16、GPU名与显存。
确认打印出的GPU仍为RTX4080 SUPER、显存约31.48GiB；CUDA/BF16不可用、数据版本不符或CPU检查失败时停止，
回传具体报错，不开始500步。--preflight-only不会更新模型。
2026-10-13用户回传上述预检查PASS，43 passed/2 skipped（3.53秒），Ruff通过。
两项跳过来自隐藏CUDA后的GPU测试；此处通过不能代替实际500步运行。

## 6. 已完成运行的启动方式（复现参考）

先确认实际计费单价和预算。下面1.58按本次页面报价填写；价格改变就替换变量。
本次运行已经完成，当前执行第7节归档收尾；本节保留当时的启动方式，不再次执行。
输出目录checkpoints/autodl_500_v1必须不存在，日志同样不覆盖。每次失败的诊断产物保留。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp/minigpt-cn-autodl-500-v2
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
MINIGPT_PROBE_PRICE_YUAN=1.58
for path in checkpoints/autodl_500_v1 logs/autodl_500_v1.stdout.log; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "已有输出，先检查，不重复启动：$path"
    exit 1
  fi
done
mkdir -p logs
(
  set -o noclobber
  nohup python -u run_autodl_500.py --hourly-price "$MINIGPT_PROBE_PRICE_YUAN" \
    > logs/autodl_500_v1.stdout.log 2>&1 < /dev/null &
  echo "训练进程PID=$!"
)
SH
tail -n 30 -f /root/autodl-tmp/minigpt-cn-autodl-500-v2/logs/autodl_500_v1.stdout.log
```

nohup让训练不依赖当前终端连接；Ctrl+C停止tail监看，不会停止后台训练。
验证只从train.bin取batch，valid.bin用于完整验证，test.bin仅保持来源版本，不用于训练或调参。
batch16每步8192 tokens，500步4096000 tokens；本批仍是pilot闭环，不计作正式预训练规模。
发生OOM、NaN/Inf或版本错误就保留日志并停止；先修实际阻塞，再明确新配置/目录，不盲目重跑。

## 7. 验收、取回结果与关机

预期输出：checkpoints/autodl_500_v1中的run.json、metrics.jsonl、ckpt_final.pt、resources.json。
只有程序完成500步并写出resources.json才有完整资源结果；预检查PASS不等于500步通过。
resources.json记录模型/数据/配置版本、实际GPU与框架、完整valid覆盖、checkpoint摘要、
两种吞吐范围、峰值分配/保留显存及用户输入的小时价格，不伪造控制台实际账单。

训练区间吞吐取热身后的相邻train日志区间，排除夹有验证的区间；本轮这些区间无checkpoint。
循环吞吐包含训练、完整验证、最终保存；PyTorch显存指标不包含其他进程或驱动占用。

取回至少resources.json、metrics.jsonl、run.json和stdout日志；checkpoint按需要同时取回保留。
本次按完整目录取回checkpoints/autodl_500_v1中的四份文件；在PC端
logs/autodl_500_v1中直接保存resources.json、metrics.jsonl、run.json、ckpt_final.pt和stdout日志。
用户执行下载；已有文件先检查，不覆盖。先检查原始报告和日志，再核对取回的checkpoint：

```powershell
Get-FileHash -LiteralPath D:\aiworkspace\transformer\logs\autodl_500_v1\ckpt_final.pt -Algorithm SHA256
```

应与云端报告的4d976c1407fa1d9b0fd9914bc6dfe93bcfac7d6203b2ab82a11fc6ddeb595540一致。
2026-10-13已实际读取PC端397351864字节的checkpoint，独立摘要与云端报告一致；五份原始文件已齐全。
记录控制台实际GPU、单价、当前费用和实际实例状态。
本次用户明确要求暂不关机以保留稀缺主机并继续主线，实例保持运行；关机不是本次验收条件。
累计费用3.09元仅为用户反馈时点，后续继续计费。收尾记录在logs/autodl_500_v1/closeout.json。
不自行停止/释放实例或删除数据，用户保留实例的决定持续有效，直到用户另行改变。

用户确认本机JupyterLab界面无法进入checkpoints目录，Terminal能正常列出四份产物。
Jupyter官方[Issue #5569](https://github.com/jupyter/notebook/issues/5569)记录了部分版本中
checkpoints目录名与Contents服务的checkpoint接口路径冲突，可能使文件浏览器无法打开该目录。
这是接口路径冲突的已知问题，不能泛称checkpoints是Linux系统保留关键字；本机具体服务版本未核对。
已回传目录权限为drwxr-xr-x，本轮不改权限或重命名已有产物目录，继续通过Terminal归档。
[JupyterLab官方文件操作](https://jupyterlab.readthedocs.io/en/stable/user/files.html)说明默认双击进入目录、右键文件下载。
在云端Terminal执行下面的Python标准库打包，不安装工具、不改原始产物：

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import zipfile

root = Path("/root/autodl-tmp/minigpt-cn-autodl-500-v2")
files = [
    root / "checkpoints/autodl_500_v1/resources.json",
    root / "checkpoints/autodl_500_v1/metrics.jsonl",
    root / "checkpoints/autodl_500_v1/run.json",
    root / "checkpoints/autodl_500_v1/ckpt_final.pt",
    root / "logs/autodl_500_v1.stdout.log",
]
for path in files:
    if not path.is_file():
        raise FileNotFoundError(f"缺少产物：{path}")

output = root / "logs/autodl_500_v1_download.zip"
with zipfile.ZipFile(output, mode="x", compression=zipfile.ZIP_STORED) as archive:
    for path in files:
        archive.write(path, arcname=path.name)
with zipfile.ZipFile(output, mode="r") as archive:
    bad = archive.testzip()
    if bad is not None:
        raise RuntimeError(f"打包校验失败：{bad}")
with output.open("rb") as handle:
    digest = hashlib.file_digest(handle, "sha256").hexdigest()
print(f"打包通过：{output}")
print(f"ZIP SHA-256：{digest}")
PY
```

ZIP使用独占创建，不覆盖已有包；不压缩以减少打包耗时，并读回校验CRC。
它包含五份原始文件，生成在项目logs目录，约379MiB。刷新该目录后右键
“autodl_500_v1_download.zip”选择Download，下载到PC的logs/autodl_500_v1中。
包内是五个文件名，用户可以直接解压到该目录；已有同名文件先检查，不覆盖。
也可只将ZIP放到该目录，通知助手按ZIP摘要和原始checkpoint摘要核对/解包。
打包通过不等于用户已下载；实际下载仍由用户执行。


## 8. 本次用户回传报告的核对结果

2026-10-13用户在聊天中贴回resources.json，报告核对PASS。
本轮重新计算本地配置/传输清单/tokenizer/token数据清单摘要，四项与报告一致；
也复算训练tokens、两种吞吐、显存范围、完整valid目标数和exp(loss)对应的PPL。

| 指标 | 本次结果 |
| --- | ---: |
| 实际GPU / 总显存 | RTX4080 SUPER / 31.475GiB |
| PyTorch / CUDA runtime | 2.8.0+cu128 / 12.8 |
| 参数 / batch / context / 精度 | 33104384 / 16 / 512 / BF16 |
| 已完成步数 / 训练tokens | 500 / 4096000 |
| 程序总调用 / 循环耗时 | 117.915秒 / 116.565秒 |
| 训练区间吞吐 / 循环吞吐 | 37969 / 35139 tokens/s |
| PyTorch峰值分配 / 保留显存 | 5.804 / 6.242GiB |
| 完整valid目标数 / 最终loss / PPL | 25271 / 6.701680 / 813.7717 |
| 页面报价 | 1.58元/时 |

完整valid loss（步0/100/200/300/400/500）：
9.784121 / 7.712046 / 7.198945 / 6.928288 / 6.783216 / 6.701680。
程序完成500步且完整验证loss均为有限值；这支持本配置的短运行资源与稳定性验收，
本批仍是pilot语料，不能据此宣称正式预训练效果或长期资源保证。

训练区间只测热身后相邻train日志区间，共3604480 tokens/94.932秒；
循环包含完整验证和最终checkpoint保存，启动前检查不计入循环。
显存只统计当前进程PyTorch分配器，不含驱动和其他进程。
cpu_threads_visible=128是系统可见CPU数量，不代表平台给本实例分配128vCPU；页面配置为16vCPU。

报告副本与核对记录在Git忽略的logs/autodl_500_report_review_v1；
resources.user_report.json是贴文的规范化副本，不是下载的原始云端报告。
用户随后将五份原始产物下载到logs/autodl_500_v1，本地19项独立核对均PASS：
五份文件摘要、原始报告与聊天报告一致、配置与实际输入摘要、随机权重来源、完整日志覆盖，
学习率调度、两种吞吐复算、stdout与metrics一致、checkpoint CPU重载和共享参数、
500步及有限FP32优化器矩状态、数据与保存环境一致。
50条训练记录和步0/100/200/300/400/500的六次完整valid均齐全，stdout无异常栈。
核对记录在logs/autodl_500_v1/artifact_acceptance.json；一次性读取脚本在
logs/autodl_500_report_review_v1/verify_downloaded.py，均Git忽略。
Windows PyTorch2.14.0+cu126仅用于CPU读取验收；云端训练版本仍为2.8.0+cu128。
未恢复CUDA RNG、未跨版本精确续训、未重跑验证集/CPU测试/500步或正式训练。
用户已回传累计费用3.09元及继续运行决定；500步任务已完成验收。
artifact_acceptance.json中费用/关机pending字段保留当时快照，closeout.json补充后续用户反馈。
新阶段总预算50元，保守包含这3.09元及后续保留实例时间；费用不是核心训练1.97分钟乘单价的固定值。
