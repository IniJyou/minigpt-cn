# 33M首轮有界中文训练准备

当前入口为[current-work.md](current-work.md)。本轮准备固定运行配置、范围记录和上传包，
准备轮未启动33M训练；用户已回传云端预检查PASS，并明确授权下一项启动2000步。
候选编码已验收，见[实际编码结果](candidate-encoding-guide.md#实际云端结果与本机验收)。

## 固定运行范围

仅使用/root/autodl-tmp/m3_candidate_tokens_16k_v1中的6529篇已编码子集，
保留原始候选ready_for_training=false和全部来源元数据。
本轮范围记录training_scope.json只限定该批输入上的2000步中文实验，
不是对整个分片、其他候选或旧quarantine的放行，不宣称正式预训练效果或逐文档许可审批。

| 项目 | 固定值 |
| --- | --- |
| 模型 | 原minigpt_33m.yaml，33104384参数，随机权重起点 |
| tokenizer | 实际16000词表，15743 merges，EOS256，原已验收版本 |
| train/valid/test token位置 | 4449461 / 46761 / 53483 |
| 数据使用 | 只从train采训练窗口；完整valid验证；test保留给训练后评测 |
| batch / context / 精度 | 16 / 512 / CUDA BF16前向，FP32权重和优化器 |
| 更新数 / 处理tokens | 2000 / 16384000 |
| 曝光量 | 约3.682倍算术比值，随机窗口采样，不是固定epoch或唯一语料规模 |
| 学习率 | 3e-4，预热100步，余弦衰减终点2000，最低3e-5 |
| AdamW / 裁剪 | betas 0.9/0.95，eps 1e-8，weight_decay 0.1，梯度范数上限1.0 |
| 日志 / 完整验证 / 保存 | 每20 / 250 / 500步；初始和最终也做完整valid |
| 完整valid目标数 | 每次46760，覆盖尾块，按目标数加权 |
| 云端框架 | 原Python3.12 / PyTorch2.8.0+cu128 / RTX4080 SUPER单卡 |
| 输出 | /root/autodl-tmp/m3_training_runs/first_2000_v1，新目录 |

不导入旧2048权重、本地短训练权重或500步pilot checkpoint，不自动扩至1亿/2亿tokens。
固定seed42、确定性算法开启、TF32关闭，复用资源验证设置。
原核心训练入口和数据工具不改动；新小脚本先核对已验收编码根清单、全部产物摘要、
实际tokenizer、配置路径及37个源码摘要，只有显式--run才调用既有train函数。

原1亿计划在本批相当于约22.476倍曝光，不作为本次默认运行规模。
按此前37969 tokens/s估计核心训练约431.51秒（7.19分钟），不是本轮实测或费用承诺；
完整验证、保存、启动和保留实例的时间另计。用户50元总预算及保留实例决定不变，不再询问账单。

## 本轮上传与只预检查

本地包：logs/m3_first_train_prepare_20261014_v1/minigpt-cn-m3-first-train-v1.zip。
10789字节，SHA-256：0aad983605b0d4f45b81a08d365c493608926e215bcd106c62c3821c44b5bfb8。
包内6个文件：运行脚本、运行YAML、原模型YAML、training_scope.json、代码许可证、传输清单。
不传语料/bin/tokenizer，不下载或安装依赖，不覆盖原云端项目。

用户上传ZIP到/root/autodl-tmp后，执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '0aad983605b0d4f45b81a08d365c493608926e215bcd106c62c3821c44b5bfb8' \
  'minigpt-cn-m3-first-train-v1.zip' | sha256sum -c -
if [ -e m3_first_train_v1 ] || [ -L m3_first_train_v1 ]; then
  echo '已有准备目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-first-train-v1.zip .
cd m3_first_train_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 5m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python \
  run_first_train.py --preflight-only
SH
```

预检查只读输入和查询GPU/磁盘，不创建训练目录或模型；默认不带--run也只做预检查。
检查至少3GiB剩余磁盘，预留5份约379MiB checkpoint及日志；不足只报告，不删除文件。
已有准备目录时先核对其transfer_manifest.json，直接从目录执行相同预检查命令，不重复解包。
已有训练输出时拒绝重训；先检查其中checkpoint/日志，再按实际状态处理恢复，不改名绕过检查。
用户贴回预检查JSON和末尾“PASS：首轮训练准备检查；未启动训练”，即可验收云端准备。

## 后续训练启动步骤

云端预检查已通过；配置/范围/编码清单摘要与本机一致，训练设置、规模及磁盘空间核对通过。
用户已按以下显式--run命令完成2000步，步骤留作复现依据；已有输出，不重跑：

```bash
bash -euo pipefail <<'SH'
cd /root/autodl-tmp/m3_first_train_v1
if [ -e first_2000_v1.stdout.log ] || [ -L first_2000_v1.stdout.log ]; then
  echo '已有训练stdout，先检查，不覆盖'
  exit 1
fi
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 20m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u \
  run_first_train.py --run 2>&1 | tee first_2000_v1.stdout.log
SH
```

新运行每次都会重新核对输入/配置/环境。训练最多20分钟；异常或超时保留诊断和已保存checkpoint，
没有training_report.json时不能冒称本次完整收尾，不自动删除、覆盖或重启。
正常会产生step_000500.pt、step_001000.pt、step_001500.pt、step_002000.pt和ckpt_final.pt；
既有核心在2000步周期保存后还保存final，两份均保留，不为了去重修改训练核心。
validation出现在0/250/500/750/1000/1250/1500/1750/2000共9个时点。
最优日志步可能没有对应checkpoint；报告会明确记录，不能把最终权重自动称为最优权重。

训练报告记录输入/配置/源码范围、随机起点、参数量、实际处理token数、完整valid、耗时、
本进程PyTorch显存峰值及全部checkpoint摘要，不自动延长训练。
训练后先返回training_report.json、run.json、metrics.jsonl、training_scope.json和stdout；
随后按验收需要取回实际checkpoint，不能只看loss下降就宣称M3完成。

## 本轮本地验证

2026-10-14 Windows默认设备定向42 passed（11.88秒）；另显式隐藏CUDA做CPU定向41 passed/1 skipped（8.18秒），
跳过项为CUDA恢复检查。覆盖配置加载、真实小模型训练/恢复回归、
损坏文件/清单、pilot路径混用、超出固定步数/调度、已有输出拒绝，以及默认预检查不得启动训练。
本地ZIP逐文件读回通过，37个源码文件与原500步云端包摘要一致；Ruff通过。
没有重跑完整CPU回归或WSL检查，准备轮没有执行本批33M训练。
用户随后已回传实际云端预检查PASS；6项本机报告核对PASS，记录在准备目录cloud_preflight_acceptance.json。
该验收记录依据用户贴文，非下载原始报告；用户随后完成实际2000步并回传报告，结果如下。

## 实际2000步结果与产物取回

2026-10-14用户回传completed；logs/m3_first_train_2000_report_review_v1中保存贴文规范化副本及24项核对PASS记录。
核对配置/范围/编码清单摘要、随机起点、同版本框架、参数量、步数/token规模、完整valid目标、PPL复算、最优步和5份checkpoint报告字段。
该报告核对轮尚未取回原始日志或权重；用户随后已下载六份文件，本机实际重算摘要并CPU重载，结果见本页末。

| 实际项 | 结果 |
| --- | --- |
| 参数 / 处理训练tokens | 33104384 / 16384000 |
| 唯一train token位置 / 算术曝光比 | 4449461 / 3.68224 |
| 完整valid loss / PPL | 9.777378→5.621109 / 最终276.195558 |
| 完整valid覆盖 | 9次，每次46760目标 |
| 最佳日志验证点 | 2000步，有对应保存点；未进行额外训练 |
| 调用 / 循环耗时 | 471.008 / 469.674秒 |
| 循环吞吐 | 34883.806 tokens/s，包含验证和周期/最终保存，不是纯训练吞吐 |
| PyTorch峰值分配 / 保留显存 | 5.804 / 6.242GiB，不含驱动/其他进程 |
| final字节 / SHA-256 | 397351864 / db1034589ab6a30d514993112642622076d2e92963516e0d05818c304a52a10f |

训练loss改善证明本轮学习与验证链路正常，不代替独立test、生成检查、正式预训练效果或M3收尾。
不因仍有验证改善而自动延长训练；本轮只完成原始产物验收。

用户取回6个文件到PC的D:\\aiworkspace\\transformer\\logs\\m3_first_train_2000_v1：

- 云端/root/autodl-tmp/m3_training_runs/first_2000_v1中的training_report.json、run.json、metrics.jsonl、training_scope.json、ckpt_final.pt。
- 云端/root/autodl-tmp/m3_first_train_v1/first_2000_v1.stdout.log。

可在云端终端一次打包，只包含上述6份，四个周期checkpoint保留云端：

```bash
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

base = Path('/root/autodl-tmp')
run = base / 'm3_training_runs/first_2000_v1'
files = [run / name for name in (
    'training_report.json', 'run.json', 'metrics.jsonl',
    'training_scope.json', 'ckpt_final.pt')]
files.append(base / 'm3_first_train_v1/first_2000_v1.stdout.log')
for path in files:
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f'缺少普通文件：{path}')
report = json.loads(files[0].read_text(encoding='utf-8'))
with (run / 'ckpt_final.pt').open('rb') as handle:
    digest = hashlib.file_digest(handle, 'sha256').hexdigest()
if digest != report['checkpoints']['ckpt_final.pt']['sha256']:
    raise SystemExit('final摘要不符，不打包')
archive = base / 'first_2000_v1_return.zip'
with ZipFile(archive, 'x', compression=ZIP_STORED) as packet:
    for path in files:
        packet.write(path, arcname=path.name)
print(f'PASS：6份产物已打包到 {archive}；final SHA-256={digest}')
PY
```

ZIP以新建模式打开，已有包拒绝覆盖；不删除任何原始产物。使用ZIP_STORED避免额外压缩等待。
从JupyterLab下载/root/autodl-tmp/first_2000_v1_return.zip，在PC新目录解压；收到原始产物后再核对日志/摘要和CPU重载权重。

## 原始产物实际验收

2026-10-14用户已取回六份原始文件至logs/m3_first_train_2000_v1；verify_artifacts.py实际执行27项验收PASS，
记录在同目录artifact_acceptance.json。六份文件均保留未改写，原始报告与用户贴文结构完全一致。

- final实际397351864字节，SHA-256=db1034589ab6a30d514993112642622076d2e92963516e0d05818c304a52a10f，与云端报告一致。
- training_scope原始字节摘要、运行配置/编码版本、记录的train/valid/tokenizer摘要与先前验收输入一致。
- 100条train、9次完整valid及finished齐全，数值有限、时间递增；学习率按step-1独立复算一致，完整valid每次46760目标。
- stdout的预检查、全部metrics与最终报告可精确对应；PPL复算、最终loss/步数、处理tokens/曝光比一致。
- 通过项目加载器实际CPU重载：2000步、33104384唯一参数、共享embedding/LM head、FP32参数有限，同版16K tokenizer匹配。
- 74份优化器参数状态无重复共享参数、顺序正确；均2000步，FP32矩形状正确且有限，二阶矩非负，最终学习率/AdamW设置一致。
- checkpoint记录的云端PyTorch2.8.0+cu128、输入摘要、随机起点一致；保存的CPU采样器/RNG状态可读。

本机PyTorch2.14.0+cu126仅CPU读取，未恢复CUDA RNG或执行精确续训、前向验证、生成或训练；
不能把读取验收当成跨框架版本精确恢复。四个周期checkpoint未取回/重载，云端token bin未在本机重新计算摘要。
本轮Ruff通过，没有重跑完整CPU回归或WSL检查；27项是实际原始文件验收，准备轮pytest数字仍保留其原范围。
本轮训练任务收尾。唯一下一项为原云端环境独立重载final后的valid/test评估及固定提示词生成，
保存评估产物，不自动延长训练；有界中文实验不能冒称1亿唯一语料或已完成正式预训练效果/M3。
