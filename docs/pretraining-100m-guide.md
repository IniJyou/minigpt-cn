# 实际48M训练输入冻结与100M运行准备

2026-10-14。用户接受按实际约4819万train位置继续。单项交付为输入冻结、固定运行配置与预检查包；实际模型训练留在冻结原始报告验收之后。
本阶段是有界33M中文文本预训练实验，不承诺指令遵循、事实准确性或展示效果；原候选状态及逐篇质量/许可限制保持。

## 固定输入与处理方法

来源为已验收批次3：train/valid/test=48202076/297390/251517个token、73528/461/450篇。
8条确切暂缓全部train，按ID/标题/URL/规范化正文和原文摘要匹配剔除：
5575868/2954494/4069965/8098260/7155618/3905452/5445840/6239167。
准确删除8篇及14082个token，冻结train=48187994/73520篇；用户已实际运行并回传报告，验收见本页末尾。

不重新编码全部正文。按原uint16磁盘EOS+LF结束位置准确截取文档，保留所有未暂缓JSONL原行/元数据与token字节顺序。
valid/test JSONL与bin逐字节复制并核对原摘要；不挪集合、不做二次NFC/t2s、不修改姓名、不使用test训练。
原input/JSONL/bin、pilot/tokenizer/权重和旧报告只读。旧346/303 heldout按ID、规范化正文和原文摘要隔离，新valid/test461/450固定保留。
保存后对全部三集合逐篇磁盘全文解码，精确核对原正文+EOS+LF；所有输入摘要再次核对后才保存根manifest。
新冻结记录继续ready_for_training=false，input_freeze_checks_passed只表示本轮技术完整性；明确的实验运行范围另存training_scope.json，不伪造逐文档审批。

## 固定运行配置

配置configs/minigpt_33m_100m.yaml，复用原33M模型和同版16K tokenizer。

| 项目 | 值 |
| --- | --- |
| 模型 | 33104384参数，vocab16000/context512，原共享权重/dropout0.1配置 |
| 起点 | 随机权重，不导入pilot或first_2000 checkpoint |
| batch/precision/seed | 16 / BF16前向、FP32权重与优化器 / 42 |
| max_steps | 12208 |
| 处理tokens | 12208×16×512=100007936 |
| 数据曝光 | 约2.07537倍算术位置曝光，随机窗口，不是实际epoch |
| LR | 3e-4→3e-5，预热500步，余弦终点12208 |
| AdamW | betas0.9/0.95、eps1e-8、weight_decay0.1、梯度裁剪1.0 |
| 日志/完整valid/保存 | 20 / 500 / 2000步，最终12208另作完整valid与保存 |

沿用实测batch/模型/优化器和学习率，调度终点与间隔按固定100M范围制定，不依据test分数调整。
共26次完整valid（0、500…12000、12208），每次297389目标；6个周期checkpoint和final均保留，不自动清理或延长训练。
原5000万语料规划未达到，使用实际48187994位置，不宣称1亿唯一语料或已有训练效果。
50元总预算继续包含历史支出与保留实例费用；按用户要求不再询问账单，由用户告知超额。

## 用户上传并完成输入冻结和预检查

小包：logs/m3_pretraining_prepare_20261014_v1/minigpt-cn-m3-pretrain-100m-v1.zip，50016字节。
SHA-256=28fd4acbd29c838013b61f8e6494dbd850c51183f43b1de4e6e0640a4b456146。
包只有本轮接入脚本、既有冻结函数副本、原模型与新运行配置、审阅/范围/绑定/许可证，无新依赖、语料或权重上传。
绑定62份既有云端输入/源码；不重读原Parquet正文、不下载补救、不安装框架。
保持Python3.12/PyTorch2.8.0+cu128/CUDA12.8、单卡RTX4080 SUPER/BF16；空闲显存至少8GiB、数据盘6GiB以覆盖已知显存与checkpoint保存需求。

用户上传ZIP至/root/autodl-tmp，执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '28fd4acbd29c838013b61f8e6494dbd850c51183f43b1de4e6e0640a4b456146' \
  'minigpt-cn-m3-pretrain-100m-v1.zip' | sha256sum -c -
if [ -e m3_pretrain_100m_v1 ] || [ -L m3_pretrain_100m_v1 ] || \
   [ -e m3_training_input_frozen_v1 ] || [ -L m3_training_input_frozen_v1 ] || \
   [ -e m3_training_runs/pretrain_100m_v1 ]; then
  echo '已有本轮脚本、输入或训练结果，先检查，不覆盖或盲目重跑'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-pretrain-100m-v1.zip .
cd m3_pretrain_100m_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 15m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u freeze_input.py
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u \
  run_pretraining.py --preflight-only --report-path preflight_report.json
SH
```

这段不调用训练入口或创建模型权重。冻结输入输出/root/autodl-tmp/m3_training_input_frozen_v1；未来模型输出为/root/autodl-tmp/m3_training_runs/pretrain_100m_v1。
已有输入或报告先检查，不覆盖；失败/超时保留所有诊断。若冻结完成但GPU预检查失败，只针对具体阻塞修正后重跑预检查，不重新冻结或重编码。
只解包过而未执行时，先核对ZIP/传输清单，再从脚本目录执行冻结与预检查两条命令。
预检查默认仅检查，显式--run还必须携带本机已验收冻结manifest SHA-256；实际启动命令在四份原始报告验收后给出。

## 取回四份小报告，完成本轮验收

程序自动打包/root/autodl-tmp/m3_pretrain_100m_v1/m3_pretraining_preparation_reports_v1.zip，含：

- frozen_manifest.json：冻结输入版本、全部文件摘要、保存后全文磁盘解码与隔离结果。
- excluded_documents.json：8条准确暂缓及实际剔除token量。
- scale_report.json：实际位置/文档数、12208步处理量和算术曝光。
- preflight_report.json：真实GPU/环境、配置/范围/冻结清单摘要，model_run=false。

用户下载这个ZIP，解压到PC的D:\aiworkspace\transformer\logs\m3_pretraining_input_v1。
大JSONL/bin继续留云端；已有小报告先检查，不覆盖。只看控制台末尾不替代四份原始文件验收。

## 本轮实际本机准备验证

Windows显式CPU定向35 passed（1.63秒），Ruff、包内CLI帮助、ZIP逐文件SHA-256读回通过。
覆盖真实16K文档片段/原行字节保留、准确暂缓匹配、旧heldout拒绝、磁盘ID/EOS/LF损坏、保存后全文解码拒绝、valid/test复制不覆盖、重复t2s姓名保留、输入/配置/版本混用拒绝，以及默认不训练/运行要求正确冻结SHA。
37个源码与既有函数副本字节不变。准备轮未云端冻结/GPU预检查或模型运行，也未重跑完整CPU/WSL回归；用户随后已完成冻结与预检查，以下保存实际验收结论。
记录在logs/m3_pretraining_prepare_20261014_v1/local_verification.json与delivery_manifest.json。

## 实际冻结与预检查验收

2026-10-14四份原始小报告已回传logs/m3_pretraining_input_v1，本轮40项核对与Ruff通过，记录artifact_acceptance.json。
本机实际重算四文件、固定配置/范围/绑定/tokenizer/37个源码摘要，并以8条原始样本独立复算扣量。
云端冻结23.399秒，实际73520/461/450篇、48187994/297390/251517个token；8条暂缓准确删除14082 token/19225字符/49972正文UTF-8字节。
valid/test源JSONL和bin大小/SHA均与批次3一致，三集合全部磁盘逐篇解码与旧346/303 heldout三键隔离报告通过；本机未重新读取云端大JSONL/bin。
GPU预检查PASS：已验收Python3.12.3/Torch2.8.0+cu128/CUDA12.8/RTX4080 SUPER，空闲显存33526841344字节、磁盘44787806208字节；model_run=false。

| 原始文件 | 字节 | 本机实际SHA-256 |
| --- | ---: | --- |
| frozen_manifest.json | 10641 | 60ff2f6cdca5b240803e4fa63fad31a87905c6907414dd12f125dbc3677682fe |
| excluded_documents.json | 5427 | 1d0f3ae546efc032a89b11b706b07ff034d8767c8e367dd461cc3f4e2ed8f187 |
| scale_report.json | 384 | 8c36897240532de7f82179215bbb2086a0e494b594fff0dc42d472b33c956219 |
| preflight_report.json | 2210 | 98fa9e48603374ba1ed3ca9332d3e5089cb8ce6972e5f2b47c6feddc470b8b74 |

冻结train.bin SHA-256=65c939af6b8fdc07210b05e9c3ca703af8e2b4f527cdfb53dace926b26ae3c3c。
以上为输入和运行准备验收；用户随后完成12208步并回传报告，结果见本页末尾。原候选状态、事实/逐篇许可和展示限制保持。
本轮没有重跑35项准备测试或完整CPU/WSL回归，35通过是准备轮记录。

## 用户启动固定100M训练

仅在上述冻结清单验收通过后执行一次。后台运行便于关闭终端后继续，stdout已有时拒绝覆盖；90分钟上限只限制本次进程，不延长配置或自动续训。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp/m3_pretrain_100m_v1
if [ -e ../m3_training_runs/pretrain_100m_v1 ] || \
   [ -L ../m3_training_runs/pretrain_100m_v1 ] || \
   [ -e pretrain_100m_v1.stdout.log ] || [ -L pretrain_100m_v1.stdout.log ]; then
  echo '已有训练输出或日志，先检查，不覆盖或重复启动'
  exit 1
fi
set -C
nohup timeout --signal=TERM 90m env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u \
  run_pretraining.py --run \
  --expected-frozen-manifest-sha256 60ff2f6cdca5b240803e4fa63fad31a87905c6907414dd12f125dbc3677682fe \
  > pretrain_100m_v1.stdout.log 2>&1 < /dev/null &
printf '训练后台进程PID：%s\n' "$!"
SH
```

启动时脚本再次核对完整冻结输入和当前GPU环境，验证摘要一致后才创建模型。打印PID只表示后台命令已启动，实际训练以日志为准。
查看进度（该命令不启动训练）：

```bash
tail -n 8 /root/autodl-tmp/m3_pretrain_100m_v1/pretrain_100m_v1.stdout.log
```

按已有吞吐，训练本体约44分钟，完整valid和保存另计；这是估算，不是本次实测耗时。
最终成功报告在/root/autodl-tmp/m3_training_runs/pretrain_100m_v1/training_report.json，必须completed/iteration12208；异常或超时保留所有产物，不能把部分输出当成功结果，也不重新从随机权重盲目开训。
完成后进入本次训练产物验收，随后独立评估/生成；不根据旧test分数选新checkpoint或自动延长训练。

## 实际100M训练报告与原始产物取回

用户已完成固定训练，原始training_report.json下载至logs/m3_training_report（5484字节，本机SHA-256=0d24ba7ff92ed11889262d6244c1f15ba467e04dd567f785c8c814e8363cf761）。
报告轮27项核对与Ruff通过，记录report_acceptance.json；用户随后取回完整日志/checkpoint并完成31项产物/CPU重载验收，见本页末尾。独立评估/生成尚未完成。

| 项目 | 报告结果 |
| --- | --- |
| 模型/起点 | 33104384参数，随机权重，无pilot/first_2000导入 |
| 训练终点/处理量 | 12208 / 100007936 tokens |
| 实际train位置/曝光 | 48187994 / 约2.07537倍算术曝光 |
| 完整valid | 26个验证点，最终297389个目标 |
| valid loss/PPL | 9.7740556→3.9629027 / 最终52.6098 |
| 最后10000→12208 loss变化 | 下降0.0382403，不自动延长训练 |
| 总调用/循环时间 | 2971.356 / 2969.517秒（约49.52分钟） |
| 含验证和保存的循环吞吐 | 33678.185 tokens/s |
| PyTorch峰值分配/保留 | 5.804 / 6.242GiB，不含驱动或其他进程 |
| checkpoint | 6个周期文件及final，均397351864字节 |

final报告SHA-256=f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d；用户随后回传，本机实际复算权重摘要一致。
当前报告全部valid点持续下降；无本次test/生成结论，不与旧不同valid集合的loss直接作受控比较，不标M3完成。
本轮未重新训练、验证、生成或完整CPU/WSL回归。运行费用以控制台账单为准，不把49.52分钟当全部实例计费。

用户在AutoDL终端执行一次，校验final摘要并打包五份剩余原始产物；使用ZIP_STORED快速打包，输出在autodl-tmp根目录，可直接在Jupyter下载：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

base = Path('/root/autodl-tmp')
run = base / 'm3_training_runs/pretrain_100m_v1'
report = json.loads((run / 'training_report.json').read_text())
assert report['training'] == 'completed' and report['result']['iteration'] == 12208
files = {name: run / name for name in ('ckpt_final.pt', 'metrics.jsonl', 'run.json', 'training_scope.json')}
files['pretrain_100m_v1.stdout.log'] = base / 'm3_pretrain_100m_v1/pretrain_100m_v1.stdout.log'
for path in files.values():
    assert path.is_file() and not path.is_symlink(), f'缺少普通文件：{path}'
checkpoint = files['ckpt_final.pt']
digest = hashlib.sha256()
with checkpoint.open('rb') as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
assert checkpoint.stat().st_size == report['checkpoints']['ckpt_final.pt']['bytes']
assert digest.hexdigest() == report['checkpoints']['ckpt_final.pt']['sha256'] == 'f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d'
packet = base / 'm3_pretrain_100m_return_v1.zip'
assert not packet.exists() and not packet.is_symlink(), '已有取回ZIP，先检查，不覆盖'
with ZipFile(packet, 'x', compression=ZIP_STORED) as archive:
    for name, path in files.items():
        archive.write(path, arcname=name)
print(f'PASS：final摘要匹配，五份原始产物已打包：{packet}')
PY
```

下载m3_pretrain_100m_return_v1.zip并解压至PC的D:\aiworkspace\transformer\logs\m3_training_report。
ZIP不包含已有training_report.json，避免覆盖；周期checkpoint保留云端，不必全部下载。
本轮训练产物验收需核对原始metrics/run/scope/stdout与报告一致、final本机大小/SHA及CPU安全重载/共享权重/优化器状态；随后另轮进行独立评估与固定生成。

## 原始产物与CPU重载验收

2026-10-14：五份取回文件实际放在logs/m3_training_report/m3_pretrain_100m_return_v1，与父目录已有training_report.json共同验收，保留实际目录/原始文件，不搬移或覆盖。
本轮31项核对与Ruff通过，记录logs/m3_training_report/artifact_acceptance.json；先前report_acceptance.json中的pending是报告轮快照，现由完整产物记录补充。

| 实际原始文件 | 字节 | 本机SHA-256 |
| --- | ---: | --- |
| training_report.json（父目录） | 5484 | 0d24ba7ff92ed11889262d6244c1f15ba467e04dd567f785c8c814e8363cf761 |
| ckpt_final.pt | 397351864 | f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d |
| metrics.jsonl | 137203 | 2b1831437903a1204ac5b0be0f9b2ccbad25e4111d17fc7e1e8ca24a093d51bd |
| run.json | 1749 | bda74704fd50f2a7ab564e12f32d5c793782d9123e4877868b6129b55370beb1 |
| training_scope.json | 11625 | 0e54215ae4588e6fd5ab17ffa46256f8026ef4622fb919857c91918ba511f857 |
| pretrain_100m_v1.stdout.log | 144958 | 183c5d1983a1290953495f9e85ae5d89550e391a7b8d075122c5dda6aa8d7429 |

实际核对611条train、26次完整valid及1条finished，全部数字有限/时间顺序正确；所有valid各297389目标、PPL与exp(loss)一致，末次loss/PPL和stdout/metrics/报告一致。
全部训练记录的学习率以独立公式复算，最终实际LR=3.0000004860020353e-05（零基更新下标12207）；固定12208步/100007936处理量、不读test或自动延长。
本机37份冻结源码、配置/范围/绑定/输入清单/tokenizer摘要核对通过；run/checkpoint中的输入SHA对应已验收冻结train/valid，不混用旧pilot或test。
项目加载器CPU重载最终checkpoint确认12208步、33104384参数、共享embedding/LM head和全部有限FP32权重；同版tokenizer实际16000/15743/EOS256。
74份优化器状态均12208步、FP32一/二阶矩形状正确且有限、二阶矩非负，参数顺序及共享参数唯一性通过；CPU采样器与Python/NumPy/Torch CPU RNG状态可读。
原始六文件在验收前后摘要一致。Windows Torch2.14.0+cu126只用于CPU读取，未恢复CUDA RNG/初始化CUDA或进行前向、验证、生成、训练，不宣称跨框架精确续训或valid复现。
本轮未重跑完整CPU/WSL回归，未重新读取云端大JSONL/bin或六个周期checkpoint；其存留位置保持。最终权重已完整保存在PC，下载/训练任务完成，可由用户停止实例，不自动释放或记录尚未确认的关机状态。
唯一下一项是本次final独立valid/test与固定生成评估准备，评估环境与协议需明确绑定；本轮不启动评估、不根据loss直接验收生成质量、不标M3完成。
