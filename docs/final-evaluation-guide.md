# 最终12208步模型独立评估与固定中文生成

2026-10-14。当前任务以[current-work.md](current-work.md)为准。用户已选择本地并执行完成，四文件/全部12条样本验收见[实际结果](#本地实际结果与完整样本验收)；下方命令保留作复现参考，已有结果不重评或覆盖，不要求启动云端GPU评估。
训练六份原始产物已完成31项核对/CPU重载，见[训练产物验收](pretraining-100m-guide.md#原始产物与cpu重载验收)。

## 固定输入与协议

- checkpoint：/root/autodl-tmp/m3_training_runs/pretrain_100m_v1/ckpt_final.pt。
- 本机验收397351864字节，SHA-256=f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d，12208步/33104384参数。
- tokenizer：原项目data/processed/pilot_tokenizer_16k_v1.json，SHA-256=21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470；实际16000词表/15743 merges/EOS256。
- valid/test：m3_training_input_frozen_v1/tokens/valid.bin与test.bin，297390/251517个token位置，完整目标297389/251516。
- 冻结根清单SHA-256=60ff2f6cdca5b240803e4fa63fad31a87905c6907414dd12f125dbc3677682fe。
- 原Python3.12/PyTorch2.8.0+cu128/CUDA12.8/RTX4080 SUPER/BF16环境，确定性算法开启、TF32关闭。

评估包绑定47份云端文件：六份训练原始产物、37份冻结源码、冻结根清单、valid/test bin及tokenizer。
独立重载final，先完整valid与训练末次loss=3.962902672139143比较，绝对容差1e-6；不符停止，尚未读取test或生成。
valid通过后，完整test只作这个固定checkpoint的结果报告，不用于选择其他checkpoint、改提示词/超参或追加训练。
复用原evaluate，覆盖所有next-token目标及不足512的尾块，不训练、续训、重新编码或改原候选标志。

固定生成协议为configs/m3_final_evaluation_protocol.json，除purpose字段外与首轮协议内容相同；在看到输出前固定：

| ID | 提示词 |
| --- | --- |
| geography | 北京是中国的 |
| history | 中国的历史可以追溯到 |
| astronomy | 太阳系中，地球 |
| technology | 计算机是一种 |
| biology | 植物通过光合作用 |
| traditional_chinese | 臺灣位於 |

每条greedy和sampling两版，共12条、最多128个新token、EOS256停止；sampling为独立seed42/temperature0.8/top_p0.9。
保持use_cache=false，全部样本按固定次序保存原始token ID、全文/续写、长度、停止原因和UTF-8异常，重复/空/异常输出也保留。
重复token 4-gram比例与最长相同token串仅作表面重复提示，不自动代表事实正确性、聊天能力或展示通过。
valid/test来源与旧2000步不同，loss不能直接作为旧模型与本次模型的受控同集合对比；相同固定提示词可作定性审阅。

## 用户上传后一次执行

本机包：logs/m3_final_evaluation_prepare_20261014_v1/minigpt-cn-m3-final-eval-v1.zip，9802字节。
SHA-256=e8724a34ed23124d3b2aceacb966c993b78b362fe0f3ff71c6539ed6129cfa5f。
包只有evaluate_final.py、evaluation_protocol.json、evaluation_binding.json、CODE_LICENSE和transfer_manifest.json，不重复传权重/语料或安装依赖。
如果实例已停止，由用户手动启动原实例；不自动开关机。上传ZIP至/root/autodl-tmp，执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  'e8724a34ed23124d3b2aceacb966c993b78b362fe0f3ff71c6539ed6129cfa5f' \
  'minigpt-cn-m3-final-eval-v1.zip' | sha256sum -c -
for path in m3_final_eval_v1 m3_final_evaluation_v1 m3_final_evaluation_reports_v1.zip; do
  if [ -e "$path" ] || [ -L "$path" ]; then
    echo "已有本轮目录或报告：$path；先检查，不覆盖或盲目重评"
    exit 1
  fi
done
python -m zipfile -e minigpt-cn-m3-final-eval-v1.zip .
cd m3_final_eval_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 5m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u evaluate_final.py
SH
```

脚本目录为m3_final_eval_v1，结果目录为m3_final_evaluation_v1；已有输出先检查，不覆盖或盲目重评。
如只解包而尚未运行，核对传输清单后从脚本目录运行同一timeout命令，勿重新解包或安装。
每个集合和生成条目打印进度；总命令限5分钟。所有输入/源码在运行前后摘要核对，checkpoint保持不变；异常/超时保留诊断，缺少最终manifest.json不算完成。

## 一次下载四文件报告ZIP

成功后自动生成/root/autodl-tmp/m3_final_evaluation_reports_v1.zip，里面是：

- evaluation_report.json：完整valid/test、valid复现差值、环境和固定协议。
- generations.jsonl：全部12条原始token ID/文本与生成元数据。
- generation_review.md：全部12条可读续写，不挑选或修改。
- manifest.json：checkpoint与三份输出大小/SHA-256、完成状态。

用户下载这个ZIP，解压到PC的D:\aiworkspace\transformer\logs\m3_final_evaluation_v1。
不必重下379MiB权重；下载后本轮GPU任务即可结束，用户可停止实例，保留数据。
回传原始四文件后验收摘要、完整目标数、valid复现、固定次序/采样配置/解码，并阅读全部12条续写。
完成语义审阅前不标生成质量或M3通过；本轮停在固定评估验收点，不追加训练或自动改提示词。

## 本轮实际本机验证

Windows显式CPU定向15 passed（4.47秒），Ruff通过；真实小模型完整valid/test与12条输出、权重不变、
valid复现失败前不读取test、目标数量错误、包/输入版本变化与越界路径拒绝、已有输出/报告ZIP保护、原始UTF-8与Markdown围栏、独立重复量检查均通过。
ZIP逐文件SHA-256读回/CLI帮助通过，37份冻结源码与旧评估产物不变。记录见logs/m3_final_evaluation_prepare_20261014_v1。
这些CPU测试不等于本次33M实际GPU评估结果；本轮未启动云端或完整CPU/WSL回归。

## 转到Windows本机评估

用户明确选择本地运行。实际核对Windows .venv：Python3.12.4/Torch2.14.0+cu126/CUDA12.6，单卡RTX4060 Laptop（8585216000字节显存）、CUDA可用/BF16支持。
最终权重已在logs/m3_training_report/m3_pretrain_100m_return_v1/ckpt_final.pt，tokenizer在data/processed/pilot_tokenizer_16k_v1.json，无需重下或重训。
本地脚本logs/m3_final_local_evaluation_prepare_v1/evaluate_local.py，绑定47份本机产物/冻结源码/输入；原云端包与严格复现协议保留，不覆盖。
本轮9项显式CPU定向通过（4.70秒）、Ruff/CLI帮助通过；未运行正式33M评估或生成、未重跑完整CPU/WSL回归。

本机当前缺少本次冻结valid/test，旧pilot的两个bin不能替代。只需以下两份，共1097814字节（约1.05MiB），不取train/Parquet：

| 文件 | 字节 | 固定SHA-256 |
| --- | ---: | --- |
| valid.bin | 594780 | e0091ec51ccd9ee7ca99c7314c28129dd18387c77854a129334847a13a5bc9eb |
| test.bin | 503034 | 9b1efe3c350d9bf1b4070173bdf5cd2b9a71b564a0aa404ef6306ac81b481412 |

用户在云端终端打包（不创建模型或运行GPU评估）：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile
source = Path('/root/autodl-tmp/m3_training_input_frozen_v1/tokens')
archive = Path('/root/autodl-tmp/m3_final_eval_inputs_v1.zip')
for name in ('valid.bin', 'test.bin'):
    assert (source / name).is_file() and not (source / name).is_symlink(), name
assert not archive.exists() and not archive.is_symlink(), '已有ZIP，先检查，不覆盖'
with ZipFile(archive, 'x', compression=ZIP_STORED) as packet:
    for name in ('valid.bin', 'test.bin'):
        packet.write(source / name, arcname=name)
print(archive)
PY
```

下载这个ZIP，解压使两份bin直接位于D:\aiworkspace\transformer\data\processed\m3_final_eval_inputs_v1。
已有同名文件先核对，不覆盖；助手不下载。取回后这轮评估不再需要云端运行，用户可停止实例并保留数据。

在Windows PowerShell执行：

```powershell
Set-Location D:\aiworkspace\transformer
.\.venv\Scripts\python.exe logs/m3_final_local_evaluation_prepare_v1/evaluate_local.py --preflight-only
if ($LASTEXITCODE -ne 0) { throw '本地输入或环境预检查未通过；保留原文件' }
.\.venv\Scripts\python.exe -u logs/m3_final_local_evaluation_prepare_v1/evaluate_local.py
```

预检查先核对47份文件大小/SHA和本机GPU环境，不创建模型；缺失/版本混用即停止。正式运行保持确定性算法/TF32关闭/BF16前向，完整valid/test297389/251516目标与原12条续写协议不变。
因为Torch/GPU与云端不同，本地报告独立指标和valid差值，valid_reproduction_passed=null，不套用云端1e-6门槛或宣称精确复现；不依据差值/test调整模型、提示词或追加训练。
输入摘要、全部目标覆盖、数值有限和固定协议检查仍严格；本地结果不会覆盖原训练报告或云端评估输出。
结果直接保存logs/m3_final_local_evaluation_v1的evaluation_report.json/generations.jsonl/generation_review.md/manifest.json，无需再次下载。
已有输出拒绝重评/覆盖，异常保留诊断；用户随后已运行，完整四文件与12条样本实际验收见下节，不重评或覆盖。

## 本地实际结果与完整样本验收

2026-10-14用户在Windows/RTX4060 Laptop完成，54.6179447秒；四份原始文件保存logs/m3_final_local_evaluation_v1。
本轮31项核对/Ruff通过，记录artifact_acceptance.json；实际读取/复算输出摘要、47份输入/冻结源码/权重绑定、已下载valid/test bin、全部目标/PPL/跨环境差值和12条原始ID/文本/重复量，未重跑模型或完整CPU/WSL回归。

| 集合 | 完整目标 | loss | PPL |
| --- | ---: | ---: | ---: |
| valid | 297389 | 3.9628930030 | 52.6093048833 |
| test | 251516 | 3.9957795353 | 54.3682060471 |

本地valid与云端训练末次绝对差9.6691184721e-6；Torch2.14/CUDA12.6/4060与云端2.8/CUDA12.8/4080不同，valid_reproduction_passed=null，差值只记录为跨环境观察，不把它当同环境1e-6精确复现PASS。
本机valid/test的大小/SHA与冻结清单一致，最终checkpoint本机摘要仍匹配已验收12208步final；没有改权重或追加训练。

| 原始文件 | 字节 | 本机实际SHA-256 |
| --- | ---: | --- |
| evaluation_report.json | 3390 | 9c6c731a8b80afa64531bca2bbeed6c098850a7b810f536e1f7d5af10eef031f |
| generations.jsonl | 32440 | 0c7d150f3e2b4863cdbbc1f78bed28a8cefbc3aa6826890a331821a75fd4aecf |
| generation_review.md | 6904 | 869ae4583a4e55141139f01774ed35017dd43335c1dfb10039516237ca92ffcd |
| manifest.json | 710 | 0c620d9d75f5938572d568bd5af0759a5312fc3127806a855c72d335e9238946 |

全部12条次序/采样设置正确，原始ID拼接和UTF-8解码、EOS/长度、Markdown精确重建与重复量复算通过；UTF-8异常0、提前EOS1（technology/greedy为18个新token），其余11条均128个token，最大token 4-gram重复比例0.928。
12条全部全文审阅，逐条结论保存artifact_acceptance.json的sample_review：

| 提示词 | greedy观察 | sampling观察 |
| --- | --- | --- |
| 北京是中国的 | 开头短句后反复北京 | 上海八三/制程/年代拼接，偏离北京主题 |
| 中国的历史可以追溯到 | 年代后长段重复同一句 | 西汉/新英格兰/英文人名背景混杂及英文碎片 |
| 太阳系中，地球 | 轨道循环 | 太阳表面/地球大气/流星关系混乱、残缺外链 |
| 计算机是一种 | 计算机计算机重复后提前EOS | RC/DTL/SAL/CLDF等缩写与人名拼接、不连贯 |
| 植物通过光合作用 | 植物循环 | 植物生长重复及以防止植物等混乱表达 |
| 臺灣位於 | 转向鱼类分布，印度循环 | 九头八脚后八脚长循环 |

结论：评估产物与技术链路PASS；生成质量未接纳为展示终稿，6条greedy均明显重复，sampling仍离题/句意混乱或循环。
以上只描述这12条固定样本，不推算总体事实正确率，不证明唯一失败原因；不挑样本隐藏问题，也不以test或输出不足自动改采样/选checkpoint/延长训练。
评估验收轮停在上述结论；随后已整理[M3实验结果报告](m3-experiment-report.md)，保留指标与全部12条原样输出/失败结论。下一项按current-work.md核对阶段收尾范围，M3远端尚未在本轮关闭，云端无需再运行本次评估。
