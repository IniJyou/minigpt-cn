# 有界续训执行准备

2026-10-15。用户授权准备执行文件与固定评估协议。本轮没有在云端启动训练，也没有续训用户33M模型。
原云端8项输入摘要、50项Python检查（含37份源码）已通过，来源和驱动差异记录见logs/m3_bounded_continuation_precheck_v1/cloud_precheck_acceptance.json。
本页落实[有界续训方案](bounded-continuation-plan.md)，不更换语料、模型、tokenizer或重置优化器。

## 交付包与固定范围

PC文件：logs/m3_continue_2000_prepare_v1/minigpt-cn-m3-continue-2000-v1.zip，18926字节。

- ZIP SHA-256：085f549e9d366aa46a95a53859c75cff5f3973678b0b19491a75365012cf36c1。
- 续训协议SHA-256：8820873517d99422be35cde38f672f66c422031fdc8822a04662496a1dea607b。
- 评估协议SHA-256：467ae51fcc20a259710309e16aa87750e92ff1be44bdcf8ea8f4f9adadec4e54。

包内只有执行脚本、已验收只读检查脚本、两个协议、6篇固定valid前缀和说明；没有语料、tokenizer、checkpoint、库或安装/下载逻辑。
训练仍调用原云端项目源码，全部37份原源码、8份输入/配置和父checkpoint固定摘要；本地新的生成参数只用于后续PC评估，不替换云端冻结源码。

| 项目 | 固定值 |
| --- | --- |
| 父权重 | 原12208步final，SHA-256=f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d |
| 终点 | 14208，恰好追加2000步 |
| 唯一训练配置变化 | max_steps=14208 |
| batch/context/precision | 16/512/BF16 |
| 追加学习率 | 原调度终点不变，2000步均3e-5，不重新预热 |
| 追加/累计处理tokens | 16384000 / 116391936 |
| 输出目录 | /root/autodl-tmp/m3_training_runs/continue_2000_v1 |
| 完整valid步数 | 12208/12500/13000/13500/14000/14208，每次297389目标 |
| 周期/最终保存 | step_014000.pt与14208步ckpt_final.pt |

## 本轮用户上传与预检查

将ZIP文件原样上传到/root/autodl-tmp，不编辑其中内容。使用现有原项目.venv，不安装任何依赖。
复制下面这段到AutoDL终端；它只核对ZIP、解压到新执行目录，再做只读预检查，不调用训练。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' '085f549e9d366aa46a95a53859c75cff5f3973678b0b19491a75365012cf36c1' 'minigpt-cn-m3-continue-2000-v1.zip' | sha256sum -c -
if [ -e m3_continue_2000_v1 ] || [ -L m3_continue_2000_v1 ]; then
  echo '已有执行包目录，先检查，不覆盖或盲目解压'
  exit 1
fi
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -B -m zipfile -e minigpt-cn-m3-continue-2000-v1.zip .
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -B -u m3_continue_2000_v1/run_continuation.py --preflight-only
SH
```

末尾应为“PASS：续训交付文件与原云端只读检查；未启动训练”。请回传末尾结果或具体报错，已有目录不自动删除/改名。
准备轮停在执行包交付点。用户随后已回传完整包预检查PASS，回传副本及验收在logs/m3_continue_2000_prepare_v1/cloud_package_preflight.user_report.json与cloud_package_preflight_acceptance.json；来源是用户终端贴文，不是助手直接访问云端。
显式启动必须使用--run并携带已审阅续训协议SHA-256，普通运行或--preflight-only都只读。

## 新包预检查通过后的启动指令

下面这段会正式启动固定追加2000步，请只执行一次。先拒绝已有输出与控制日志，再用noclobber创建新日志，避免重跑覆盖。
当前只交付启动指令，尚未收到实际启动/完成回传，不宣称已经训练。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp/m3_continue_2000_v1
for item in controller.stdout.log execution_attempt.json execution_status.json continuation.stdout.log ../m3_training_runs/continue_2000_v1; do
  if [ -e "$item" ] || [ -L "$item" ]; then
    echo "已有产物，先检查，不重启：$item"
    exit 1
  fi
done
set -C
nohup /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -B -u run_continuation.py --run --expected-protocol-sha256 8820873517d99422be35cde38f672f66c422031fdc8822a04662496a1dea607b > controller.stdout.log 2>&1 < /dev/null &
printf '已提交控制进程 PID=%s；实际状态以日志为准\n' "$!"
SH
```

启动后先做输入预检查；训练日志出现后查看：

```bash
tail -n 8 /root/autodl-tmp/m3_continue_2000_v1/continuation.stdout.log
```

如果训练日志还不存在，查看controller.stdout.log；可能仍在预检查，或已因具体检查失败而停止，不要重复启动。
按原吞吐预计约8分钟，总限时900秒含预检查，终止宽限另最多10秒；本次实际时间由报告记录。
结束后回传execution_status.json；如果execution_status=completed，再回传模型输出目录的continuation_report.json。失败/超时则回传状态与两份stdout日志末尾，保留所有文件。
原final保持，本次只是有界续训实验；完成不自动接纳生成质量，也不继续加步数。

## 显式启动后的行为

- 先校验交付文件和协议，再复用只读检查；不符合原恢复契约就停止，不能偷偷改版本、路径、batch或学习率。
- 训练子进程在导入torch前设置CUBLAS=:4096:8，随后明确启用确定性、关闭两项TF32与cuDNN benchmark，CPU线程数1。
- 900秒上限从本次启动开始计，包含预检查（自身最多60秒）、加载/训练/验证/保存和成功报告；超时先TERM，最多再给10秒终止宽限后KILL。不是无限等待或计费承诺。
- Linux worker要求本次控制进程PID和协议匹配，并通过PR_SET_PDEATHSIG设置父进程退出即TERM；如果设置失败则在构建训练模型前停止。Windows本机未实际验证该Linux系统调用。
- 已有训练输出、执行尝试、状态或stdout日志一律拒绝覆盖/盲目重跑；训练失败、超时或用户中断都保留已写产物，绝不自动重试、删除或追加步数。
- 成功后独立保存实际训练配置、恢复来源、输入/源码摘要、6次完整valid、耗时和本worker显存峰值；显存不包含驱动/其他进程，驱动变化不声称逐位等价。

执行目录/root/autodl-tmp/m3_continue_2000_v1将保存execution_attempt.json、continuation.stdout.log、execution_status.json。
模型输出目录由原train函数新建，保存run.json、metrics.jsonl、step_014000.pt、ckpt_final.pt及continuation_report.json。
只有固定终点成功并完成校验才写completed报告；部分checkpoint或仅有worker退出码0都不能冒充完整完成。
日志中的process_seconds与cumulative_seconds含义沿用原训练接口，区分本段与累计值；父进程另记含预检查的墙钟时间。
中断后本次尝试记录仍保留，先验收其状态，不复用同目录再次启动。

## 已冻结的后续评估协议

评估尚未运行，不拿test作调参、延长或选checkpoint依据。只比较原12208步final与成功终点14208步final，不选step_014000。
完整valid在同一PC评估环境重算两份权重，固定同版valid与源码；本地原参考loss=3.962893003020671，复现绝对容差1e-6。
同环境valid NLL须至少下降0.01；这是事先约定的实用阈值，没有统计显著性含义，不拿云端/PC不同环境分数直接作差判定。

生成固定同6篇valid、种子42/43/44，每份18条：temperature0.8/top_p0.9/repetition_penalty1.1/BF16/无缓存/max_new_tokens128。
保留全部ID/文本/停止/UTF-8/长度及完整和共同长度重复统计；原采样基线的协议、36条原始JSONL及源码版本已绑定，不挑种子。
逐对审阅主题延续、句意成立与有效正文推进，记录改善/相同/变差和证据；至少10对明确改善、最多3对明显变差才通过连贯性约定。
这不是盲评、独立人工或事实准确率，6篇valid已被反复查看，不能冒称新的泛化测试集。
valid与连贯性均满足才接纳为本批改善；重复率降低或训练完成本身不算质量通过。没有改善也记录并停止，不重新搜索解码参数或继续加步数。
实际候选SHA-256在训练成功后由独立验收回执补充绑定，不改现在冻结的协议内容。

## 本轮实际验证与边界

本机Windows定向检查17 passed（8.02秒），包括默认只读、必须显式--run及协议摘要、输入包变化/已有产物拒绝、真实CPU子进程超时清理、失败不伪造completed和worker缺少控制进程拒绝。
其中一个测试仅用临时目录中的极小合成模型/数据验证：原4步checkpoint只改终点续至6步，超过原学习率终点后保持min LR；模型权重、优化器矩、采样器及CPU RNG与连续6步相同，父文件不变。
这不是用户33M续训、真实语料训练或Torch2.8跨版本/跨驱动复现验证；Linux父进程退出保护和真实GPU训练尚未实际运行。
项目/执行脚本Ruff、ZIP全部文件读回及真实包内协议/帮助检查通过；未重跑完整CPU/WSL回归。详细证据在logs/m3_continue_2000_prepare_v1/preparation_acceptance.json。

## 实际续训报告验收（2026-10-15）

用户已取回两份原始JSON至logs/m3_contiune_2000_report，沿用用户实际目录拼写，没有改名或覆盖。
本轮26项报告一致性核对与新核对脚本Ruff通过，记录在同目录report_acceptance.json；这是报告验收，尚未读取新权重、metrics或run.json。

| 项目 | 实际报告值 |
| --- | ---: |
| 追加/最终累计步数 | 2000 / 14208 |
| 追加/累计处理tokens | 16384000 / 116391936 |
| 控制进程总耗时（含预检查） | 460.795723秒，约7.68分钟 |
| 训练调用/循环耗时 | 451.332507 / 448.852807秒 |
| 训练期valid起点/终点loss | 3.962902672139143 / 3.9458879316974587 |
| loss下降 | 0.0170147404416845 |
| 终点valid目标/PPL | 297389 / 51.72224354815517 |
| worker分配/保留显存峰值 | 6231435776 / 6916407296字节，约5.803 / 6.441GiB |

两份报告均completed，worker_exit_code=0、error=null，总耗时在900秒内。
恢复来源、两份协议的实际PC摘要、唯独max_steps变化的配置、6次valid步数、PPL和处理量复算均一致。
已实际CPU读取本地原12208步final的元数据，累计耗时按其保存的run_state.elapsed_seconds复算；没有新模型前向、读取test或重跑pytest。
实际运行报告确定性true、TF32两项false、CUBLAS=:4096:8；GPU/Torch/CUDA与已验收环境匹配，驱动580.105.08与更早训练报告不同，不声称逐位等价。
原final前后摘要不变。云端完整输入/原始metrics没有在PC重读，报告中的实际LR检查与6次valid仍需原始记录读回补充。

报告给出的终点候选为397352056字节，SHA-256=abd3d468c09b724c746e66bac769487073b17e734c8fc3f7f18956bc5bbf9f48。
step_014000.pt的报告SHA为fd09ee607b1db9eb4f169ea6117f323ca6ebedf1eeb00cee690b5afc428bac6e，留在云端，不作为候选选择。
这次训练期valid改善已记录，但预先约定的PC同环境独立valid与18对连贯性比较尚未执行，质量仍未接纳，不自动加步数。

### 取回本地验收所需的三份原始文件

报告验收当时PC目录仅有两份报告；用户随后已取回ckpt_final.pt、metrics.jsonl、run.json至同一PC目录D:\aiworkspace\transformer\logs\m3_contiune_2000_report，原件验收见下节。下面打包命令仅保留操作记录，不必再次执行。
可以在原AutoDL终端一次打包，原数据只读；ZIP使用x模式，已有目标则拒绝覆盖。存储模式不压缩权重，减少打包CPU耗时，包大小约379MiB。

```bash
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -B - <<'PY'
from pathlib import Path
from zipfile import ZipFile
source = Path('/root/autodl-tmp/m3_training_runs/continue_2000_v1')
names = ('ckpt_final.pt', 'metrics.jsonl', 'run.json')
for name in names:
    if not (source / name).is_file():
        raise FileNotFoundError(source / name)
target = Path('/root/autodl-tmp/m3_continue_2000_return_v1.zip')
with ZipFile(target, 'x') as packet:
    for name in names:
        packet.write(source / name, name)
print(target)
PY
```

报告验收阶段尚未在PC验证权重；随后原件已取回并通过下列核对。不重新下载原语料、原final或周期权重，也不启动下一轮训练。

## 原始产物与CPU重载验收（2026-10-15）

本轮实际35项核对通过，记录在logs/m3_contiune_2000_report/artifact_acceptance.json；新verify_artifacts.py的Ruff通过。旧report_acceptance.json保留当时尚待原件的历史状态，没有覆盖。
三份原件实际大小/SHA均匹配continuation_report.json：final为397352056字节、SHA=abd3d468c09b724c746e66bac769487073b17e734c8fc3f7f18956bc5bbf9f48；metrics为21840字节、SHA=d0c714b1c0b11325744266a13b77b653da13a483e7f86e46f604d4f9eadfe323；run为1927字节、SHA=632c6f490c28203aa0527f72acb4aec11446bc063a525807abdfe525a8213e16。

完整日志有101条train、6次完整valid和1条finished，步数与预定间隔一致；全部实际train学习率3e-5，loss/梯度有限，valid目标297389、loss/PPL及累计计时与报告一致。run与权重的数据配置/摘要、恢复父来源、模型和训练配置匹配原final，只有max_steps变为14208。run中optimizer_settings是恢复起点记录，final优化器实际LR为3e-5。
项目加载器在CPU重载14208步/33104384参数，FP32权重有限，embedding与LM head共享同一参数；74份优化器状态均14208步、两矩FP32有限，顺序/形状匹配。16K tokenizer/15743 merges/EOS256及本地valid摘要一致。
CPU采样器和CPU RNG通过独立发生器读取；CUDA RNG仅检查保存状态存在，没有恢复全局/CUDA RNG。权重保存累计耗时3417.524117秒，位于最后valid与最终保存完成之间，未误当训练循环最终耗时。

候选实际摘要、原final摘要与冻结evaluation_protocol摘要已在独立回执candidate_binding绑定，没有改写协议或选择step_014000。所有原始文件、父权重、本地输入、两份协议与原M3报告验收前后摘要不变。
本轮Windows Torch2.14.0+cu126只做CPU读取验收，不宣称可精确恢复云端2.8；未重新计算云端train.bin摘要、读取周期权重、模型前向、读test、训练或重跑pytest/完整CPU/WSL回归。
原件验收阶段尚待同一PC环境的两模型完整valid与18对固定生成/连贯性比较；用户随后授权并完成下节对照。训练期valid下降本身不能认定生成质量已通过。

## 本地独立对照结果（2026-10-15）

本轮按原样冻结的evaluation_protocol执行，候选仅为已验收的14208步final，不改门槛或选择周期权重。数值运行实际125.326718秒，包含输入检查、两次模型加载、两份完整valid、36条续写及末次摘要核对，不包括之后逐对语义审阅。
实际环境为Windows/Python3.12.4/PyTorch2.14.0+cu126/CUDA12.6/RTX4060 Laptop GPU，与协议一致；确定性开启、两项TF32关闭、线程1、CUBLAS=:4096:8。没有恢复训练优化器或全局/CUDA RNG。

| 指标 | 原12208步 | 续训14208步 |
| --- | ---: | ---: |
| 完整valid目标 | 297389 | 297389 |
| valid loss | 3.962893003020671 | 3.9458841234747717 |
| valid PPL | 52.609304883285425 | 51.72204657870892 |
| 完整输出平均重复token 4-gram | 7.780309% | 7.655319% |
| 剔除EOS并对齐共同正文长度后的重复均值 | 6.523893% | 7.704540% |
| 平均新增token | 110.222222 | 101 |
| EOS条数 / UTF-8异常条数 | 5 / 0 | 6 / 0 |

同环境loss下降0.017008879545899358，达到预设至少0.01门槛；原模型valid与存档差0。原模型18条重新生成，与旧采样对照penalty1.1的18条ID逐条精确一致，未靠更换基线取得差异。
两模型均固定6篇valid原64-token前缀、种子42/43/44、temperature0.8/top-p0.9/repetition_penalty1.1、BF16/无缓存/max_new_tokens128；每条使用相同配对种子的独立CUDA发生器。所有18对完整ID/文本/停止/UTF8和两种长度统计保留，没有挑选样本。

全部36条按18对完整阅读后：**1对明确改善、12对无明确整体变化、5对变差**，未满足至少10改善且至多3变差。same表示未发现明确整体改善或明显整体变差，不表示文本相同。每对判断和两侧原文证据见semantic_review.json/md。
例如valid文档0/seed42从长数字版本串转为多句叙述，记相对改善，但仍有错误混合；文档90/seed42退化成没有事件的日期行，文档183/seed43在破碎首句后直接参考文献/M/EOS，记变差。较短或少重复本身不算连贯性改善。
这是助手按预定主题延续、句意成立与正文推进进行的定性审阅，不是盲评、独立人工或事实准确率。6篇valid已反复检查，不能据此声称未见数据泛化能力提高。

所有产物在logs/m3_continuation_comparison_v1：evaluation_protocol.json为冻结协议的逐字节副本；runtime_binding.json绑定实际输入/环境；baseline_valid.json、candidate_valid.json、samples.jsonl和comparison.json记录数值；comparison_review.md收录全部18对原样输出；semantic_review.json/md记录逐对判断；artifact_acceptance.json为最终验收。
数值comparison.json中的语义pending/quality_accepted=false是审阅前快照，保留不覆盖；最终回执包含完整语义判定。独立31项读回与项目/新脚本Ruff通过，包括全部输出解码/ID/valid前缀/停止/PPL/长度控制统计复算、18条基线精确复现和18对语义证据定位。
权重、tokenizer、valid、旧采样基线、冻结协议、全部37份源码与原M3报告摘要未变；没有读test、训练、安装/下载或Git写操作，未重跑pytest/完整CPU/WSL回归。

**结论：数值valid改善通过，连贯性门槛未通过，整体质量验收不通过。** 保留两份权重和全部结果，按约定停止，不自动增加步数或搜索参数；原M3报告继续作为优化前历史基线，M3不因此标记完成。
