# 首轮2000步独立评估与固定提示词生成

当前任务以[current-work.md](current-work.md)为准。本轮仅评估已验收final，不训练、续训、重建数据或安装依赖。
首轮原始权重和日志的27项验收见[训练产物](first-training-guide.md#原始产物实际验收)。

## 固定输入与评估范围

- checkpoint：/root/autodl-tmp/m3_training_runs/first_2000_v1/ckpt_final.pt。
- 实际397351864字节，SHA-256=db1034589ab6a30d514993112642622076d2e92963516e0d05818c304a52a10f，2000步/33104384参数。
- tokenizer：原项目data/processed/pilot_tokenizer_16k_v1.json，SHA-256=21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470。
- 只评估m3_candidate_tokens_16k_v1/tokens/valid.bin与test.bin，分别46761/53483 token位置，覆盖46760/53482预测目标。
- 使用原Python3.12/PyTorch2.8.0+cu128/RTX4080 SUPER单卡/CUDA BF16，确定性算法开启、TF32关闭。

脚本绑定47个既有云端文件摘要：六份原始训练产物、37个项目源码、编码根清单、valid/test bin和tokenizer。
独立新进程通过项目加载器重载final，复用原evaluate覆盖全部next-token目标，包含不足512的尾块。
valid与训练末次loss=5.621109157183723比较，绝对容差1e-6；不符停止并保留诊断。
test只用于报告这个固定checkpoint的结果，不据此改提示词/超参、选其他checkpoint或延长训练。
本批仍是约445万train token位置上的有界中文实验，评估不改变旧candidate状态或宣称正式规模/效果。

## 固定生成协议

协议保存于configs/m3_first_evaluation_protocol.json，在看到本轮输出前固定：

| ID | 提示词 |
| --- | --- |
| geography | 北京是中国的 |
| history | 中国的历史可以追溯到 |
| astronomy | 太阳系中，地球 |
| technology | 计算机是一种 |
| biology | 植物通过光合作用 |
| traditional_chinese | 臺灣位於 |

每条分别生成greedy和sampling两版，共12条；最多128个新token，EOS256停止。
sampling固定seed42、temperature0.8/top_p0.9；greedy记录temperature1/top_p1，实际取argmax。
每条调用使用独立GPU采样器，不依赖模型加载消耗的全局随机数；本轮use_cache=false，复用现有无缓存生成入口。
这是百科风格短文本续写抽检，不是聊天/指令遵循、事实正确率或泛化能力基准。
保存全部token ID、全文/续写、实际生成长度、停止原因、UTF-8异常和替换字符；不删除空/重复/乱码样本。
额外报告token 4-gram重复比例与最长相同token串，仅作表面重复提示，不自动判定生成质量或事实正确性。

## 上传后一次执行

本机包：logs/m3_first_evaluation_prepare_20261014_v1/minigpt-cn-m3-first-eval-v1.zip。
9585字节，SHA-256=de53aabec0a6a6529cb9ed5c1ce6cbff7daa070068c690e51c1076939915d0d3。
包内仅evaluate_first.py、evaluation_protocol.json、evaluation_binding.json、CODE_LICENSE、transfer_manifest.json。
不传权重或数据，不新增依赖。用户上传ZIP到/root/autodl-tmp后执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  'de53aabec0a6a6529cb9ed5c1ce6cbff7daa070068c690e51c1076939915d0d3' \
  'minigpt-cn-m3-first-eval-v1.zip' | sha256sum -c -
if [ -e m3_first_eval_v1 ] || [ -L m3_first_eval_v1 ]; then
  echo '已有评估脚本目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-first-eval-v1.zip .
cd m3_first_eval_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 5m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u \
  evaluate_first.py
SH
```

每个集合与生成条目都有进度。输出默认/root/autodl-tmp/m3_first_evaluation_v1，必须新建；已有输出先验收，不覆盖或盲目重评。
脚本目录名m3_first_eval_v1与结果目录名m3_first_evaluation_v1不同。若脚本已解包但尚未运行，核对传输清单后执行同一timeout命令。
总命令最多5分钟，失败/超时保留诊断；最终manifest.json最后写入，没有该文件不算本轮完成。
在完成后重新核对全部输入/源码摘要，确认checkpoint等既有文件没有改写。

## 取回与验收

用户贴回末尾evaluation_report JSON和PASS，并下载结果目录中的4份小文件到PC：
D:\\aiworkspace\\transformer\\logs\\m3_first_evaluation_v1。

- evaluation_report.json：独立valid/test loss/PPL/完整覆盖、valid复现差值、版本与固定生成协议。
- generations.jsonl：全部12条样本，含原始token IDs与UTF-8状态。
- generation_review.md：全部12条可读续写，未挑选结果。
- manifest.json：绑定checkpoint与三份输出字节数/SHA-256。

不必重复下载379MiB权重。助手收到报告/样本后核对版本/摘要、完整目标数、valid复现、12条固定次序和解码一致性，
阅读所有续写并记录重复/乱码/事实或连贯性问题；未完成语义审阅前不宣称模型质量通过或M3完成。
本轮结束后停在评估验收点，不自动增加训练步数或数据规模。

## 本轮实际本地验证

2026-10-14 Windows显式CPU定向73 passed/2 skipped（8.75秒），两项为CUDA生成/恢复检查；Ruff通过。
真实小模型执行完整valid/test和12条固定生成，验证全覆盖、权重不变、原样保留输出；还检查已有输出拒绝、
valid复现差异/目标数量错误拒绝、原始UTF-8异常和Markdown围栏保留、独立重复量计算。
ZIP逐文件读回通过；未重跑完整CPU回归或WSL检查，未运行本批33M云端评估，实际结果待用户回传。

## 实际结果与原始产物验收

2026-10-14用户在原云端完成，33.887911秒；四份原始文件已下载至logs/m3_first_evaluation_v1。
本机verify_artifacts.py实际25项核对PASS，记录artifact_acceptance.json，Ruff通过；未重跑模型或完整CPU回归。
包括三份输出摘要、固定checkpoint/tokenizer/协议/绑定版本、两集合全文目标/PPL、valid复现、
12条固定次序/采样设置、全部原始ID和UTF-8重新解码、EOS/长度、重复量复算、Markdown与JSONL精确重建一致。

| 集合 | 完整目标 | loss | PPL |
| --- | ---: | ---: | ---: |
| valid | 46760 | 5.6211091572 | 276.195558 |
| test | 53482 | 5.7310125140 | 308.281249 |

独立valid与训练末次绝对差值0；test仅作这个固定checkpoint的结果记录，不用于选模型或延长训练。
全部12条UTF-8有效，sampling有2条提前EOS；这仅证明解码合法，不能证明文本质量。
6条greedy均明显循环，最大token 4-gram重复比例0.992；sampling存在碎片拼接、离题和句意混乱。
例如geography/greedy的128个新token反复“中国”，technology/sampling转向动物颌骨等内容，
traditional_chinese两版均转向动物形态描述；逐条审阅结论已写入验收记录。
本轮评估产物/技术链路验收通过，生成质量未接纳为展示终稿，不把这12条当总体事实正确率或单一失败原因的证明。

用户随后询问正式训练数据路线：建议先规划50M去重后的train token位置，优先已有固定Parquet首分片的有界处理，
不足才逐个获取下一固定分片，复用冻结工具/原16K tokenizer；不是直接重复445万train位置凑1亿处理tokens。
这是规模建议而非已生成产物或效果保证；新下载、处理和训练本轮均未执行，下一任务以current-work.md为准。
