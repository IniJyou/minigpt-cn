# M3同版本16K候选编码与实际token规模验收

当前任务以[current-work.md](current-work.md)为入口。本轮只有编码/磁盘读回/规模报告，不训练BPE或模型。
有界候选准备结论见[候选三集合](formal-corpus-guide.md)，原candidate/未批准状态保持不变。

## 固定输入与最小接入

- 云端v2目录：/root/autodl-tmp/m3_corpus_candidates_v2。
- 根清单SHA-256：5b41a1ce0534aadb2d311eb34f0e82944e4da7d7a153322082f40e40390e7d60。
- 本机候选验收记录SHA-256：1e599a8ebef1ac848a13073c4ca7e4c33afeba4398b06a3984e1e8d168c32abb。
- tokenizer仍为原500步项目内data/processed/pilot_tokenizer_16k_v1.json，SHA-256
  21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470。
- 必须实际16000词表、15743 merges、EOS ID256，不能改为16384、填伪token或混用旧2048版本。
- 应用已读全文的3篇暂缓ID950724/14968/251723；逐篇核对标题/URL、规范化正文及原始正文摘要和原集合。
- 其余JSONL行原样复制，正文不再清洗、不补词、不重分集合；拟编码6529篇（6384/68/77），最终以实际检查为准。

配置在configs/m3_candidate_encoding_policy.json。本轮一次性脚本复用既有encode._load_tokenizer、
_check_source和_encode_split；不把候选伪装成pilot，也不改变冻结源码的入口/审批门槛。
15个相关源码文件与原500步云端包逐文件摘要相同。

## 用户上传小包并执行一次

本地包：logs/m3_candidate_encode_20261014_v1/minigpt-cn-m3-candidate-encode-v1.zip。
10389字节，SHA-256：c923374460893952c69941675cba05d02a2c6bee418f56b73cdff7e3ae728f85。
包内仅本轮编码脚本、固定策略、已完成候选审阅记录、代码许可证和传输清单；不传正文/tokenizer或新依赖。
用户上传ZIP至/root/autodl-tmp后执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  'c923374460893952c69941675cba05d02a2c6bee418f56b73cdff7e3ae728f85' \
  'minigpt-cn-m3-candidate-encode-v1.zip' | sha256sum -c -
if [ -e m3_candidate_encode_v1 ] || [ -L m3_candidate_encode_v1 ]; then
  echo '已有编码脚本目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-candidate-encode-v1.zip .
cd m3_candidate_encode_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 15m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python \
  encode_candidates_once.py
SH
```

不需要安装或下载。每500篇报告编码进度；各集合编码后立即从磁盘按EOS逐篇解码核对全文。
输出/root/autodl-tmp/m3_candidate_tokens_16k_v1必须新建；已有输出先验收，不覆盖、不盲目重编码。
完整运行限15分钟，超时/失败保留诊断；缺少根manifest.json不算本轮完成，中间source_subset清单不是最终成功标志。
如果只有脚本目录已解包、编码尚未执行，核对传输清单后从目录执行同一timeout命令，不必重复上传/解包。

## 输出与本轮验收

- source_subset/：应用确切暂缓名单后的原样JSONL和清单；包含原来源/candidate标记。
- tokens/train.bin、valid.bin、test.bin：无文件头小端uint16，与同一16K tokenizer绑定。
- excluded_documents.json：3篇ID、正文摘要、原因和原集合，记录前后计数。
- encoding_acceptance.json：词表/EOS/merges、暂缓名单、固定划分/精确泄漏和三集合磁盘全文往返结果。
- scale_report.json：实际篇数/token数、正文token与EOS/LF计数、1亿处理tokens计划的算术使用倍数。
- 根manifest.json最后创建，绑定原v2输入、候选审阅、tokenizer、全部输出摘要与耗时。

用户贴回末尾JSON，将manifest.json、encoding_acceptance.json、scale_report.json和excluded_documents.json
四份小报告下载到PC的logs/m3_candidate_tokens_16k_v1。正文和bin继续留在云端，不必为了验收再传回大文件。
助手核对报告、输出摘要/边界计数及实际规模后停在本轮验收点，不继续启动训练。

“唯一训练token位置”指去重正文集合中的token位置数量（含文档边界），不是词表中不同ID个数。
1亿计划按batch16/context512每步8192计算，12208步实际100007936处理tokens；
处理tokens/训练集合token位置只是暴露量算术比值，现有随机窗口采样不是顺序遍历的epoch。
报告不把重复处理冒称1亿唯一语料tokens；实际训练范围要根据本轮真实规模另定。
ready_for_training仍为false，编码完成不替代语义/许可审阅或训练阶段批准，也不放行旧quarantine。

## 本轮本地准备验证

Windows定向34 passed（2.93秒），包括原编码测试和本轮暂缓/原样保留/归属/重复ID/候选状态检查。
本轮还用真实已验收16K tokenizer编码合成文档并从磁盘逐篇读回；另用真实257字节基线独立计算UTF-8预期ID。
没有训练300词表教程、重训BPE、填伪token或运行模型。
即使更新了磁盘文件校验和，测试中的正文token/边界损坏仍被全文解码或EOS/LF检查拒绝。
Ruff与ZIP逐文件读回通过，未重跑完整CPU回归；以上为准备轮检查，真实候选云端结果如下。

## 实际云端结果与本机验收

2026-10-14用户实际执行完成，54.100721秒；四份云端原始报告已下载至logs/m3_candidate_tokens_16k_v1。
本机verify_downloads.py实际30项核对PASS，Ruff通过，保存local_artifact_acceptance.json；未重跑模型或完整CPU回归。
本机重算三份子报告并与根清单摘要匹配，固定v2来源/审阅/tokenizer/15个源码摘要一致，暂缓名单3篇逐字段一致。
报告间集合篇数/EOS、uint16字节/token、ID范围、全文磁盘读回结果和规模算术一致，valid/test来源摘要保持不变。
正文JSONL和bin仍留云端；本机接受已核对脚本产生的云端全量读回报告，不冒称独立重新读回云端全文。

| 集合 | 文档/EOS | tokens | uint16字节 |
| --- | ---: | ---: | ---: |
| train | 6384 | 4449461 | 8898922 |
| valid | 68 | 46761 | 93522 |
| test | 77 | 53483 | 106966 |

共6529篇/4549705 token位置，train正文4436693 tokens，边界12768 tokens。
报告SHA-256：encoding_acceptance.json为bd18faf278624d4cb8440bfae404a8e557155c22350153608b1eba6910e37fb2，
scale_report.json为04b7e1c49572574a353ff3759ff848abdf1114b4ceb4a3806c0d2ddb6b043452，
excluded_documents.json为fae9fdb813fece7581429cd4cad711dc39ee8e382f91fd716487701c852a0373。
完整输入/输出摘要及根清单自身摘要记录在本机local_artifact_acceptance.json。

1亿计划实际100007936处理tokens/约22.476倍算术曝光；本批不按该旧规模默认开训。
下一项建议先2000步（16384000处理tokens/约3.682倍曝光），依据完整valid决定是否继续；
这是待落实的运行方案，不是训练结果、固定epoch或新增语料规模。
本任务已收尾，原ready_for_training=false不变；下一任务仅落实有界33M训练范围与配置，不再追加常规清洗循环。
