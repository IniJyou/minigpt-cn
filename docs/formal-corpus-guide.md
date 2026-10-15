# M3有界正文候选与三集合验收

当前任务见[current-work.md](current-work.md)。本轮仅处理已下载中文首分片，不新增下载/安装，不编码或训练。
来源已核对，质量审阅发现正文缺项；通过规则的正文仍为candidate，不冒称逐文档批准。

## 固定范围与复用

- 输入：/root/autodl-tmp/m3_source_review_v1/raw/train-00000-of-00006.parquet。
- 来源revision/大小/完整SHA-256与[来源验收](formal-source-guide.md)一致；处理前后复核摘要。
- 最多处理80000篇、512MiB原始正文，保留最多128MiB完整规范化候选正文。
- 正文范围300～50000码点；不切文档、不越过边界后跳选短文；记录实际停止原因和数量。
- row group按独立固定哈希顺序读取，避免仅取最早文章；不完整扫描230792篇，不宣称全分片质量比例。
- 复用冻结的quality.classify、clean_document、prepare_jsonl和seed42正文哈希划分；10个源码文件与原500步包摘要相同。
- 在原规则之外保守暂缓空学名等带标签空括号、空书名/引号、已发现损坏的文章ID和正文中的保留EOS。
- 不补词、不删除括号伪装修好、不批准旧quarantine/eligible，不建设维基渲染或审批平台。
- 候选选择限600秒；用户命令对完整处理限15分钟，超时保留诊断，不自动重跑或扩大预算。

配置保存为configs/m3_corpus_candidate_policy.json。候选先NFC/OpenCC1.4.2 t2s规范化，
prepare再复用其去重/划分逻辑并关闭重复繁转简；其processing.to_simplified=false只描述prepare阶段，
前面的转换记录在每篇preprocessing和本轮根manifest中。标题/URL/原始正文摘要保留。

## 上传小包并执行一次

本机包：logs/m3_corpus_prepare_20261013_v1/minigpt-cn-m3-corpus-prepare-v1.zip。
15533字节，SHA-256：7c1c7f2a9d83f0d24a455d0956ef37087a185aa5c44792d5b3210816666a2e65。
包内仅本轮脚本、固定策略、来源元数据、前轮审阅结论、README、代码许可证和传输清单。
数据保持在原目录，旧500步项目/checkpoint/源码不修改；复用既有.venv和PyArrow26.0.0。

用户上传ZIP至/root/autodl-tmp，然后执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '7c1c7f2a9d83f0d24a455d0956ef37087a185aa5c44792d5b3210816666a2e65' \
  'minigpt-cn-m3-corpus-prepare-v1.zip' | sha256sum -c -
if [ -e m3_corpus_prepare_v1 ] || [ -L m3_corpus_prepare_v1 ]; then
  echo '已有检查包目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-corpus-prepare-v1.zip .
cd m3_corpus_prepare_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 15m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python \
  prepare_corpus_once.py
SH
```

每1000篇打印进度；末尾打印根manifest.json。没有重新安装依赖的步骤。
输出/root/autodl-tmp/m3_corpus_candidates_v1必须新建，已有输出先检查，不覆盖、不盲目再做。
失败/超时保留所有部分文件；缺少根manifest.json不算完成，candidate_splits/manifest.json单独存在也不代表本轮完成。
若只有上传包目录已经成功解包、尚未运行处理，先核对传输清单后从该目录执行同一timeout命令即可；不用再上传或解包。

## 返回本轮验收材料

用户贴回末尾JSON，并下载输出目录中的manifest.json、review_samples.jsonl、review.md至
PC的logs/m3_corpus_candidates_v1。三份train/valid/test JSONL保留云端，不必再传大量正文回PC。
review_samples.jsonl保存最多24篇完整候选正文，review.md只提供前1600字预览；来源/质量复核读完整样本。

程序会读回全部三集合，核对完整记录（包含全部来源元数据）、正文摘要、文章ID、固定归属、非空集合、
跨集合ID/全文精确泄漏及真实篇数/字符/字节。这个读回验证不代表语义和许可已逐篇批准。
所有记录保留status=candidate、quality_reviewed=false、training_approved=false；根manifest的ready_for_training=false。
本轮得到机器筛选和候选三集合后，还需助手读取回传完整样本并记录验收结论。
不要将candidate_splits/manifest.json直接交给旧pilot编码入口；正式编码入口接入和训练是后续任务。

## 本轮本地准备验证

Windows定向检查82 passed/1 skipped（最终2.57秒），本轮脚本和项目Ruff通过。
检查覆盖损坏正文拒绝、规范化去重、预算停点、候选不自动批准、真实prepare三集合读回和已有输出保护。
ZIP已逐文件读回，固定10个源码摘要匹配原云端包；实际Parquet处理尚待用户在云端执行。
未重跑完整CPU回归，未执行真实正文选择/编码/训练；临时脚本/测试/检查包与数据均在Git忽略目录。

## 实际v1结果与本轮补筛

用户已完成v1运行并取回三份原始材料：云端35.2687秒（选择32.2608秒），处理80000篇，
原始正文326170945字节，保留7086篇/19754320规范化正文字节。
候选train/valid/test=6929/73/84篇，正文UTF-8字节分别19294886/193013/266421。
这19.75MB正文不是1亿唯一tokens，具体唯一tokens和重复次数在后续编码后确定，不冒称正式规模达标。

本机21项报告/摘要/规则/计数/样本核对通过；24篇完整样本已阅读，仍有表格行残片、句子缺名词、
人员/列表/公式缺项，机器规则过筛没有达到直接收录条件。原始文件不改，记录在
logs/m3_corpus_candidates_v1/artifact_review_v1.json。云端完整三集合读回是实际脚本报告，本机没有下载全部正文重跑它。

补筛仅读取原selected_candidates.jsonl（26919104字节），暂缓已发现10篇和明显同类风险记录，
不修文本、不加新来源、不重读Parquet、不放宽原质量规则。保留记录的原始来源/摘要/row group原样复制，
继续复用原prepare和全部输出读回；本地11项定向验证通过，Ruff及包读回通过，云端补筛尚未执行。

用户上传logs/m3_corpus_postfilter_20261013_v2/minigpt-cn-m3-corpus-reject-v2.zip至/root/autodl-tmp。
包5578字节，SHA-256：b6828f17b34ff1f8b3fb7f779a900bb6248a7bc5defe3fb859b56fbb0eb6f50e。
在云端执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  'b6828f17b34ff1f8b3fb7f779a900bb6248a7bc5defe3fb859b56fbb0eb6f50e' \
  'minigpt-cn-m3-corpus-reject-v2.zip' | sha256sum -c -
if [ -e m3_corpus_reject_v2 ] || [ -L m3_corpus_reject_v2 ]; then
  echo '已有补筛脚本目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-corpus-reject-v2.zip .
cd m3_corpus_reject_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 5m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python \
  reject_candidates_once.py
SH
```

输出/root/autodl-tmp/m3_corpus_candidates_v2，原v1完整保留；已有v2拒绝覆盖，缺少新根manifest不算完成。
成功后贴回末尾JSON，将新manifest.json、review_samples.jsonl和review.md下载到PC的logs/m3_corpus_candidates_v2。
候选仍ready_for_training=false，助手阅读完整样本后记录本轮结论再进入下一项；不重复原v1命令或扩大扫描。

## v2验收与本轮收尾（2026-10-14）

用户已完成补筛并下载三份原始材料；实际4.5849秒，7086篇中暂缓554篇，保留6532篇/16656130正文UTF-8字节。
候选train/valid/test=6387/68/77篇，正文UTF-8字节16281548/170171/204411。
本机26项核对通过：v1来源摘要、拒绝策略、计数/预算、样本原文摘要、来源元数据、未批准状态、
旧规则/新规则、样本顺序、Markdown重建及下载文件摘要；云端完整三集合读回报告通过，本机没有下载全部正文重跑。

24篇完整新样本已阅读。仍明确暂缓卜塔（id950724，地名原文缺项）、1448年（id14968，事件句缺项）、
花巷基督教堂（id251723，英文名称缺失）。ID与对应正文SHA-256及所在集合保存到
logs/m3_corpus_candidates_v2/candidate_preparation_acceptance.json；其他样本保留所见观察，不承诺全批没有文本损失或事实问题。

候选准备任务已完成，结论为PASS并附3篇编码前暂缓名单；不再新建补筛包或反复补抽样。
下一轮在16K候选编码入口应用该ID/正文摘要名单并记录前后计数，固定集合不重分。
拟编码范围6529篇（6384/68/77），还未实际输出/编码；原候选三集合、报告及其ready_for_training=false原样保留。
真实唯一token量尚未统计，这批只是有界中文文本实验，不能冒称1亿唯一tokens或逐篇许可/质量保证。

必要检查保留：固定来源/许可记录、文件摘要、tokenizer版本、EOS/ID边界、集合归属和泄漏、完整读回。
样本中的明确损坏记录暂缓，不以追求每句完美为由继续扩大规则和平台。
两次云端处理合计39.8535秒，按1.58元/时仅作处理时间算术估计约0.0175元；不含下载/环境准备/等待，不代表账单。
