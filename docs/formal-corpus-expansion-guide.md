# 正式语料有界扩充：首分片继续批次1

当前入口为[current-work.md](current-work.md)。用户已授权“下一步”，本轮只产出一个新批次：
固定首分片继续范围＋旧train合并/旧heldout隔离＋新三集合与16K编码/磁盘读回＋实际规模与抽检材料。
不下载新分片、不安装依赖、不训练模型、不修改冻结源码或旧数据。

## 固定本批范围

| 项目 | 本批约定 |
| --- | --- |
| 原始输入 | 已下载的/root/autodl-tmp/m3_source_review_v1/raw/train-00000-of-00006.parquet |
| revision / SHA-256 | b04c8d1ceb2f5cd4588862100d08de323dccfbaa / 853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56 |
| 继续顺序 | 原row group独立哈希顺序，累计已处理80000篇之后 |
| 新读取上限 | 150000篇或512MiB原始正文，先到即停 |
| 新候选正文上限 | 128MiB完整规范化正文，不含继承的旧train |
| 选择时间 / 完整命令 | 600秒 / 15分钟 |
| 继承 | 旧6384篇train，全文与元数据保持不变 |
| 隔离 | 旧68篇valid＋77篇test，ID/规范化正文SHA/原文SHA三种键均保护 |
| 新划分 | 冻结的正文哈希seed42，旧train保持train，新增文章按同一规则划分 |
| tokenizer | 原实际16000词表，15743 merges，EOS256，固定摘要 |
| 新输出 | /root/autodl-tmp/m3_corpus_expansion_batch1_v1，只能新建 |

配置在configs/m3_corpus_expansion_batch1.json。5000万去重train token位置是阶段目标，
不承诺此首分片批次能达到，也不会把旧重复正文或处理次数算成新增语料。
本轮记录真实新增/总train token位置、目标比例、原1亿处理tokens计划的曝光比和继续位置。

只通过Parquet元数据跳过已处理的完整row group；若80000篇停止点位于group中间，
会重新读取该边界group的一小段前缀并跳过，保留原row_in_group坐标，不重扫前8万篇。
停止位置按实际已计数的新输入计算，字节边界未计入的文档仍留在下次继续位置，不猜测跳过。

复用三份已验收一次性脚本的原始字节：旧candidate_selection/prepare_and_verify、
旧v2 rejection_reasons和旧full_disk_readback；三份脚本与37个项目源码摘要均已核对。
旧机器规则、已知损坏暂缓ID、NFC/OpenCC1.4.2 t2s、精确去重和正文边界不变，不放宽审批门槛。
所有候选记录保留candidate/quality_reviewed=false/training_approved=false，最终ready_for_training=false。
新语料先产出完整候选三集合和编码供审阅，不自动开始训练；旧quarantine和其他未指定输入不混入。

## 用户上传小包，一次完成本批处理

本地包：logs/m3_corpus_expansion_prepare_20261014_v1/minigpt-cn-m3-corpus-expand-v1.zip。
32692字节，SHA-256=662b75bcf17678b469749411802088d8aa2188303c8213e80054b0444c4e8810。
包内只有接入脚本、三份只读复用脚本、来源/规则/审阅/输入绑定、许可证和传输清单，不传大语料或权重。
原云端环境已经有PyArrow26.0.0与OpenCC1.4.2；不安装或下载替代版本。
用户上传ZIP到/root/autodl-tmp后执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '662b75bcf17678b469749411802088d8aa2188303c8213e80054b0444c4e8810' \
  'minigpt-cn-m3-corpus-expand-v1.zip' | sha256sum -c -
if [ -e m3_corpus_expand_v1 ] || [ -L m3_corpus_expand_v1 ]; then
  echo '已有扩充脚本目录，先检查，不覆盖'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-corpus-expand-v1.zip .
cd m3_corpus_expand_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 15m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u \
  expand_corpus_once.py
SH
```

每1000篇新输入、每500篇编码打印进度；完成后输出根manifest JSON。
脚本与结果目录名分别m3_corpus_expand_v1和m3_corpus_expansion_batch1_v1，互不覆盖。
已有脚本目录但尚未运行时，先核对传输清单，再从该目录执行同一timeout命令，不重复解包。
已有结果目录先检查产物，不能改名绕过诊断或盲目重处理。
失败/超时保留中间文件；缺少根manifest.json不算完整成功，嵌套candidate_splits清单不代表本批完成或可训练。

## 输出与本轮返回材料

云端结果包含new_selection原始筛选记录、合并/排除台账、全部候选JSONL、新三集合、
同版本无头小端uint16 token文件和逐篇磁盘全文往返结果；原分片和旧目录只读。
本轮新test是新三集合固定留出的部分，不重新评估已见test，不使用任何test分数指导数据选择。

用户贴回末尾JSON，并下载5份小文件到PC的D:\\aiworkspace\\transformer\\logs\\m3_corpus_expansion_batch1_v1：

- manifest.json：来源、旧输入/规则版本、本批范围、停止原因/继续位置、各文件摘要、三集合与磁盘读回结果。
- scale_report.json：实际train/valid/test token位置、新增train量、5000万阶段目标比例与1亿计划曝光比。
- isolation_report.json：旧68/77 heldout隔离和重复排除计数。
- review_samples.jsonl：按独立固定哈希保留的最多24篇**新增**候选完整正文。
- review.md：同一批完整正文的可读材料，未截短或挑选展示。

候选JSONL/bin继续留在云端，不必下载大文件。本机核对报告/摘要/继续位置/隔离与实际新增量，读完新全文样本后收尾本批。
目标未达时按本批真实产量决定下一次固定范围；不自动下载余下五个分片、不全站扫描、不靠重复正文凑5000万。
抽检不是逐篇事实/许可完整性的证明，损坏样本只作明确暂缓，不猜词补正文或循环扩建质量工具。

## 本轮实际本地验证

2026-10-14 Windows显式CPU定向63 passed（3.22秒），Ruff与ZIP逐文件读回通过。
覆盖独立元数据定位/边界坐标、不读取已完成group、三键heldout保护、旧train/新增精确去重、
既有拒绝规则、真实prepare完整三集合读回、真实16K编码与磁盘全文解码、状态篡改/已有输出/无新增候选拒绝。
绑定44份既有云端输入/源码，37个源码与3份一次性脚本字节摘要一致；不改旧脚本或src。
本机未读取完整真实Parquet或运行本批云端处理；元数据继续测试使用合成group，真实处理待用户执行。
未重跑完整CPU回归或WSL检查，不冒称已新增正式token规模。

## 实际批次结果与回传验收

2026-10-14用户已完成云端处理并下载五份原始小文件至logs/m3_corpus_expansion_batch1_v1。
云端200.562847秒（选择53.233632秒），新增读取147147篇/536870642原始正文字节，触及512MiB预算后停止。
原机器候选11914篇，旧v2拒绝规则等暂缓934篇，新增保留10980篇；继承6384篇train，合计17364篇/44114965规范化正文字节。

| 集合 | 文档/EOS | token位置 | uint16字节 |
| --- | ---: | ---: | ---: |
| train | 17140 | 11872985 | 23745970 |
| valid | 125 | 93084 | 186168 |
| test | 99 | 88115 | 176230 |

实际新增train token位置7423524，总train达5000万目标的23.74597%；原1亿处理计划在此规模下仍约8.423倍曝光。
旧68篇valid＋77篇test的三键隔离报告通过，无旧heldout混入；本轮新增valid/test尚未用于模型评估。
云端全部JSONL读回、跨集合ID/正文精确去重、三集合磁盘逐篇全文解码PASS。

本机verify_artifacts.py实际31项核对PASS、Ruff通过；保存artifact_acceptance.json。
包括原始小文件摘要/来源/策略/绑定、计数与预算、合并/隔离、新增量/目标比例、uint16字节/ID/EOS与云端读回报告一致性，
以及全部24篇全文的摘要、规范化/原候选状态、旧机器规则、固定样本哈希顺序和Markdown完整重建一致性。
本机未重算云端大JSONL/bin摘要、未读取真实Parquet元数据或重跑模型；下一group与旧顺序对应已核对，group内偏移接受已核对脚本的报告。

累计固定顺序处理227147/230792篇，首分片余3645；继续位置order_position227、row_group12、row_in_group355，未把未计入文档丢弃。
下一批按原数据元数据重新定位此位置，不猜测跳过剩余正文。

24篇新增全文已审阅，6篇明确暂缓（均train）：

| ID | 标题 | 可见问题 |
| --- | --- | --- |
| 797533 | 石井街道 (广州市) | 首句主语/隶属字段缺失，行政区划列表只剩句号 |
| 751280 | 轉拴式槍栓 | 出现“枪栓上的并且完成”的缺词连接结构 |
| 14155 | 698年 | 马库里亚等领袖条目姓名字段缺失 |
| 14091 | 646年 | 多个领袖条目只有国家/职位/年份，缺姓名 |
| 736054 | 2003年的市町村合併列表 | 合并式存在“甲西町+=”悬空加号，不猜测修补 |
| 669960 | 賦值向量環 | 定义/性质中核心数学符号和公式大量缺失 |

确切ID/URL/正文及原文SHA-256保存于验收记录hold_records，供下一次合并时匹配排除；不新增泛化清洗规则或补抽样循环。
原manifest/JSONL/bin未修改，ready_for_training仍为false；17134/125/99只是应用6条暂缓后的拟文档数，未实际重编码，token数仍按上表原产物记录。
其他样本仍有外链/分类残留、空标题和措辞风险；全文抽检不保证逐篇事实正确或全部内容完整。
本批候选准备/编码技术验收已收尾，下一项按固定清单获取第2分片并准备下一有界批次，
顺带使用首分片余3645篇继续位置，应用这6条确切暂缓并保留全部heldout隔离；本轮未执行新下载/处理/训练。
