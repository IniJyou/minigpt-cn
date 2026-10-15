# 固定第2分片与有界扩充批次2

2026-10-14。本轮交付用户执行包：获取同revision第2分片，顺带处理首分片余3645篇，合并、编码并逐篇读回；未启动模型训练。

包为logs/m3_corpus_batch2_prepare_20261014_v1/minigpt-cn-m3-corpus-batch2-v1.zip，43966字节，
SHA-256：5ba14a76a9001b6067f8d9cc677bd7419aabf4ee1509f0e54ad92b9e2e17cdb0。

## 固定范围

- 数据集wikimedia/wikipedia，20231101.zh，revision=b04c8d1ceb2f5cd4588862100d08de323dccfbaa。
- 第2分片train-00001-of-00006.parquet：263629009字节，SHA-256=038b08e5aae38b5a1671794cee3fa0d4952c0d00fa0890a903d2de3f3fe5f703。
- 首分片从固定顺序累计227147行继续，用实际Parquet元数据复核order_position227/row_group12/row_in_group355，随后才读第2分片。
- 两来源合计最多250000篇新输入、1GiB原始正文、256MiB新增候选正文、选择600秒；处理命令限15分钟，下载不设过短总传输时限。
- 批次1的6条暂缓按ID、标题、URL、规范化及原文SHA-256准确匹配移除，冻结规则不扩建。
- 实际同版本16K复算：6条合计4321个train token（含EOS/换行边界），继承基线11868664；17134篇train。此扣除量已本机算出，尚无新的云端合并/bin产物。
- 原68/77与批次1新增125/99篇valid/test累计193/176篇，三键隔离于全部新三集合；不使用旧test分数决定数据或训练。

只复用现有PyArrow26.0.0、OpenCC1.4.2、项目环境与16K tokenizer。旧源码、脚本、语料和权重只读，ZIP绑定48份既有输入/源码。
重复或缺失记录不会靠伪token/重复正文补足阶段目标；候选状态与ready_for_training=false保持，语义审阅待本批结果返回。

## 上传后一次执行

用户把ZIP上传至/root/autodl-tmp，然后执行整段。该命令只下载第2个固定分片和处理本批。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '5ba14a76a9001b6067f8d9cc677bd7419aabf4ee1509f0e54ad92b9e2e17cdb0' \
  'minigpt-cn-m3-corpus-batch2-v1.zip' | sha256sum -c -
if [ -e m3_corpus_batch2_v1 ] || [ -L m3_corpus_batch2_v1 ]; then
  echo '已有脚本目录，先核对，不覆盖或重复解包'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-corpus-batch2-v1.zip .
cd m3_corpus_batch2_v1
bash run_batch2.sh
SH
```

已有完整分片先核对长度和摘要，不覆盖；.part未完整时保留并断点续传，最多4次，失败仍保留文件。
完整长度但摘要不符则阻塞，不盲目覆盖。下载前使用已存在的/etc/network_turbo，只有用户执行时才联网。
已有本批结果目录会在下载前拒绝；不能改名绕过已有诊断重处理。未执行的脚本目录如已解包，先核对ZIP及传输清单，再执行bash run_batch2.sh。

云端结果目录为/root/autodl-tmp/m3_corpus_expansion_batch2_v1。
每1000篇输入/500篇编码有进度；断点和每来源实际row group大小/顺序均记录，候选字节预算边界的未输出文档保留给下一次。
失败或超时保留诊断；缺少根manifest.json不算整批成功，嵌套三集合清单不代表训练批准。

## 返回验收

贴回末尾JSON，下载下列5份小文件至PC的D:\aiworkspace\transformer\logs\m3_corpus_expansion_batch2_v1：

- manifest.json
- scale_report.json
- isolation_report.json
- review_samples.jsonl
- review.md

大JSONL/bin继续留云端。以实际新增train token、继续位置、隔离结果和24篇新增完整样本验收收尾，不自动获取后续分片或启动训练。
5000万train token是阶段目标，当前未达到；今天是否完成取决于网络和实际可用产量。

## 本轮实际准备验证

Windows显式CPU定向46 passed（2.75秒），Ruff通过；WSL仅bash -n语法检查成功，未运行WSL pytest/本批处理。
覆盖6条准确暂缓、累计两代heldout的ID/正文/原文摘要隔离、多来源共享预算、来源摘要归属、元数据位置拒绝、预算边界留存、部分下载/损坏文件保留，以及复用真实16K编码和三集合全文磁盘往返。
ZIP逐文件读回、包内CLI帮助、37个冻结源码及复用脚本字节核对通过。记录见logs/m3_corpus_batch2_prepare_20261014_v1/delivery_manifest.json。
未运行完整CPU回归；助手未下载第2分片、读取真实第2分片元数据、运行本批云端处理或新增模型训练。

## 一次下载全部固定中文分片

用户随后授权一次下载齐全，获取范围改为同revision全部6个中文分片，总1721150260字节（约1.72GB）。
第一个已完整下载后，其他5个合计最多1134041233字节；部分文件的实际剩余量更少。
只改变获取范围：批次2仍按前述两来源与预算处理，其余下载文件保留供后续使用，不自动扫描/编码全量或训练。

下载脚本为logs/m3_all_shards_download_20261014_v1/download_all_pinned.py。
用户上传该小脚本至/root/autodl-tmp，然后执行：

```bash
bash -e <<'SH'
cd /root/autodl-tmp
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo
fi
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u download_all_pinned.py
SH
```

复用已解包的m3_corpus_batch2_v1/source.json并核对固定清单SHA-256。
已有完整文件先核对字节/SHA-256后跳过；部分文件断点续传，每个最多4次，无过短总传输时限。
完整但损坏、超长或符号链接会拒绝，保留原文件。中断后重跑同一命令即可，避免两个下载进程同时写同一个.part。
末尾全部6个分片PASS后，继续此前bash run_batch2.sh；如果批次2已有结果则先验收，不盲目重复处理。
脚本语法与Ruff本机核对通过，助手未执行网络下载；以云端实际6个文件摘要结果为准。

## PC下载后上传（当前选择）

用户已选择Windows PC下载，再上传云端。云端首分片已PASS，PC只下载后5个，总1134041233字节；没有重下首分片的必要。
脚本在logs/m3_local_shards_transfer_20261014_v1/download_on_pc.py，默认镜像，固定来源清单及逐文件大小/SHA-256不变。
先停止云端仍在运行的下载，保留其.part，然后在PC PowerShell执行：

```powershell
Set-Location -LiteralPath D:\aiworkspace\transformer
.\.venv\Scripts\python.exe -u .\logs\m3_local_shards_transfer_20261014_v1\download_on_pc.py
```

输出为data/raw/m3_wikipedia_20231101_zh。中断后重跑同一命令，已有完整文件先核对后跳过，部分文件续传，损坏文件保留并拒绝覆盖。
如需改官方地址，只在同一命令末尾添加--endpoint official；更换传输地址不改变固定版本或摘要。
全部5个PASS后，上传5个完整.parquet（不上传.part）至/root/autodl-tmp/m3_shards_upload_v1，
并上传本地logs/m3_local_shards_transfer_20261014_v1/verify_and_promote_on_cloud.py至/root/autodl-tmp。
接收目录在云端先执行mkdir -p -- /root/autodl-tmp/m3_shards_upload_v1。已有同名接收文件先检查，不选择覆盖。

```bash
/root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python \
  /root/autodl-tmp/verify_and_promote_on_cloud.py
```

先核对全部6个文件才接入；已有raw完整文件必须正确，否则阻塞不覆盖。
新文件通过同盘硬链接加入raw，接收副本和.part都保留；目标出现同名文件则硬链接拒绝覆盖。
末尾全部6个固定分片就绪后，按上文当前批次2命令继续，原ZIP/处理范围不变；已有批次2输出先验收。
Windows离线mock核对续传URL/偏移、已完整跳过、损坏保留、真实合成文件硬链接不覆盖及固定清单；未执行实际下载/上传。

## 用户回传的报告尾部（原始产物待验收）

2026-10-14用户提供末尾json_m3_batch2.txt，12366字节，SHA-256=889940ce2ed08f2f762834d3b7e5bbe6e9d80d20df3c676d774325c385f3260b。
文本从selection中段开始，缺少根JSON头部、来源身份/两个元数据继续位置/策略/绑定信息，不当成完整manifest。
可独立核对的29项报告内算术/元数据通过，记录在logs/m3_batch2_report_review_20261014_v1/report_review.json。
报告合并30705篇（train/valid/test=30425/153/127），实际token位置20321998/125251/64320。
同版16K、6条准确暂缓、4321 token扣除、继承11868664、新增8453334及EOS/uint16字节计数一致。
达5000万阶段目标40.643996%，还差29678002；计划1亿处理tokens在此规模下约4.921倍算术曝光，不据此自动开训。
三集合全文磁盘读回报告PASS，耗时336.033850秒；尚未取回五份原始小文件或审阅本批24篇全文。
当前只取回manifest/scale_report/isolation_report/review_samples/review.md到logs/m3_corpus_expansion_batch2_v1完成原验收，不重跑本批或自动新建下一批。

## 五份原始文件实际验收（本批已收尾）

2026-10-14用户已回传五份原始文件，41项核对通过，24篇新增全文全部读完；Ruff通过。
完整根manifest27789字节，SHA-256=9debc7b376010b77e5f51c94cfc357c8ed1b404ec9b3df7860bb2784358cabf9。
原文件摘要、固定来源/规则/绑定、合并计数、完整三集合读回报告、同版16K、EOS/uint16计数、累计heldout隔离、全文样本摘要/排序/Markdown内容与旧贴文尾部一致。
原始产物train/valid/test为30425/153/127篇，20321998/125251/64320个token；本批新增8453334个train token，达5000万目标40.643996%。
两来源合计本批读234437篇/415236631原始正文字节，选择71.435884秒，完整处理336.033850秒。
首分片尾3645和第2分片230792均已耗尽；分别从绑定位置与第0行开始，按报告实际group大小/固定哈希顺序独立复算坐标一致。
原193/176 heldout隔离报告通过；本批新增153/127，下一批累计保护346篇valid/303篇test。
本机后5个原始Parquet共1134041233字节，逐文件实际大小/SHA-256均与固定来源清单相同。
本机未装PyArrow，未实际解码这些Parquet元数据；云端全量JSONL/bin读回仅核对报告，不冒称在本机重跑。未重跑完整CPU或WSL pytest。

4篇可见缺项暂缓（均train），确切ID、标题、URL、规范化正文/原文摘要保存在artifact_acceptance.json的hold_records：

| ID | 标题 | 可见问题 |
| --- | --- | --- |
| 1482014 | 第15號交響曲 (蕭士塔高維奇) | 开头地点中文字段缺失、配器“响板、、乐鞭”空项 |
| 1172443 | 伊拉斯莫斯世界計劃 | 德国/澳洲/瑞典多处学校清单只剩国家名 |
| 1037822 | 贝尔纲定理 | 证明段变量/公式大量缺失，多处仅剩句号 |
| 1901806 | 螺山镇 | 首句隶属缺项、行政区划列表只剩句号 |

其他20篇不作逐篇事实/许可完整性批准；仍有疑似错字、标题/参考尾部残留和具体步骤缺失风险。
同版16K实际复算4条拟扣5057个train token，拟文档数30421/153/127，拟继承train基线20316941。
这是下一次合并的扣除记录，不是已改好的新编码；原报告/JSONL/bin均不修改，ready_for_training=false仍保持。
本批技术验收收尾，下一项复用后续固定分片，在有界合并时应用确切4条暂缓、累计heldout保护；不新增泛化清洗规则或单独补筛循环，不自动训练。
