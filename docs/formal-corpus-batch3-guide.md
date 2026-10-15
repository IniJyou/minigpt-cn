# 剩余固定分片有界扩充批次3

2026-10-14。用户已授权下一项，单批处理第3至第6个已上传的固定中文分片，应用批次2四条暂缓、累计heldout隔离、合并及16K编码/全文磁盘读回。
接入包已完成，实际云端运行待用户执行；不下载/安装或训练模型，不修改冻结工具和旧产物。

## 本批范围

- 固定wikimedia/wikipedia、20231101.zh，revision=b04c8d1ceb2f5cd4588862100d08de323dccfbaa。
- 来源仅train-00002至train-00005-of-00006.parquet，逐文件摘要沿用configs/wikipedia_20231101_zh_source.json。
- 首两个分片已耗尽；每个后续来源从固定哈希group顺序第0行开始，真实元数据和全部起止位置记录在报告中。
- 四来源共用上限：100万篇新输入、2GiB原始正文、256MiB新增候选正文、选择600秒；完整处理命令限30分钟。
- 因预算停止时保留当前来源继续位置，不跳过正文读取后一个来源；候选字节边界读取但未输出的文档留给下一轮。
- 继承批次2的train，精确移除1482014/1172443/1037822/1901806，必须匹配ID/标题/URL/正文及原文摘要。
- 同版16K扣除5057个token，继承20316941个train token/30421篇；用户运行后才产生实际新JSONL/bin。
- 累计346篇valid/303篇test按ID、规范化正文及原文摘要隔离于所有新三集合，不使用test分数指导任何数据/训练选择。
- 复用原质量、繁转简、正文/ID去重、seed42划分、16K编码和全文读回，不增加泛化过滤规则。

剩余4个来源和选择预算内实际可用产量决定能否达到5000万阶段目标。不足时报告真实差额/停止原因，不重复正文凑数、不自动另找来源或启动训练。
原候选status/candidate、quality_reviewed=false、training_approved=false、ready_for_training=false均保持。

## 用户上传并一次执行

包：logs/m3_corpus_batch3_prepare_20261014_v1/minigpt-cn-m3-corpus-batch3-v1.zip，48706字节。
SHA-256=17c6c3f03010641ca609333af2c42e20eeb0b4c00fc9a37f32d6628317eba0a4。
只有接入脚本、冻结函数副本、许可/审阅/策略及版本绑定，复用现有项目环境和云端数据，不上传权重或大语料。
运行前核对57份既有输入/源码（含全部6个Parquet）；缺失或摘要不符只报告具体阻塞，不下载补救或放宽旧门槛。
PyArrow26.0.0/OpenCC1.4.2与同版16K仍必须一致，无新依赖。

用户把ZIP上传至/root/autodl-tmp，执行整段：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '17c6c3f03010641ca609333af2c42e20eeb0b4c00fc9a37f32d6628317eba0a4' \
  'minigpt-cn-m3-corpus-batch3-v1.zip' | sha256sum -c -
if [ -e m3_corpus_batch3_v1 ] || [ -L m3_corpus_batch3_v1 ] || \
   [ -e m3_corpus_expansion_batch3_v1 ] || [ -L m3_corpus_expansion_batch3_v1 ]; then
  echo '已有本批脚本或结果目录，先检查，不覆盖或盲目重处理'
  exit 1
fi
python -m zipfile -e minigpt-cn-m3-corpus-batch3-v1.zip .
cd m3_corpus_batch3_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout --signal=TERM 30m \
  /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/python -u prepare_batch3.py
SH
```

只解包过而未运行时，先核对ZIP/包内传输清单，再执行最后的timeout命令。
已有结果目录先核对完整性，不能改名绕过诊断重处理；失败/超时保留中间文件，缺根manifest.json不算完整批次。
每1000篇新输入/500篇编码有进度。先完成全部候选三集合与全文磁盘读回，才保存根manifest。

## 一次取回本批验收材料

输出/root/autodl-tmp/m3_corpus_expansion_batch3_v1。
程序将5份小文件自动打包为同目录m3_batch3_reports_v1.zip：

- manifest.json
- scale_report.json
- isolation_report.json
- review_samples.jsonl
- review.md

用户下载这个报告ZIP，解压到PC的D:\aiworkspace\transformer\logs\m3_corpus_expansion_batch3_v1。
已有同名文件先检查，不覆盖；不必下载云端大JSONL/bin。
本机核对完整来源、继续位置、聚合预算、规模/隔离/摘要，阅读24篇新增全文后收尾；不把仅控制台末尾片段当成完整产物。
本批达到5000万也只是实际候选规模，仍需本批验收和明确训练配置；不自动开训或宣称效果。

## 本轮实际准备验证

Windows显式CPU定向52 passed（3.69秒），Ruff、包内CLI帮助、ZIP逐文件读回通过。
验证准确4条暂缓、3代heldout的三键保护、继承train交叉拒绝、4来源摘要归属/共享预算/未输出边界保留、已有输出拒绝。
复用真实16K编码与全部三集合全文磁盘解码；37个冻结源码与既有函数副本字节不变。
Parquet选择测试使用合成group，不冒称实际云端处理；本机未读取真实Parquet元数据、重跑完整CPU/WSL回归或启动模型。
本轮准备记录见logs/m3_corpus_batch3_prepare_20261014_v1/local_verification.json与delivery_manifest.json。

## 实际云端结果与本机验收（本批已收尾）

2026-10-14用户回传并解压五份原始小文件至logs/m3_corpus_expansion_batch3_v1。
本轮46项核对与Ruff通过，24篇全文已完整阅读；记录在artifact_acceptance.json。
根manifest44108字节，SHA-256=349b22608f207b065d76fb8e633813cf6702d5f3dbb20119cff35e3f4865e260。
原始小文件摘要、全部来源/绑定/冻结规则、共享预算、按报告group元数据独立复算起止坐标、合并/隔离、EOS/uint16计数、全文样本摘要/排序/Markdown内容一致。

| 集合 | 文档 | 实际token位置 |
| --- | ---: | ---: |
| train | 73528 | 48202076 |
| valid | 461 | 297390 |
| test | 450 | 251517 |

继承20316941个train token，上批4条暂缓5057已扣；本批新增27885135。
剩余4分片各230791篇全部读完，共923164篇/1347285233原始正文字节；选择154.044793秒，完整处理763.003549秒。
新机器候选49154篇，保留44018、暂缓5135、去掉1条重复，加上旧train30421，共74439篇。
六个固定分片累计1384748源记录均耗尽。5000万阶段目标仅达96.404152%，还差1797924；明确记录未达到，不用伪token或重复正文凑数。
1亿计划处理tokens在当前规模约2.0748倍算术曝光，不能称已执行的epoch或效果。
云端三集合逐篇全文磁盘读回报告PASS、旧346/303 heldout隔离通过；本机未实际重读云端JSONL/bin或真实Parquet元数据，未重跑完整CPU/WSL回归。
新heldout461/450保留，下一状态累计保护valid807/test753。

8篇明确暂缓全部train，下一次训练输入冻结时按确切ID、标题、URL、规范化正文和原文摘要匹配剔除，不新增泛化规则：

| ID | 标题 | 可见问题 |
| --- | --- | --- |
| 5575868 | 长春公交54路 | 车站一览段完全空缺 |
| 2954494 | 陈学良 | 人物短传后接入泛论清官的编号段，关联/完整性异常 |
| 4069965 | 多丰铁路 | “天桥至铁路”终点名称缺项 |
| 8098260 | 2022年8—9月深圳市2019冠状病毒病聚集性疫情 | 深圳确诊时序/统计图表及声称“具体如下”的措施清单缺失 |
| 7155618 | 圣灵医院 (奥格斯堡) | 院徽段所称“以下村庄”名单缺失 |
| 3905452 | Tanh函数展开法 | 方程/变量代换/积分及最终解大量公式缺失 |
| 5445840 | 2016年香港立法會選舉民意調查 | 核心民调结果表缺失，备注标识空缺 |
| 6239167 | 菲律宾火山地震研究所地震烈度表 | 声称列出罗马数字烈度等级，列表完全缺失 |

其他16篇仍有翻译、史实、分类和时效风险，不等于逐篇事实或许可批准。
6910196豐原素的人名再次OpenCC t2s会从“陈昇明”变成“陈升明”；原来源规范化只转换一次，本轮保存正文不变，不强求二次转换固定点。
同版16K实际复算本批8条拟扣14082个train token，拟冻结73520篇/48187994 token；当前原JSONL/bin仍未改，不能把拟值说成已有编码。
候选技术验收PASS、原ready_for_training=false保持。建议下一项按实际约4819万规模准备输入冻结和明确的100M处理tokens训练范围（拟曝光约2.0754倍），不为了整数目标自动另找来源。
本轮停在验收点，未启动输入冻结、训练准备或模型运行。
