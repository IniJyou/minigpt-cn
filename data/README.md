# 数据来源与产物状态

**当前任务见 [工作交接](../docs/current-work.md)。16K tokenizer及pilot编码已验收，数据工具扩建冻结。**
本页保存来源和状态，不要求重新下载、全站解析或逐项重跑历史实验。
工具接口见 [数据参考](../docs/data-reference.md)。

## 当前可复用：pilot train

已验收pilot有1000篇文档，train/valid/test=986/7/7。
处理清单：data/processed/m2_user_pilot_jsonl/manifest.json。
BPE训练文本：data/processed/m2_user_bpe_16k_probe_text/train.txt；
同目录export_manifest.json记录来源、train归属和摘要。

- train文本986篇、17677160字节，含正文后的EOS/LF。
- SHA-256：0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
- 对应train.jsonl SHA-256：1250226874b95e47ef9f900140a83c315e420886558902cb23c4ca61346b3699。
- 这批只用于16K与33M训练闭环验证，不是正式1亿token规模或正式语料许可验收。
- 旧tokenizer为2048，旧pilot checkpoint为3.74M模型；原文件不覆盖。
- 16K实际词表16000（含EOS）、merges15743、EOS ID256，已保存并通过重载/往返验收。

其他机器或新会话先核对实际路径/清单/摘要。输入缺失时报告具体缺项，不自行下载或
回到全量维基流水线。不要用validation/test文本训练tokenizer。

## 已验收：同版本16K token数据

tokenizer：data/processed/pilot_tokenizer_16k_v1.json；SHA-256
21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470。
新token目录：data/processed/pilot_tokens_16k_v1，含三份无头小端uint16、manifest.json和acceptance.json。

| 集合 | 文档/EOS | tokens | 字节 | 输出SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 986 | 4168196 | 8336392 | e5675c806734fca49a0f6d9bed2ae6282ba1652a73ee570ff15885b4eaa3354e |
| valid | 7 | 25272 | 50544 | 8bce0b106fc9f55e7a4d9cfbd063150a38f1ba4e5a1c5488727cb3368759ccfb |
| test | 7 | 42263 | 84526 | 4c91e07cb838e89baf783cd5146f57c9575415b68c2d4748e20671c900b6e631 |

实际编码51.57秒；全部1000篇磁盘解码与来源正文/EOS/LF一致，train拼接摘要与BPE输入一致。
操作及独立只读验收见 [Token编码验收](../docs/token-encoding-guide.md)。不覆盖旧数据、不提交新产物。

## 后续正式来源：已下载，不等于可训练

20261001固定版本的中文Wikipedia和Wikisource压缩包已通过本地完整性复核。
下载配置为configs/wikimedia_20261001.yaml，原始目录分别为
data/raw/zhwiki_20261001_v1、data/raw/zhwikisource_20261001_v1。
每个目录保存download_manifest.json、配置/官方状态快照和压缩包；所有后续下载由用户执行。

| 来源 | 压缩字节 | 官方SHA-1 | 本地SHA-256 |
| --- | ---: | --- | --- |
| zhwiki | 3430912430 | 04d5eec3c249fdf8604b97f4435e019c60a4ec94 | 5db9052efe9ca97aa3da05ffe5b2f12c8ffe5e040f0a9d1ced82ec2fda5d7e2f |
| zhwikisource | 9565159689 | b1a6e0c5ae0b55692c41010a47f7eb02e8087e2a | 8433426aee0ad8c00121a0bd057182543fef9bb12e2fb4fbb8f06258e091b703 |

来源URL和许可声明保留在固定配置/下载快照中。许可依据：
[官方dump说明](https://dumps.wikimedia.org/legal.html)、
[Wikimedia使用条款](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)。
Wikisource的作品/译本仍需分别核对；不自动把全站标为公有领域。
未来正式语料按明确预算处理，记录来源和必要署名/许可要求，不以全站完成为前置条件。

## 已核对的正式候选：中文Parquet首分片

wikimedia/wikipedia、20231101.zh，revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa。
用户在云端下载首分片train-00000-of-00006.parquet，587109027字节，完整SHA-256为
853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56。
云端读取报告230792行、231个row group；24篇首尾摘录及原始报告已取回logs/m3_source_review_20261013_v1，
本轮15项本地核对通过，实际样本审阅及限制见[正式来源验收](../docs/formal-source-guide.md)。

摘录可见空括号、空学名、转换残留及名称/列表缺项；来源可留作候选，整批正文不能直接训练。
数据集许可声明cc-by-sa-3.0/gfdl原样记录，后续选用范围保留id/url/title、日期/revision和来源摘要。
ready_for_training仍为false；来源核对不替代正式正文筛选、划分、编码或逐文档审批。
旧20261001 raw、旧候选与pilot保持原状，另外五个中文分片未下载。

有界候选v1已在云端实际生成：7086篇/19754320正文UTF-8字节，三集合6929/73/84篇。
本机报告与24篇完整样本在logs/m3_corpus_candidates_v1，21项核对通过；抽检仍见损坏/残片，不能直接训练。
本轮仅对既有7086篇做保守拒绝补筛，另存v2，不改变原v1或旧数据；步骤见[正式候选三集合](../docs/formal-corpus-guide.md)。
v2已完成：6532篇/16656130正文UTF-8字节，三集合6387/68/77篇；本机26项核对通过并已读24篇完整样本。
候选准备验收记录在logs/m3_corpus_candidates_v2/candidate_preparation_acceptance.json，列出3篇编码前暂缓ID/正文摘要。
实际编码6529篇（6384/68/77），云端54.101秒完成；原candidate/ready_for_training=false保持不变，不自动审批旧候选。
云端目录/root/autodl-tmp/m3_candidate_tokens_16k_v1；本机4份原始报告及30项核对结果在logs/m3_candidate_tokens_16k_v1。

| 集合 | 文档/EOS | tokens | uint16字节 | 云端输出SHA-256 |
| --- | ---: | ---: | ---: | --- |
| train | 6384 | 4449461 | 8898922 | 40d54ce51ebfc70be3519cc7cc3947f82593746a764eaf7b712b149c18377c27 |
| valid | 68 | 46761 | 93522 | 21cd94950967f4e37cc1352936d7716a044b413099c53f2cdc2ac6ec6acb1182 |
| test | 77 | 53483 | 106966 | b941993cba98193c2eb43c5c4fd2a92e4d260be9cda3dd3d6391e89af335e040 |

云端逐篇磁盘解码PASS；本机重算小报告摘要并核对一致性，未重新读取云端bin/全文。
实际train正文tokens为4436693，其余12768为EOS/LF边界；1亿处理tokens约22.48倍曝光，不能冒称唯一语料规模。

首分片扩充批次1随后完成：累计读227147/230792篇，本批新增147147篇，合并后17364篇（17140/125/99）。
实际train/valid/test token位置11872985/93084/88115，新增train7423524；云端三集合全文磁盘读回PASS。
本机五份小报告及31项核对记录在logs/m3_corpus_expansion_batch1_v1，完整新增样本24篇已读，明确暂缓6篇ID/摘要见artifact_acceptance.json。
6篇尚未从本批JSONL/bin实际剔除，原candidate/ready_for_training=false保持不变；以上token数是暂缓前的真实产物规模。
首分片还剩3645篇，继续位置order_position227/row_group12/row_in_group355；详情见[本批结果](../docs/formal-corpus-expansion-guide.md#实际批次结果与回传验收)。
固定第2分片与批次2执行包已准备，见[批次2步骤](../docs/formal-corpus-batch2-guide.md)。尚未下载或生成新云端产物；准备轮准确复算6条暂缓扣除4321个token，继承train基线11868664/17134篇。
该包累计保护193篇valid/176篇test，复用冻结规则并顺带处理首分片尾部，不自动下载后续分片或训练；旧JSONL/bin不修改。
用户现已完成批次2并回传五份原始小文件，41项核对通过；train/valid/test=20321998/125251/64320个token，新增train8453334。
首分片和第2分片固定顺序均耗尽；本机后5个Parquet文件的大小/SHA-256已实际复核，分片属于固定20231101.zh版本，不能当成已审批训练数据。
24篇全文已读，4条明确缺项（1482014/1172443/1037822/1901806）均train，记录在logs/m3_corpus_expansion_batch2_v1/artifact_acceptance.json，待下次合并时剔除。
同版16K拟扣5057 token，20316941只是扣除后的拟train基线，尚无新JSONL/bin。累计待保护heldout为346篇valid/303篇test，原ready_for_training=false不变。
批次3随后完成并验收五份原始小文件/24篇全文，46项核对与Ruff通过。实际train/valid/test=48202076/297390/251517个token，新增train27885135。
六个固定中文分片均按固定顺序读完，累计1384748篇来源记录；原5000万规划目标达96.404152%，仍差1797924，不能冒称已达到或已训练。
本批8条确切暂缓全部train，ID/正文/原文摘要与证据见logs/m3_corpus_expansion_batch3_v1/artifact_acceptance.json；只记录名单，原JSONL/bin保持。
同版16K拟扣14082 token，拟冻结train48187994/73520篇；这是后续输入冻结的计算，尚非新产物。当前valid/test461/450保留，累计旧heldout807/753继续隔离。
后续建议按实际规模冻结训练输入和范围，保留来源/版本/许可声明与候选状态，不自动逐篇批准或凑整数新增来源。
输入冻结与GPU预检查已实际完成，四份原始报告保存至logs/m3_pretraining_input_v1，40项本机核对与Ruff通过，见[运行准备](../docs/pretraining-100m-guide.md#实际冻结与预检查验收)。
云端23.399秒按原EOS/LF边界准确剔除8篇/14082 token，保留其他原行和token字节；冻结train实际48187994/73520篇，valid/test源JSONL与bin摘要/大小均与批次3一致。
保存后三集合逐篇磁盘全文解码报告PASS，旧346/303 heldout三键隔离报告PASS；本机核对小报告/配置/tokenizer/源码摘要与8篇扣量，未重新读取云端大JSONL/bin。
冻结根清单SHA-256=60ff2f6cdca5b240803e4fa63fad31a87905c6907414dd12f125dbc3677682fe；实验范围单独绑定12208步/100007936处理tokens，原候选ready/质量/审批标志保持。
大语料/bin继续留云端，不提交Git。用户随后完成固定100M训练，六份原始模型产物和CPU重载已验收，31项核对/Ruff通过；最终12208步/100007936处理tokens、valid loss=3.9629。
最终权重在logs/m3_training_report/m3_pretrain_100m_return_v1/ckpt_final.pt，本机实际SHA-256=f563104f903b0b174a355055d965a33e1c815602124627fdfeab5494524fad2d，与云端报告一致。
用户随后取回冻结valid/test至data/processed/m3_final_eval_inputs_v1，两份本机大小/SHA实际与清单一致，未重新编码或读取云端train bin。
最终模型本地完整valid/test与12条固定生成产物已验收，31项核对通过；valid/test loss=3.962893/3.995780。12条全部审阅仍明显重复/离题，生成质量未通过展示要求，不把权重重载或低loss当成展示效果通过。

## 保留作诊断，不能直接训练

- data/processed/m2_body_zhwiki_full_v1：用户主动中断的全量尝试，无成功manifest；
  不代表全量提取完成，不自动重启或补写成功清单。
- data/processed/m2_verification/interrupted_zhwiki_subset32m_v2：9667篇待审候选，
  正文33553167字节；只证明子集选择/记录一致性，不证明内容完整或许可获批。
- data/processed/m2_verification/zhwiki_quality32m_v2：自动规则候选3741、暂缓3872、
  排除2054；全部仍未批准，机器候选可能缺词，不能勾选整批后直接训练。
- 已保存的XML原始小样本67/81条：wikitext不是训练正文；可只读复核新提取器。
- 提取器local_wikitext_body_v2的修复不会恢复旧正文中已丢失的字；
  需要新数据时从原始文本重新按预算提取，不能事后猜测补词。

上述文件保留，不删除原始下载、旧pilot、tokenizer、checkpoint或诊断产物。
当前已有32MiB子集的ready_for_training仍为false；不放宽代码中的审批门槛。

## 编码与划分约定

固定seed=42、train/valid/test期望98%/1%/1%；小样本实际篇数不是精确比例。
已验收pilot沿用正文级哈希划分，未来作品语料需要共同work_id和跨集合泄漏检查。
训练BPE只读取train；正文边界为正文 + <|endoftext|> + LF。
模型输入文件为无.npy文件头的小端uint16 token ID，需匹配同一tokenizer；
新16K编码入口为minigpt_cn.data.encode，产物已验收；模型读取新目录的bin，不能直接读取train.txt。
data/raw/、data/processed/均由.gitignore排除，不提交数据。
