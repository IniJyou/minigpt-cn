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
