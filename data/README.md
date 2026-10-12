# 数据目录

此目录只保存数据说明和清单，不保存原始语料、tokenized shard 或缓存。

## 计划数据源

- 中文 Wikipedia、中文 Wikisource：下载前查询最新完整发布的 dump，固定有日期的 URL，
  不用会变化的 latest；本轮版本均为 20261001。两份压缩包已下载并通过独立完整性复核；
  两者全量正文与逐篇许可均未验收。后续所有下载由用户亲自执行。

2026-10-11 已只读核对官方状态元数据；本地 SHA-256 必须在下载完成后另行计算，
以下不是正式训练数据已验收清单：

| 字段 | 中文 Wikipedia | 中文 Wikisource |
| --- | --- | --- |
| dump 版本 | 20261001 | 20261001 |
| 下载 URL | [pages-articles.xml.bz2](https://dumps.wikimedia.org/zhwiki/20261001/zhwiki-20261001-pages-articles.xml.bz2) | [pages-articles.xml.bz2](https://dumps.wikimedia.org/zhwikisource/20261001/zhwikisource-20261001-pages-articles.xml.bz2) |
| 官方状态 | [articlesdumprecombine: done](https://dumps.wikimedia.org/zhwiki/20261001/dumpstatus.json) | [articlesdump: done](https://dumps.wikimedia.org/zhwikisource/20261001/dumpstatus.json) |
| 压缩文件字节数 | 3430912430 | 9565159689 |
| 官方 SHA-1 | 04d5eec3c249fdf8604b97f4435e019c60a4ec94 | b1a6e0c5ae0b55692c41010a47f7eb02e8087e2a |
| 本地 SHA-256 | 5db9052efe9ca97aa3da05ffe5b2f12c8ffe5e040f0a9d1ced82ec2fda5d7e2f | 8433426aee0ad8c00121a0bd057182543fef9bb12e2fb4fbb8f06258e091b703 |
| 许可状态 | 一般文本 CC BY-SA 4.0 / GFDL；附加条件需保留 | 公有领域 / 授权作品需核对，不能全部标为公有领域 |

许可依据见 [官方 dump 说明](https://dumps.wikimedia.org/legal.html)与
[Wikimedia 使用条款](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)。
计划保留页面/修订 ID、原文及历史 URL、适用许可和处理改动；未确认的记录不计入正式验收数据。
压缩文件合计约 12.10 GiB，不含正文 JSONL 与分片；全量正文提取和正式规模的内存验收仍未完成。
完整来源依据、限制与 BPE 建议预算见 [第九小步方案](../docs/m2-formal-data-plan.md)。

下载入口已接入，固定配置为configs/wikimedia_20261001.yaml；2026-10-11再次查询的官方元数据
与上表一致。默认离线预览，`--check-remote`只查状态JSON，显式`--download`才传输压缩文件。
小文件离线测试和真实元数据预览已通过；两份本地SHA-256已独立重算并匹配下载清单，如上表。
详见 [下载入口教程](../docs/m2-download-guide.md)；实际传输中断后已补上显式`--download --resume`。

实际下载需新目录并检查压缩文件大小加1GiB余量，分块写.part，核对本地大小和官方SHA-1，
再保存本地SHA-256及配置/官方状态快照，完整download_manifest.json最后不覆盖定名。
status=verified仅表示压缩文件完整性通过，ready_for_training=false；逐篇许可仍待核对。
默认3次有限网络重试，可用`--retries`指定0～20次；显式续传仅追加匹配旧快照的.part，
不覆盖完整dump或成功清单。失败保留诊断文件，缺完整清单不能进入正文提取。

2026-10-11，Wikipedia从已有171835020字节续传完成，总计3430912430字节，SHA-1匹配官方，
助手独立重算两个整文件摘要并确认旧前缀和快照未变；本次续传含校验耗时658.30秒。
本地清单为data/raw/zhwiki_20261001_v1/download_manifest.json，status=verified，
ready_for_training=false；这是压缩文件验收，不是正文提取/正式训练数据验收。

2026-10-12已复核用户完成的Wikisource下载：9565159689字节，两个整文件摘要和两份快照匹配。
清单为data/raw/zhwikisource_20261001_v1/download_manifest.json，两份均status=verified。

## 已接入：流式XML与原始维基文本小样本

`python -m minigpt_cn.data.parse_xml`接受本地完整下载清单，默认扫描前100页、解压预算64MiB，
只生成新目录中的wikitext.jsonl与sample_manifest.json；不联网，不保存完整解压XML。
16KiB读取缓冲，page结束后clear并remove，保留标题/页面和修订ID/时间/URL/来源与许可待核对状态。
只保留主命名空间的非重定向wikitext；XML实体按标准解码，维基模板/链接/表格不清除。
使用wikitext而非text字段，不能直接喂现有prepare入口、BPE或模型；ready_for_training=false。
Wikisource目录/章节关系和逐篇许可仍需后续核对，不能把main namespace全部当作品正文。

两份前100页小样本保存67/81页，37项离线测试通过；复核步骤见
[流式XML小样本教程](../docs/m2-xml-sample-guide.md)。仅小样本已验收，不是全量纯正文验收。

## 预处理约定

```text
解析 → Unicode NFC → 可选繁转简 → 清理空白/过滤空文档 → 文档级精确去重 → 长度过滤 → hash 分割
```

- 固定随机种子：42。
- 按文档划分 train/validation/test = 98%/1%/1%。
- tokenizer 只能使用 train split 训练。
- 正式 BPE 首版建议限 16 MiB（包含 EOS/LF）；export_text 已支持独立固定哈希选完整文档，
  需显式启用 selection=hash。默认仍取源文件顺序的前缀；两种模式均不保证来源均衡。
- token ID 以 `uint16` 二进制分片保存，并通过 memory map 读取。

本地目录 `data/raw/` 和 `data/processed/` 已由 `.gitignore` 排除。

## 已接入：单篇文档清洗

`src/minigpt_cn/data/cleaning.py` 提供 `normalize_text` 和 `clean_document`。
规范化顺序与已验收 pilot 相同：NFC、繁转简、CRLF/CR 统一为 LF、逐行 strip、整体 strip。
繁转简默认开启，可传 `to_simplified=False` 保留繁体；使用与 pilot 相同的
`OpenCC==1.4.2` / `t2s.json`（[官方 Python 用法](https://github.com/BYVoid/OpenCC#python)）。

保留行内空格、tab、段落空行、全角字符、标点、英文和 emoji；不做 NFKC 或正文格式重写。
`clean_document` 默认只过滤空文档，返回 None 表示过滤；可选长度上下限包含边界，
超长文档整篇过滤，不截断。长度按规范化正文的 Unicode 码点数计算，包含保留的空白，
不是字节数或 token 数；正式语料的长度阈值尚未决定。

本模块不解析 Wikipedia/Wikisource dump 或 HTML，不读写文件；输入必须已经是正文字符串。
去重和划分函数见下一节，JSONL入口和处理清单见末节；原始XML小样本入口见上节，纯正文与编码仍待接入。
旧 pilot 数据、tokenizer 和 checkpoint 不重写。独立复核见 [清洗教程](../docs/m2-cleaning-guide.md)。

## 已接入：正文精确去重与固定划分

`src/minigpt_cn/data/splitting.py` 提供 `deduplicate_texts` 和 `choose_split`：

- 输入是已清洗的非空正文；先过滤 clean_document 返回的 None，不隐式再次清洗。
- 按完整正文是否相等去重，保留第一次出现的正文和输入顺序；不同标题/来源的相同正文也算重复。
- 划分规则与原 pilot 相同：`int(SHA256(UTF8(f"{seed}\0{text}")), 16) % 100`。
  固定 seed=42，桶0～97为train，98为valid，99为test；valid对应validation。
- 同一正文固定属于一个集合，重排/追加输入不改变已有正文归属。不使用Python内置hash。
- 98/1/1是桶的期望比例，不是精确文档数或token数比例；小样本允许验证/测试集为空，
  后续全量CLI须检查并报告，不能擅自把训练文档挪入空集合。

此函数层只处理正文；下述JSONL入口保留首次记录的来源信息。
去重输入只遍历一次，但seen仍保存全部唯一正文，未验收正式大语料的内存开销。
详见 [去重与划分教程](../docs/m2-splitting-guide.md)。

## 已接入：JSONL 处理入口与清单

`python -m minigpt_cn.data.prepare` 接受已提取正文的UTF-8 JSONL：每个非空行必须是JSON对象，
text字段必须是字符串；不解析XML dump、HTML或维基标记。可重复指定`--input`，按参数顺序处理。

- 输出目录必须不存在，生成train.jsonl、valid.jsonl、test.jsonl和manifest.json；已有目录即使为空也拒绝复用。
- 默认NFC、繁转简、清理空白，过滤空正文后全局精确去重，再按可选长度阈值过滤、固定hash划分。
  仅替换text字段；首次记录的标题、URL、来源、许可及其他JSON字段保留。后续重复记录不聚合来源。
- min_chars默认1、max_chars默认无上限；seed默认42。`--keep-traditional`关闭繁转简。
- 清单记录输入路径/字节数/SHA-256、处理规则/字典版本/阈值、各类过滤计数，以及输出文件的文档数、码点数、字节数和SHA-256。
- 全部检查通过后才写清单；非法输入报告文件与行号，输入版本变化或任一空split均报错。
  失败可能保留本次新建的部分文件，不自动删除；缺少完整清单的输出不可用于训练，修复后换新目录重跑。

这是逐行文件入口，不是目录级事务或正式大语料资源验收；seen仍保存全部唯一正文。
处理清单追踪本地文件与规则，不自动核实下载来源/版本/许可；下载来源摘要已填写本页来源表。
此入口不导出纯文本或token二进制，不改旧pilot文件，详见 [JSONL处理教程](../docs/m2-data-prepare-guide.md)。

## 已接入：仅 train 的 BPE 训练文本导出

`python -m minigpt_cn.data.export_text` 接受已验收的处理清单，固定读取其同目录的train.jsonl，
不开放split选项，不读取valid/test正文，不跟随train文件符号链接或允许清单重定向来源文件名。
导出前核对train字节数/SHA-256，逐行核对归属与完整文档/字符计数，结束时再次检查来源版本。

- 输出新目录中的train.txt和export_manifest.json，不覆盖旧文件，也不改原处理清单。
- 只写原text正文，文档顺序不变；沿用原pilot边界：`正文 + <|endoftext|> + LF`。
  不再次清洗，不写标题/URL。正文已有保留EOS字面量时拒绝导出，不悄悄替换或过滤。
- 默认导出全train；可选`--max-bytes`只取容得下的完整文档前缀，预算包括EOS/LF。
  不截断文档/UTF-8，不跳过超预算文档再挑后面的小文档；即使限量，仍校验整份train。
- 导出清单记录来源处理清单/训练文件哈希、抽样规则、边界规则、实际篇数/正文码点数/字节数/文本哈希。
  失败无完整导出清单的文本不可用于BPE，需换新目录重跑。清单不证明来源许可或代表性。

1MiB前缀复现旧pilot样本，不是正式16K语料规模验收；既有2048词表/权重不改，后续BPE训练必须将
`<|endoftext|>`声明为特殊token（现有CLI默认如此）。该文本不是uint16训练数据，不能直接喂模型。
详见 [训练文本导出教程](../docs/m2-train-text-guide.md)。

## 已接入：有预算的确定性哈希采样

沿用同一export_text入口，新增`--selection hash --sample-seed 42 --max-bytes 16777216`；
hash必须给正整数字节预算，seed默认42，只改变采样，不改变train/valid/test归属。
按SHA-256（bpe-sample-v1、NUL、十进制seed、NUL、正文）的值排序，平局按UTF-8正文字节；
取此顺序的完整文档前缀，装不下第一篇就报错，不截断、不跳选。输入重排不改变输出文本。

扫描完整train并核对来源哈希/计数/归属后，写新的train.txt与export_manifest.json；
不读取valid/test正文，不改变旧prefix模式或清单。正文不再次清洗，仍追加EOS/LF。
最大堆仅保留预算附近的候选，但单篇巨文档和Python对象开销不受硬RSS限制，正式规模未测。
新清单记录采样算法版本/seed/规则，以及入选来源的篇数、码点数、字节数（含EOS/LF）；
来源支持source字符串或source.name，缺失/其他结构为null，不从URL猜测来源或许可。

pilot复核942篇/16742929字节，与独立全排序及重复导出逐字节一致；旧1MiB前缀哈希未变。
哈希采样本身不证明正文/许可或16K tokenizer已交付；详见 [哈希采样教程](../docs/m2-hash-sampling-guide.md)。
