# M2：流式XML解析与小样本复核

本步只读两份已下载的本地压缩包，各扫描前100页，生成原始wikitext样本和来源清单。
**不下载、不全量提取、不渲染模板、不繁转简/去重/划分、不训练BPE或编码token。**
后续所有下载由用户亲自执行；助手本轮仅查看官方格式说明，未下载语料或安装依赖。

## 1. 放在哪里，为什么先保留wikitext

- src/minigpt_cn/data/parse_xml.py：独立的流式读取、小样本输出和本地输入校验入口。
- tests/test_parse_xml.py：37项离线小文件测试。
- data/processed下的样本属于忽略的本地产物，不提交Git。

XML 0.11的主文本是revision/text里的维基标记，不是渲染后的纯正文；辅助content/text不能误取。
依据见 [MediaWiki导出说明](https://www.mediawiki.org/wiki/Help:Export)与
[官方0.11结构](https://www.mediawiki.org/xml/export-0.11.xsd)。因此输出使用wikitext而不是text字段，
现有prepare入口要求text字段，不能直接把本样本送去清洗划分或训练。

每条记录保留原始标题、namespace、page_id、revision_id、timestamp、wikitext及其SHA-256，
source.name/dump_date、页面URL、固定修订URL、历史入口，以及来源许可说明/待逐篇核对状态。
XML只按标准解码一层实体，例如`&amp;`变为`&`；不额外做HTML反转义，不清理{{模板}}、[[链接]]或表格。
不导出编辑者用户名/IP，署名溯源保留页面/历史入口；不把来源一般许可当逐篇许可已核实。

仅保留namespace=0、非XML重定向、主槽位wikitext/text/x-wiki格式的非空文本。
其他命名空间、无修订/无文本、非wikitext、删除文本、外部location文本、空文本分别计数。
外部内容URL绝不跟随或下载；当前只接受中文export-0.11且siteinfo/dbname匹配下载来源。
这不是完整XSD验证器；目录、消歧、模板为主的页以及作品/章节关系仍待下一步判定。

## 2. 流式原理与安全边界

直接用bz2.open按块解压，不先写一个完整解压XML文件；每次最多向解析器提供16KiB。
ElementTree在page的end事件才交付完整一页，随后clear内容并从根节点remove，
避免只clear却累积数百万个空page节点。原理见 [Python ElementTree](https://docs.python.org/3.12/library/xml.etree.elementtree.html)。

内存关系约为O(最大单页XML + 固定解析/解压缓冲)，不是O(整个dump)。巨型单页仍可能占较多内存；
默认解压预算64MiB，触顶则失败，不能当截断正文成功。这里处理bytes/页面记录，不是模型张量。
默认扫描100页（含被过滤页），max_pages范围1～10000；到页边界停止，记录stop_reason=page_limit。
此时xml_document_complete=false正常，只说明没有读完后面的XML，不代表压缩包校验失败。

输入必须是完整的verified下载清单；检查固定文件名、两份快照及其摘要，解析前后各重算
**整份压缩包**的大小/SHA-1/SHA-256。不全量解压，但会全量校验压缩字节，耗时主要在这些读取。
失败保留本次诊断输出，缺完整sample_manifest.json不可算成功；新目录以x方式创建，不覆盖旧样本。
保守拒绝NUL编码及DTD/ENTITY声明；不展开自定义实体，不读取外部实体或外部文本。

计数恒等式：pages_scanned = saved + sum(skipped)。样本若没有可保存页就报错，不写成功清单。
输出仅wikitext.jsonl和sample_manifest.json；scope=prefix_sample、ready_for_training=false。

## 3. 在WSL项目环境复核

```bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_parse_xml.py -q --tb=short
python -m ruff check .
```

预期37 passed、Ruff通过。助手本次完整CPU回归为Windows513 passed/9 skipped，WSL517 passed/5 skipped。
跳过项目包括CUDA及Windows符号链接权限测试，不是本模块测试被跳过。

由用户在两个**新的样本目录**运行以下单行命令（都是本地解析，不联网）：

```bash
python -m minigpt_cn.data.parse_xml --download-manifest data/raw/zhwiki_20261001_v1/download_manifest.json --output-dir data/processed/m2_user_xml_zhwiki100_v1 --max-pages 100
python -m minigpt_cn.data.parse_xml --download-manifest data/raw/zhwikisource_20261001_v1/download_manifest.json --output-dir data/processed/m2_user_xml_zhwikisource100_v1 --max-pages 100
```

同名输出目录若已存在，换v2，不删除旧样本。助手自己的样本在m2_verification/xml_zhwiki100_v1
与m2_verification/xml_zhwikisource100_v1，不会被上述用户命令覆盖。

| 来源 | 扫描 | 保存 | 非0命名空间 | 重定向 | 解压XML读取字节 | JSONL字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Wikipedia | 100 | 67 | 30 | 3 | 5210112 | 4242482 |
| Wikisource | 100 | 81 | 6 | 13 | 1458176 | 1435883 |

其他跳过计数均为0，stop_reason=page_limit，xml_document_complete=false，ready_for_training=false。
Wikipedia输出SHA-256：2e4d24749ceafe3b9eb1f94466b750cdc990247e23e0b06ed6b5a7463ed03331。
Wikisource输出SHA-256：51d7fc4c8deeb63ffbd8afdadbd47143c2c78b84ba0b27a6d933805ae9b47632。
摘要只指wikitext.jsonl；样本清单包含耗时/路径，不要求清单整体逐字节重复。

查看两份sample_manifest.json并各抽看几条记录，确认source、标题、ID、原始标记和许可待核对状态。
助手已用独立XMLPullParser逐项对比两份样本的所有保存记录，标题/两种ID/时间/wikitext一致。
端到端耗时（含前后整份压缩包校验）34.66/96.78秒；同一WSL进程累计峰值RSS约36.26MiB，
不能把它当两个独立进程的各自峰值，更不能推算全量清洗、去重或BPE的资源成本。

## 4. 复核后由你提交

上一轮续传补丁尚未提交，本轮文档已经同时引用它和新XML入口；以下一组显式暂存相关文件，
不暂存data/raw、data/processed、缓存、学习笔记或根目录无关文件。

```powershell
git add README.md data/README.md docs/m2-formal-data-plan.md docs/m2-download-guide.md docs/m2-xml-sample-guide.md docs/experiment-log.md src/minigpt_cn/data/__init__.py src/minigpt_cn/data/download.py src/minigpt_cn/data/parse_xml.py tests/test_download_dump.py tests/test_download_resume.py tests/test_parse_xml.py
git --no-pager diff --cached --stat
git --no-pager diff --cached
git commit -m "feat: stream bounded XML samples and support interrupted dump downloads"
```

add仅暂存列出的代码/测试/文档；两条diff分别看范围与具体内容；commit保存本地快照。
所有Git写操作仍由用户执行。复核后停在此提交点；下一小步再做维基标记到纯正文的清理测试。
M2未全部完成，正式16K词表、全量正文/许可与作品级划分、uint16编码和M3训练仍待验收。
