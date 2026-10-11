# M2 第九小步：正式数据版本与 BPE 采样预算

本轮只核对官方元数据并制定方案，不下载语料、不修改数据处理代码、
不保存正式 tokenizer、不编码 uint16、不训练模型。以下是待用户复核的执行方案，
不是正式语料已验收的声明；原 pilot 的数据、2048 tokenizer 和 checkpoint 保留不变。

## 1. 保留两个数据源，固定到有日期的版本

沿用项目的中文 Wikipedia + 中文 Wikisource 路线，不擅自改为单一来源。
2026-10-11 查询官方目录及 dumpstatus.json 后，选择 20261001 为本轮候选版本：
两份状态文件中所有已列出的 job 均为 done。后续不使用会变化的 latest URL。

| 项目 | 中文 Wikipedia | 中文 Wikisource |
| --- | --- | --- |
| dump 版本标识 | 20261001 | 20261001 |
| 候选文件 | zhwiki-20261001-pages-articles.xml.bz2 | zhwikisource-20261001-pages-articles.xml.bz2 |
| 对应 job | articlesdumprecombine | articlesdump |
| job 状态 | done | done |
| job updated 原值 | 2026-10-01 12:45:10 | 2026-10-02 00:29:06 |
| 压缩文件字节数 | 3430912430 | 9565159689 |
| 压缩文件大小 | 约 3.195 GiB | 约 8.908 GiB |
| 官方 SHA-1 | 04d5eec3c249fdf8604b97f4435e019c60a4ec94 | b1a6e0c5ae0b55692c41010a47f7eb02e8087e2a |
| 本地 SHA-256 | 未下载，待计算 | 未下载，待计算 |

来源：[Wikipedia 状态文件](https://dumps.wikimedia.org/zhwiki/20261001/dumpstatus.json)、
[Wikisource 状态文件](https://dumps.wikimedia.org/zhwikisource/20261001/dumpstatus.json)。
job updated 是服务端原值，不额外猜测时区；版本日期也不表示每篇文章的修订日期。

候选下载地址（仅记录，本轮没有下载）：

- [中文 Wikipedia 压缩 XML](https://dumps.wikimedia.org/zhwiki/20261001/zhwiki-20261001-pages-articles.xml.bz2)
- [中文 Wikisource 压缩 XML](https://dumps.wikimedia.org/zhwikisource/20261001/zhwikisource-20261001-pages-articles.xml.bz2)

两份压缩文件合计约 12.10 GiB，不含提取后的 JSONL、处理中间产物和 token 分片。
这不是所需磁盘空间或内存的完整估计，也不代表下载时长。进入下载小步前检查本地空间，
重新核对官方大小和 SHA-1；下载完成后验证它们，再计算本地 SHA-256，保存下载清单。
SHA-1 与 SHA-256 不能互相替代或从对方推算；失败或不完整文件不进入后续处理。

选择的是 pages-articles 的当前修订文本 dump，不是全历史 dump 或图片包。
它仍然是压缩 XML / 维基标记，不能直接交给现有 JSONL 入口或 BPE。
正式正文提取器尚未接入；需要分别验收两个站点的正文范围、重定向处理与标记清除。
官方也提醒 dump 并非整个站点完整、一致的备份，不能把 job=done 解读为没有缺失。
见 [Wikimedia Data dumps](https://meta.wikimedia.org/wiki/Data_dumps)。

## 2. 许可和来源信息不能省略

Wikipedia 原创文本的一般许可是 CC BY-SA 4.0 / GFDL，导入内容可能有不同适用条件；
保留署名所需的文章 URL / 历史入口及额外要求。Wikisource 包括公有领域和授权作品，
不能把所有作品统一标成公有领域。站点允许下载不等于已逐篇完成许可核对。
依据 [官方 dump 许可说明](https://dumps.wikimedia.org/legal.html)、
[Wikimedia 使用条款第 7 节](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)、
[中文 Wikisource 版权信息](https://zh.wikisource.org/zh-hans/Wikisource%3A%E7%89%88%E6%9D%83%E4%BF%A1%E6%81%AF/%E5%85%A8%E6%96%87)。

正式正文 JSONL 的来源字段至少计划保留：站点、dump 日期、标题、page ID、revision ID、
修订时间、原文 URL / 历史入口、适用许可及其依据；记录提取和繁转简等改动。
无法确认的许可标记为未确认，不伪造为已获授权，也不进入已验收的正式训练清单。
仓库 MIT 许可不代替语料条件，本步骤也不判定将来模型权重的法律许可。

当前 prepare 入口保留首次记录的来源信息，但重复正文不会汇总后续来源；
正式提取/去重时需另保留溯源记录，不能把“精确去重通过”当作“许可记录完整”。
Wikisource 还需区分作品正文、目录、校对页及章节：按正文哈希划分只防止完全相同的
文档跨集合，不自动保证同一本书的不同章节不会跨集合。这些是正式数据验收项，尚未完成。

## 3. BPE 第一版建议预算：16 MiB，只使用 train

| 项目 | 本轮建议 |
| --- | --- |
| 目标词表 | 16000，包含 256 初始字节、1 EOS、15743 次 merge |
| BPE 样本文本上限 | 16 MiB = 16777216 字节，UTF-8，包含每篇 EOS 和 LF |
| 可用集合 | 仅清洗、去重、长度过滤、固定划分后的 train |
| 文档边界 | 完整正文 + EOS + LF，EOS 字面量见下文 |
| 样本 seed | 42，与处理清单一起记录 |
| 推荐选择方式 | 独立采样哈希排序后的完整文档前缀，而不是源文件前缀 |
| 初次资源边界 | 沿用 300 秒、RLIMIT_AS=6 GiB；RSS 另行实测 |

EOS 的实际字面量为 `<|endoftext|>`，沿用原 pilot，不另改边界约定。

16 MiB 是基于已通过的 17677160 字节 pilot / 16K 探测提出的保守起点，
不是词表质量结论，也不保证正式文本相同耗时/内存；超过资源边界就报错，不能算成功。
限制的是 BPE 训练样本，不是整个正式语料库，也不是模型的训练 token 预算。
首轮模型训练仍按 README 计划的 1 亿 tokens，在 M3 实测吞吐量后决定。

建议下一模块实现以下确定性采样约定（本轮未实现）：

1. 先确认处理清单和整份 train 的版本、计数与归属，不读取 valid/test 正文。
2. 对每篇正文计算 SHA-256：UTF-8 编码的 `bpe-sample-v1\0{seed}\0{text}`。
   这里的 `\0` 表示 NUL 分隔符；与划分使用的哈希输入分离，便于各自版本化。
3. 按采样哈希排序，平局按 UTF-8 正文字节排序；输入重排不改变选择或输出顺序。
4. 在 16 MiB 内取完整文档前缀，遇到第一篇装不下的文档就停止，不截断也不跳选。
   首篇本身超预算时明确报错；需要调整上游长度规则或预算，不能悄悄产生空训练文本。
5. 不设置未经验证的 Wikipedia/Wikisource 固定配比；记录实际入选的各来源篇数和字节数。
   哈希选择减少文件顺序偏置，但不保证主题/来源均衡；若一个来源明显主导，先报告再调整策略。
6. 新目录输出 train.txt 和采样清单：来源哈希、seed、算法版本、预算、入选/总篇数、
   正文码点数、字节数及输出 SHA-256。文档不写标题/URL，溯源信息留在清单或配套记录。

现有 export_text 的 max_bytes 仍是**源文件顺序的完整文档前缀**，不能冒充上述采样。
新增模式必须有独立测试并保留旧行为，不能改写已验收 pilot 的导出结果。
正式语料中的唯一词块数、长文档分布以及 prepare 的 seen 内存开销仍需要另测。

训练并保存第一版正式 tokenizer 后，检查实际词表 16000、ID/merge/EOS、中文与 emoji 往返、
保存重载一致性、来源哈希及资源数字；用固定 validation 诊断 bytes/token 等压缩指标。
test 留作最终评测，不用它挑样本预算或 tokenizer。若质量不合适再独立修改预算/配比，
不能仅因达到 16000 就宣称词表质量达标。旧 2048 权重不能配新 16K tokenizer。

## 4. 本小步怎么复核

1. 核对上方两个官方状态链接的文件名、done、字节数和 SHA-1。元数据若变化，
   停止使用本方案中的旧数字，先重新固定版本；不要直接下载 latest。
2. 确认保持双来源路线；BPE 从 train 中选不超过 16 MiB，不使用全部正文训练 BPE。
3. 明确当前只验收方案：没有下载命令要执行，也没有正式 tokenizer 或 uint16 产物。
   复核后，下一小步实现有测试的哈希采样模式，再逐项验收下载和正文提取。

本轮只变更文档，未新增运行依赖或改变代码，因此不把之前的 pytest 成绩当作本轮重跑结果。

## 5. 复核后由你提交

在项目 PowerShell 执行，暂存四份文档，不暂存数据或无关根目录文件：

```powershell
git add README.md data/README.md docs/experiment-log.md docs/m2-formal-data-plan.md
git --no-pager diff --cached --stat
git --no-pager diff --cached
git commit -m "docs: plan pinned formal dumps and bounded BPE sample"
```

add 仅暂存这四份文件；两条 diff 分别检查暂存范围和完整内容；commit 保存本地快照。
本轮助手未执行 Git 写操作，也不自动 push。M2 尚未完成，M3 正式配置训练仍未启动。
