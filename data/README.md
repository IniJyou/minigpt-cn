# 数据目录

此目录只保存数据说明和清单，不保存原始语料、tokenized shard 或缓存。

## 计划数据源

- 中文 Wikipedia：使用正式训练时最新、已经完整发布的 dump。
- 中文 Wikisource：使用正式训练时最新、已经完整发布的 dump。

正式下载前必须记录：

| 字段 | 内容 |
| --- | --- |
| 数据集名称 | 待填写 |
| 发布日期 | 待填写 |
| 下载 URL | 待填写 |
| 许可 | 待核实并填写 |
| 文件大小 | 待填写 |
| SHA-256 | 待填写 |

## 预处理约定

```text
解析 → Unicode NFC → 可选繁转简 → 清理空白/过滤空文档 → 文档级精确去重 → 长度过滤 → hash 分割
```

- 固定随机种子：42。
- 按文档划分 train/validation/test = 98%/1%/1%。
- tokenizer 只能使用 train split 训练。
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
全量处理、去重、划分、编码和来源清单仍待接入。后续清单须记录转换开关、字典版本及长度阈值；
旧 pilot 数据、tokenizer 和 checkpoint 不重写。独立复核见 [清洗教程](../docs/m2-cleaning-guide.md)。
