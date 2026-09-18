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
解析 → Unicode NFC → 清理空白 → 文档级精确去重 → 长度过滤 → hash 分割
```

- 固定随机种子：42。
- 按文档划分 train/validation/test = 98%/1%/1%。
- tokenizer 只能使用 train split 训练。
- token ID 以 `uint16` 二进制分片保存，并通过 memory map 读取。

本地目录 `data/raw/` 和 `data/processed/` 已由 `.gitignore` 排除。
