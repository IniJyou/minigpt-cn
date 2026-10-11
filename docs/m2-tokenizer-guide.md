# M2 第二小步：小语料 tokenizer 训练与保存 CLI

第一小步的 `train_bpe` 已复核。这一小步只接命令行、保存和重载检查，不下载/清洗数据，
不编码token分片，不训练模型，也不训练正式16K词表。

## 代码在哪里

- `src/minigpt_cn/tokenizer/train_bpe.py`：第一小步的原BPE算法，本轮不改。
- `src/minigpt_cn/tokenizer/train.py`：参数检查、调用训练、保存重载与结果报告。
- `src/minigpt_cn/tokenizer/bpe.py`：save增加可选overwrite参数；原默认行为不变，CLI显式拒绝覆盖。
- `tests/test_train_tokenizer_cli.py`：小语料、独立进程入口、不覆盖、错误输入与重载失败测试。

## 1. 激活项目环境并检查

以下全部在WSL运行，不使用作业的Python环境：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_train_tokenizer_cli.py tests/test_train_bpe.py tests/test_tokenizer.py -q
python -m ruff check .
```

WSL应为48 passed；Windows没有符号链接创建权限时为47 passed、1 skipped。
不覆盖测试包括“训练期间另一调用者创建同名输出”和“已存在的悬空符号链接”。

## 2. 只训练一个300词表的入口验收样本

使用已经验收的原pilot训练文本，输入只读，产物另存到项目忽略目录：

```bash
python -m minigpt_cn.tokenizer.train \
  --input /mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/bpe_train.txt \
  --vocab-size 300 \
  --output data/processed/tokenizer_m2_user300.json
```

这份约1MiB文本来自原pilot的train split；只用于入口演练，不作为正式16K词表来源。
JSONL不是纯文本输入，CLI不自动提取text字段，也不接受bin/npy文件。
后缀检查不能证明数据内容或划分正确；后续数据管线会明确输出训练文本与数据清单。

报告应包含：

- requested_vocab_size=300、actual_vocab_size=300、merges=43。
- special_token_ids中的`<|endoftext|>`为256。
- reload_verified=true、round_trip_verified=true。
- 输入/输出路径、SHA-256、输入字节数与training_seconds。

公式是实际词表大小=256个初始字节+特殊token数量+实际merge次数。
CLI保存后核对词表、merge顺序和特殊token完全一致；再检查中文/emoji编解码。
词表的bytes按hex保存，不能逐token UTF-8解码后保存，因为一个token可能只包含半个字符。

输出已存在时拒绝覆盖，换一个新文件名；不要删除旧2048 tokenizer或原checkpoint。
已有目录会复用，缺失的父目录会创建。默认特殊token为EOS；重复传`--special-token`可替换该列表，
特殊token的ID仍按声明顺序从256开始。`--check-text`可指定另一个往返检查文本。
语料不足时报告实际词表并给出警告，不填充token；后续模型必须匹配实际词表大小。

JSON文件仍是兼容原pilot的vocab/merges/special_tokens格式，来源哈希和耗时打印到终端。
本步没有自动写正式数据清单或训练配置；training_seconds仅计训练函数，不含hash/保存/重载。
算法仍整读文件并逐轮扫描，本次耗时不是正式16K的资源基准。

## 3. 自己加载并检查

```bash
python - <<'PY'
from minigpt_cn.tokenizer import BPETokenizer

tokenizer = BPETokenizer.load('data/processed/tokenizer_m2_user300.json')
text = '数学是一门学科。😀\n'
ids = tokenizer.encode(text)
assert tokenizer.decode(ids) == text
assert tokenizer.encode('<|endoftext|>') == [256]
print('词表大小：', tokenizer.vocab_size)
print('Token IDs：', ids)
print('PASS：保存重载与中文编解码正确')
PY
```

如果前面换了输出名，这里同步更改。这个300词表不能替换旧pilot的2048词表，
也不能用于正式16K模型；本轮不修改模型配置和已有checkpoint。

## 4. 停在本模块验收点

贴测试摘要与CLI报告中的词表/重载结果即可。本模块复核后，再讨论数据清洗与划分，不一次接完。
在项目PowerShell由用户执行：

```powershell
git add README.md docs/experiment-log.md docs/m2-tokenizer-guide.md src/minigpt_cn/tokenizer tests/test_train_bpe.py tests/test_train_tokenizer_cli.py
git --no-pager diff --cached --stat
git commit -m "feat: add BPE training and tokenizer CLI"
```

add暂存本模块源码/测试/文档，diff核对全部暂存项，commit保存本地快照。
第一小步若尚未commit，也会包含在此次提交里；助手不清空暂存区、不执行Git写操作。
不提交data/processed产物，不使用git add -f，不暂存根目录的无关文件。

本模块复核通过后，下一小步见 [单篇文档清洗教程](m2-cleaning-guide.md)。
