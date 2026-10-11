# M2 第六小步：仅 train 的 BPE 训练纯文本导出

后续第十小步新增可选哈希采样，见 [哈希采样教程](m2-hash-sampling-guide.md)。
本页的默认源文件前缀、复核数字和命令保持不变，新模式需显式启用。

第五小步的清单已验收。本轮只准备文本，不执行BPE训练、不写token二进制、不改旧tokenizer或模型。

## 1. 文件与边界规则

- `src/minigpt_cn/data/export_text.py`：清单/哈希/归属检查、完整文档前缀导出、来源记录与CLI；关键规则有中文注释。
- `tests/test_export_train_text.py`：只读train、预算边界、EOS边界、计数/版本校验、防覆盖与错误输入。

已有清洗、划分和BPE算法没有修改。只导出正文，沿用原作业train_pilot_tokenizer.py的
`piece = text + '<|endoftext|>\n'`，不再清洗；不是JSON对象，也不含标题/URL。
EOS隔开文档，BPE必须将它声明为特殊token，不能把它纳入pair统计或跨边界合并。
正文已有EOS字面量会报错，防止混淆真实正文与追加边界；不会悄悄删文档。

## 2. 激活环境并运行测试

在WSL项目环境执行，本轮无新依赖：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_export_train_text.py -q
python -m ruff check .
```

WSL预期26 passed、Ruff通过。Windows没有符号链接权限时为24 passed、2 skipped。
完整回归：Windows313 passed、4 skipped；WSL CPU312 passed、5 skipped。

## 3. 先导出与旧pilot相同的1MiB前缀

使用你上一小步已经验收的项目清单，只写一个新的忽略目录：

```bash
python -m minigpt_cn.data.export_text \
  --manifest data/processed/m2_user_pilot_jsonl/manifest.json \
  --output-dir data/processed/m2_user_bpe_text_1m \
  --max-bytes 1048576
```

若上一小步换了目录名，同步修改--manifest。输出目录必须不存在，重复运行请换新名，
不要删除或覆盖旧样本；下面检查脚本的output也要改为新名。
预期source_split.documents=986，output.documents=50，output.bytes=1010020。
1MiB是字节上限，不是“恰好截取1MiB”；最后一篇放不下就停止选取，不截断、不跳过去挑更短文章。
虽然只导出前缀，仍读取校验全部train，不打开valid/test正文。

## 4. 核对来源、边界与旧样本

以下读取全文的脚本仅用于本轮约1MiB样本检查：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

output = Path('data/processed/m2_user_bpe_text_1m')
report = json.loads((output / 'export_manifest.json').read_text(encoding='utf-8'))
raw = (output / report['output']['path']).read_bytes()
assert len(raw) == report['output']['bytes'] == 1010020
assert hashlib.sha256(raw).hexdigest() == report['output']['sha256']
assert report['source_split']['split'] == 'train'
assert report['source_split']['documents'] == 986
assert report['output']['documents'] == raw.count(b'<|endoftext|>') == 50
assert raw.endswith(b'<|endoftext|>\n')
assert b'\r\n' not in raw
source = Path(report['source_split']['path'])
assert hashlib.sha256(source.read_bytes()).hexdigest() == report['source_split']['sha256']
manifest = report['source_manifest']
assert hashlib.sha256(Path(manifest['path']).read_bytes()).hexdigest() == manifest['sha256']
old = Path('/mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/bpe_train.txt')
assert raw == old.read_bytes()
print('篇数：', report['output']['documents'], '；字节数：', len(raw))
print('PASS：仅train导出，来源哈希与EOS边界正确，旧pilot样本逐字节一致')
PY
```

应显示50篇、1010020字节、最后PASS。已知SHA-256：
`3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba`。
text的码点数是382621，文件字节数还包含多字节UTF-8以及每篇的EOS/LF，不能混为一谈。

## 5. 全量用法与本轮边界

不传--max-bytes就导出全train；这一轮可以只复核上述1MiB样本，无需再执行全量命令：

```bash
python -m minigpt_cn.data.export_text \
  --manifest data/processed/m2_user_pilot_jsonl/manifest.json \
  --output-dir data/processed/m2_user_bpe_text_full
```

助手已验证全量986篇、6580357正文码点、17677160字节，与逐篇拼接train完全一致。
公式：`输出字节数 = Σ正文UTF-8字节数 + 篇数 × (EOS字节数 + 1个LF)`。
前缀预算有顺序偏差，只用于复现和小规模验证；不宣称代表正式语料或足够训练16K词表。
第六小步验收时BPE仍整读文本、逐轮扫描；17.7MB导出成功不代表正式16K训练的速度/内存已验收。
第七小步已接入增量训练，见 [优化与复核教程](m2-bpe-optimization-guide.md)；正式样本预算仍需单独确认。
本轮不自动接BPE训练；未来使用该文本时必须保留默认EOS，或在自定义special-token列表里显式加入它。
失败可能留下部分train.txt，没有完整export_manifest.json不可用于BPE，换新目录重跑。

## 6. 停在本模块提交点

复核测试和PASS后，由用户在项目PowerShell保存本轮代码：

```powershell
git add src/minigpt_cn/data/export_text.py src/minigpt_cn/data/__init__.py tests/test_export_train_text.py README.md data/README.md docs/m2-data-prepare-guide.md docs/m2-train-text-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git commit -m "feat: export verified train-only text for BPE"
```

add暂存源码/测试/说明，diff检查整个暂存区，commit保存本地快照。
不要提交train.txt、导出清单等data/processed产物或根目录无关文件；助手不执行Git写操作。
本轮验收后先讨论BPE训练规模与性能，再决定正式16K流程，不直接开始正式训练。
