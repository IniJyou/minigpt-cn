# 16K tokenizer操作与验收

2026-10-13已保存验收：实际16000词表、15743 merges、EOS ID256，训练122.11秒。
产物data/processed/pilot_tokenizer_16k_v1.json，SHA-256
21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470；
同目录有.report.json与.acceptance.json。已有文件先重载检查，不再重训到同一路径。
同版本三集合编码也已验收，见 [Token编码验收](token-encoding-guide.md)。下一任务只以
[current-work.md](current-work.md)为准，以下保留复现接口。

仅在用户开始本任务后执行；文档收尾轮不启动BPE训练。
用已有pilot train验证项目闭环，不再跑300词表教程或“训练但不保存”的探测。
目标16000，含256个初始字节与EOS；旧2048词表/权重不改。

## 1. 核对现有输入（WSL）

```bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
python - <<'PY'
import json
from pathlib import Path
from minigpt_cn.data.prepare import _sha256
p = Path('data/processed/m2_user_bpe_16k_probe_text/export_manifest.json')
m = json.loads(p.read_text(encoding='utf-8'))
assert m['kind'] == 'bpe_training_text' and m['format_version'] == 1
assert m['source_split']['split'] == 'train' and m['output']['path'] == 'train.txt'
assert m['boundary']['suffix'] == '<|endoftext|>\n'
t = p.parent / 'train.txt'
assert t.stat().st_size == m['output']['bytes'] == 17677160
assert _sha256(t) == m['output']['sha256'] == '0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c'
for info in (m['source_manifest'], m['source_split']):
    assert _sha256(Path(info['path'])) == info['sha256']
print('PASS：已有pilot train、来源清单与摘要一致')
PY
```

输入缺失/不匹配就停止并说明具体文件，不自行下载或解析整站。
这里核对的是已验收pilot输入的一致性，不证明其满足正式预训练数据规模。

## 2. 训练并保存（一次完整命令）

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.tokenizer.train --input data/processed/m2_user_bpe_16k_probe_text/train.txt --vocab-size 16000 --output data/processed/pilot_tokenizer_16k_v1.json
```

输入只读，输出独占创建；若已存在，先检查是否已有可验收产物，不重训/覆盖。
保存终端报告中的输入/输出SHA-256、实际词表、merges和耗时；产物不提交Git。
该CPU实现整读文本，旧同输入探测约148秒/3.14GiB峰值RSS，不承诺其他环境相同。
若实际词表不足16000不填充，明确报告验收未满足，不能把小词表配到16K模型。

## 3. 保存重载与验收

```bash
python - <<'PY'
from minigpt_cn.tokenizer import BPETokenizer
t = BPETokenizer.load('data/processed/pilot_tokenizer_16k_v1.json')
assert t.vocab_size == 16000 and len(t.merges) == 15743
assert t.encode('<|endoftext|>') == [256]
for text in ('数学是一门学科。😀\nEnglish 123', '繁體中文', 'é e\u0301', '', '甲<|endoftext|>乙'):
    ids = t.encode(text)
    assert all(0 <= i < 16000 for i in ids) and t.decode(ids) == text
print('PASS：16000词表、EOS、保存重载、中文/emoji往返正确')
PY
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_train_bpe.py tests/test_train_bpe_incremental.py tests/test_train_tokenizer_cli.py tests/test_tokenizer.py -q --tb=short
python -m ruff check .
```

验收产物：tokenizer JSON、输入/输出摘要、实际词表/merge/EOS及重载/往返结果。
这轮只完成tokenizer，不同时实现编码或启动33M训练；验收后再接下一项。
模型须绑定实际词表与该tokenizer版本，旧2048 checkpoint不可替换tokenizer续训。
往返检查不保证模型随机采样的任意字节组合都是合法UTF-8。
