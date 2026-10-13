# 同版本16K token数据：编码与验收

当前下一任务以 [current-work.md](current-work.md) 为准。本页记录已完成的pilot编码，
不启动模型训练、不重新划分、不训练BPE。产物用于项目闭环验证，不代表正式预训练规模。

## 已验收的固定输入与输出

- 来源：data/processed/m2_user_pilot_jsonl/manifest.json，固定train/valid/test=986/7/7。
- tokenizer：data/processed/pilot_tokenizer_16k_v1.json，16000词表、15743 merges、EOS ID256。
- tokenizer SHA-256：21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470。
- 输出：data/processed/pilot_tokens_16k_v1/{train,valid,test}.bin、manifest.json、acceptance.json。
- 格式：无文件头、小端uint16（<u2），每篇编码正文 + <|endoftext|> + LF，保留源文件顺序。

| 集合 | 文档/EOS数 | tokens | 字节 | ID范围 |
| --- | ---: | ---: | ---: | --- |
| train | 986 | 4168196 | 8336392 | 9..15999 |
| valid | 7 | 25272 | 50544 | 9..15997 |
| test | 7 | 42263 | 84526 | 10..15999 |

2026-10-13实际WSL编码耗时51.57秒，含写出/读回和结束时输入复核，不含初始输入检查。
全部1000篇已在Windows下用项目read_tokens读取，逐篇解码与原JSONL正文/EOS/LF完全一致。
train解码拼接摘要等于BPE训练输入摘要。输入/输出SHA-256、词表版本、计数、耗时和测试结果
保存在输出manifest.json与acceptance.json；数据目录Git忽略，不提交。

## 编码入口（复现时使用新目录）

现有v1已经通过验收，先执行下面的只读检查，不再重编码到同一路径。
需要复现时明确选新的output-dir；以下命令展示本轮实际参数，已有目录会立即拒绝。

```bash
cd /mnt/d/aiworkspace/transformer
source /home/luoyu/.venvs/minigpt-cn/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.data.encode \
  --manifest data/processed/m2_user_pilot_jsonl/manifest.json \
  --tokenizer data/processed/pilot_tokenizer_16k_v1.json \
  --tokenizer-sha256 21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470 \
  --vocab-size 16000 \
  --output-dir data/processed/pilot_tokens_16k_v1
```

入口先核对全部三集合文件/来源摘要及tokenizer摘要，再逐篇确认原划分、ID范围和正文往返。
逐篇写入uint16前检查范围，避免溢出回绕；落盘后核对字节摘要和EOS/LF边界。
成功manifest最后独占创建；失败保留部分产物，缺少成功manifest的目录不可用于训练。
不重新清洗、不把valid/test用于BPE训练、不扩展冻结的Wikimedia工具或审批门槛。
本入口只支持已有pilot正文哈希划分；未来作品语料另行验收，不从候选直接接入。

## 独立只读验收（WSL/Windows同一Python代码）

在项目根目录运行。下面只读现有文件，不重编码、不改写验收记录。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path
import numpy as np
from minigpt_cn.data.prepare import _sha256
from minigpt_cn.data.splitting import choose_split
from minigpt_cn.data.tokens import read_tokens
from minigpt_cn.tokenizer import BPETokenizer

out = Path('data/processed/pilot_tokens_16k_v1')
m = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
a = json.loads((out / 'acceptance.json').read_text(encoding='utf-8'))
src = Path('data/processed/m2_user_pilot_jsonl/manifest.json')
s = json.loads(src.read_text(encoding='utf-8'))
t_path = Path('data/processed/pilot_tokenizer_16k_v1.json')
t = BPETokenizer.load(t_path)
assert m['kind'] == 'encoded_pilot_tokens' and m['status'] == 'completed'
assert a['acceptance'] == 'PASS' and _sha256(out / 'manifest.json') == a['manifest_sha256']
assert _sha256(src) == m['source_manifest']['sha256'] == a['source_manifest_sha256']
assert _sha256(t_path) == m['tokenizer']['sha256'] == a['tokenizer_sha256']
assert t.vocab_size == 16000 and len(t.merges) == 15743
assert t.special_tokens == ['<|endoftext|>'] and t.encode('<|endoftext|>') == [256]
assert m['storage'] == {'dtype': '<u2', 'byte_order': 'little', 'header': False}
train_digest = hashlib.sha256()
for split, count in (('train', 986), ('valid', 7), ('test', 7)):
    source = src.parent / f'{split}.jsonl'
    info = m['files'][split]
    target = out / f'{split}.bin'
    assert _sha256(source) == s['files'][split]['sha256'] == m['source_splits'][split]['sha256']
    assert _sha256(target) == info['sha256'] == a['files'][split]['sha256']
    ids = read_tokens(target, 16000)
    assert len(ids) == info['tokens'] and target.stat().st_size == info['bytes'] == len(ids) * 2
    ends = np.flatnonzero(ids == 256)
    assert len(ends) == info['documents'] == info['eos_count'] == count
    assert int(ends[-1]) == len(ids) - 2 and np.all(ids[ends + 1] == 10)
    start = n = chars = 0
    with source.open(encoding='utf-8') as f:
        for line in f:
            text = json.loads(line)['text']
            assert choose_split(text, seed=s['processing']['seed']) == split
            stop = int(ends[n]) + 2
            piece = text + '<|endoftext|>\n'
            assert t.decode(ids[start:stop]) == piece
            if split == 'train':
                train_digest.update(piece.encode('utf-8'))
            start = stop
            n += 1
            chars += len(text)
    assert start == len(ids) and n == count
    assert chars == info['characters'] == s['files'][split]['characters']
    del ids
assert train_digest.hexdigest() == a['tokenizer_training_input_sha256'] == (
    '0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c'
)
print('PASS：同版本16K、三集合摘要、全部1000篇磁盘正文/EOS/LF与BPE train一致')
PY
```

## 本轮代码检查

定向测试覆盖新编码入口、tokenizer、保存CLI和uint16读取器：
Windows53通过/1跳过（7.91秒），WSL54通过（4.54秒），两边Ruff通过。
Windows跳过项为已有CLI的符号链接权限测试。本轮未重跑完整CPU回归。
代码改变后可使用下列命令，不把小语料测试当成正式预训练评估：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_encode_data.py tests/test_tokenizer.py \
  tests/test_train_tokenizer_cli.py tests/test_pilot_verification.py -q --tb=short
python -m ruff check .
```

模型以后须使用同一tokenizer与这批token ID；旧2048 tokenizer/checkpoint保持独立。
本轮停在数据编码验收，33M本地短训练/恢复在用户启动下一轮后开展。
