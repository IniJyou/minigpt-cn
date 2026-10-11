# M2 第八小步：16K BPE 资源探测

本轮只用已验收的pilot train文本做可行性检查，不保存新tokenizer，不下载正式语料，
不写uint16、不训练模型，不修改旧词表/权重/作业源码。BPE算法和模型配置没有修改。
正式目标是16000，不是16384；词表包含256个初始字节、1个EOS和15743次merge。

## 输入与资源边界

- 输入：986篇train，17677160字节，与此前完整文档拼接结果相同，不使用valid/test训练BPE。
- 输入SHA-256：0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
- WSL项目Python3.12.3、CPU；本次算法commit为319a8c1。
- 训练限时300秒；Linux RLIMIT_AS设为6GiB，限制的是该进程虚拟地址空间，不是GPU显存。
  实际峰值常驻内存用ru_maxrss测量，两种指标不能混同。到上限报错，不视为训练成功。
- 单次探测不是统计平均，不承诺正式大语料同样快，也不按文件字节数线性外推内存。

本次实测通过：训练148.1185秒、峰值RSS3210.609MiB（约3.14GiB），
actual_vocab=16000、merges=15743、EOS ID=256；全部token bytes唯一、ID连续、
每个新词条等于对应merge的左右字节拼接，6组往返样例通过，全部受检文件哈希未变。
助手只在独立进程内加少量阶段日志，未改磁盘源码；计时不含哈希/构建编码器/往返检查。
本次WSL总内存约15.27GiB，探测前可用约14.27GiB；未使用GPU。

## 1. 激活环境并检查已有组件

以下全部在WSL项目环境运行：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_train_bpe.py tests/test_train_bpe_incremental.py \
  tests/test_train_tokenizer_cli.py tests/test_tokenizer.py -q
python -m ruff check .
```

聚焦测试预期82 passed。没有新依赖，也不需要重新安装作业环境。

## 2. 为你的复核导出独立全train文本

从已验收处理清单导出新目录，不传max-bytes，不覆盖任何旧样本：

```bash
python -m minigpt_cn.data.export_text \
  --manifest data/processed/m2_user_pilot_jsonl/manifest.json \
  --output-dir data/processed/m2_user_bpe_16k_probe_text
```

应为986篇、17677160字节。目录若已存在，请换新名，并同步修改下方manifest路径。
助手探测用的是已存在的data/processed/m2_verification/bpe_text_full；两者源自同一处理清单，
正文/顺序/边界相同。下面脚本先验证输入来源和哈希，未通过就不开始BPE。

## 3. 只读训练与完整性检查，不保存词表

此脚本使用Linux的resource/signal，仅在WSL运行。它训练临时内存中的词表，退出即释放；
没有tokenizer.save调用，不能把本次探测当成已交付的正式tokenizer。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python - <<'PY'
import hashlib
import json
import resource
import signal
from pathlib import Path
from time import perf_counter
from minigpt_cn.config import load_model_config
from minigpt_cn.tokenizer import BPETokenizer, train_bpe

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

manifest = Path('data/processed/m2_user_bpe_16k_probe_text/export_manifest.json')
report = json.loads(manifest.read_text(encoding='utf-8'))
assert report['format_version'] == 1 and report['kind'] == 'bpe_training_text'
assert report['source_split']['split'] == 'train'
assert report['output']['path'] == 'train.txt' and report['selection']['max_bytes'] is None
assert report['boundary'] == {'special_token': '<|endoftext|>', 'suffix': '<|endoftext|>\n', 'encoding': 'utf-8'}
source = manifest.parent / 'train.txt'
processing = Path(report['source_manifest']['path'])
train_jsonl = Path(report['source_split']['path'])
watched = [manifest, source, processing, train_jsonl, Path('configs/minigpt_33m.yaml'), Path('src/minigpt_cn/tokenizer/train_bpe.py'), Path('src/minigpt_cn/tokenizer/bpe.py')]
old = Path('/mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/tokenizer_2048.json')
if old.exists():
    watched.append(old)
before = {str(p): sha(p) for p in watched}
assert before[str(source)] == report['output']['sha256'] == '0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c'
assert source.stat().st_size == report['output']['bytes'] == 17677160
assert report['output']['documents'] == report['source_split']['documents'] == 986
assert before[str(processing)] == report['source_manifest']['sha256']
assert before[str(train_jsonl)] == report['source_split']['sha256']
target = load_model_config('configs/minigpt_33m.yaml').vocab_size
assert target == 16000

def deadline(signum, frame):
    raise TimeoutError('探测超过300秒，未视为成功')

resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
signal.signal(signal.SIGALRM, deadline)
signal.setitimer(signal.ITIMER_REAL, 300)
started = perf_counter()
print('开始16K资源探测；限时300秒；不保存词表', flush=True)
try:
    vocab, merges = train_bpe(source, target, ['<|endoftext|>'])
    training_seconds = perf_counter() - started
    # 256初始字节 + 1 EOS + 15743 merges = 16000，不能用填充token凑数。
    assert len(vocab) == target and len(merges) == target - 257
    assert set(vocab) == set(range(target)) and len(set(vocab.values())) == target
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == b'<|endoftext|>'
    assert all(vocab[257+i] == a+b for i, (a,b) in enumerate(merges))
    assert all(b'<|endoftext|>' not in token for i, token in vocab.items() if i != 256)
    tokenizer = BPETokenizer(vocab, merges, ['<|endoftext|>'])
    samples = ['数学是一门学科。😀\nEnglish 123', '繁體中文與科学🦉👍🏽', 'é e\u0301\t\n', '', '<|endoftext|>', '甲<|endoftext|>乙<|endoftext|>']
    for text in samples:
        ids = tokenizer.encode(text)  # 变长一维ID列表[sequence]，不是[B,S]张量。
        assert all(0 <= i < target for i in ids)
        assert tokenizer.decode(ids) == text
    assert tokenizer.encode('<|endoftext|>') == [256]
    assert all(sha(p) == before[str(p)] for p in watched)
    print(json.dumps({'status': 'completed', 'actual_vocab': len(vocab), 'merges': len(merges), 'training_seconds': training_seconds, 'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 'round_trip_verified': True, 'eos_id': 256, 'watched_files_unchanged': True, 'tokenizer_saved': False}))
    print('PASS：16K词表/合并数、中文往返与EOS正确；输入未变；未保存词表')
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
PY
```

应看到actual_vocab=16000、merges=15743、eos_id=256和最后PASS。
如果出现超时、MemoryError或assert报错，没有PASS就不算验收通过；输入不会被覆盖。
往返检查验证的是编码产生的IDs，不证明模型随机生成的字节组合总是合法UTF-8。
旧pilot checkpoint使用2048词表；16K词表不能直接配旧权重，其embedding/输出矩阵和ID语义均不匹配。

## 本轮验收边界

完整实测见docs/experiment-log.md对应第八小步。本轮只验证这份pilot train和当前CPU实现。
BPE仍整读文本、保存唯一词块和索引；正式数据/采样预算、来源许可与16K保存交付尚未完成。
不把词表大小正确等同于词表质量，不把本次资源数字用作33M模型训练吞吐量或显存估算。
复核后再确定正式BPE的样本预算和数据路线，uint16编码入口放到后续独立小步。

## 4. 复核后仅提交文档

本轮没有源码或测试改动；由用户在项目PowerShell保存探测记录：

```powershell
git add README.md docs/experiment-log.md docs/m2-bpe-optimization-guide.md docs/m2-bpe-16k-probe-guide.md
git --no-pager diff --cached --stat
git commit -m "docs: record bounded 16K BPE resource probe"
```

add仅暂存四份说明；diff检查整个暂存区；commit保存本地快照。助手未执行Git写操作。
不要暂存data/processed文本/清单、旧词表、日志或根目录无关文件，也不自动push。
