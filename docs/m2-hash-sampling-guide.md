# M2 第十小步：固定哈希、完整文档、仅 train 的 BPE 采样

本轮只实现采样并用旧 pilot 验证；不下载正式 dump、不训练/保存 tokenizer、
不编码 uint16、不训练模型，不修改旧数据/权重或学习笔记。没有新依赖。

## 1. 改在哪里，怎么理解

- `src/minigpt_cn/data/sampling.py`：采样哈希、候选最大堆和正常顺序输出；有中文注释。
- `src/minigpt_cn/data/export_text.py`：新增 `--selection hash` / `--sample-seed`，接回已有清单校验。
- `tests/test_hash_sample_text.py`：43项新测试；原前缀导出的26项测试不改。

默认 selection=prefix 保留原源文件顺序及原清单字段。hash 必须显式指定正整数
max_bytes，采样 seed 默认42；它不改变处理清单中的 train/valid/test 划分种子。
prefix 不接受 sample_seed，避免参数看似生效、实际被忽略。

对正文 t，排序键为：

```text
(SHA256(UTF8("bpe-sample-v1" + NUL + seed + NUL + t)), UTF8(t))
```

seed 按十进制整数写入；NUL 是真实的零字节，不是反斜杠和数字0。
哈希与划分使用不同域，平局按正文字节排序；行号/标题/来源不参与选择。
这里处理的是变长文档字节串，不是模型的 [batch, sequence] token 张量。

预算公式：`B >= Σ(正文UTF-8字节数 + EOS字节数 + 1个LF)`。
按哈希顺序取完整文档前缀，遇到装不下的第一篇即停止，不截断、不跳选。
因此实际字节数可能略小于16MiB；首篇装不下则报错，没有“空样本成功”。

堆只保存累计字节首次达到预算的最短哈希前缀，末篇可能越界，写出时排除。
如果删掉堆中排序最靠后的文档后仍覆盖预算，它永远不会入选，可以丢弃；
后来增加更靠前的文档只会更早耗尽预算。测试每加入一篇都与全排序对照。
候选字节数小于预算加末篇大小，但还有 Python 对象/解析开销，不是硬 RSS 上限。
仍扫描、校验整份 train；单篇巨大文档和上游 prepare 的 seen 开销没有因此解决。

hash 清单增加 algorithm_version=1、sample_seed、hash_rule 和 output.sources。
来源支持 source 字符串或 source.name 字符串；缺失/其他结构记为 null，不猜测许可。
来源统计的 bytes 包含 EOS/LF，各来源统计之和必须等于输出总数，不强制来源配比。
哈希采样减少源文件顺序偏置，不保证来源/主题均衡，更不证明词表质量或数据许可。

## 2. 在 WSL 项目环境运行测试

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_export_train_text.py tests/test_hash_sample_text.py -q
python -m ruff check .
```

WSL聚焦预期69 passed；Windows无符号链接权限时67 passed、2 skipped。
助手完整CPU回归：Windows385 passed、9 skipped；WSL389 passed、5 skipped。
跳过包括CUDA测试和Windows符号链接权限测试；本轮没有声称跑过GPU训练。

## 3. 从已验收 pilot train 导出新的16MiB哈希样本

```bash
python -m minigpt_cn.data.export_text \
  --manifest data/processed/m2_user_pilot_jsonl/manifest.json \
  --output-dir data/processed/m2_user_bpe_hash16m \
  --selection hash --sample-seed 42 --max-bytes 16777216
```

输出目录必须不存在；复跑时换新名，并同步修改下方 output。不删除/覆盖旧样本。
预期 source_split.documents=986，output.documents=942，output.bytes=16742929，
output.characters=6244484。source=null 是这份 pilot 缺少可识别 source 字段，不是丢了正文。
这是采样行为的验收样本，不是正式双来源训练语料。

## 4. 独立排序复核

下面全读 train 的 oracle **仅用于这份约17.7MB的pilot**；生产采样器使用堆，
不要用这个检查脚本直接处理正式全量语料。

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

output = Path('data/processed/m2_user_bpe_hash16m')
report = json.loads((output / 'export_manifest.json').read_text(encoding='utf-8'))
source = Path(report['source_split']['path'])
processing = Path(report['source_manifest']['path'])
assert sha(source) == report['source_split']['sha256'] == '1250226874b95e47ef9f900140a83c315e420886558902cb23c4ca61346b3699'
assert sha(processing) == report['source_manifest']['sha256']
selection = report['selection']
assert selection['strategy'] == 'whole_document_hash_prefix'
assert selection['algorithm_version'] == 1 and selection['sample_seed'] == 42
assert selection['max_bytes'] == 16777216 and report['source_split']['split'] == 'train'
records = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
assert len(records) == report['source_split']['documents'] == 986
records.sort(key=lambda r: (hashlib.sha256(f"bpe-sample-v1\0{42}\0{r['text']}".encode()).digest(), r['text'].encode()))
parts, size, chars = [], 0, 0
for record in records:
    piece = (record['text'] + '<|endoftext|>\n').encode('utf-8')
    if size + len(piece) > 16777216:
        break
    parts.append(piece)
    size += len(piece)
    chars += len(record['text'])
raw = (output / 'train.txt').read_bytes()
assert raw == b''.join(parts)
assert len(parts) == report['output']['documents'] == 942
assert size == len(raw) == report['output']['bytes'] == 16742929
assert chars == report['output']['characters'] == 6244484
assert raw.count(b'<|endoftext|>') == 942 and raw.endswith(b'<|endoftext|>\n')
assert hashlib.sha256(raw).hexdigest() == report['output']['sha256'] == '5a322b2d46ceb5081f853ac3c4fb1ba5dca0ca19c79756a87485a66c13567f1f'
for field in ('documents', 'characters', 'bytes'):
    assert sum(s[field] for s in report['output']['sources']) == report['output'][field]
assert sha(source) == report['source_split']['sha256'] and sha(processing) == report['source_manifest']['sha256']
print('PASS：完整文档哈希前缀与独立排序一致，来源/统计/边界正确，输入未变')
PY
```

助手另验收：重复导出逐字节一致；旧默认1MiB前缀仍为50篇/1010020字节，
SHA-256仍是3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba。
采样单次实测约1.571秒、该进程截至第一次导出峰值RSS约43.625MiB，
不含后续独立排序；不是BPE训练时间，也不外推正式语料性能。
输入三份split/清单、旧词表/BPE文本/pilot权重、模型配置和BPE源码的受检哈希未变。
失败无完整export_manifest.json不可用于BPE，换新目录重跑，不把部分文本算验收成功。

## 5. 停在复核与提交点

通过测试和上方PASS后，在项目PowerShell由你执行：

```powershell
git add src/minigpt_cn/data/sampling.py src/minigpt_cn/data/export_text.py tests/test_hash_sample_text.py README.md data/README.md docs/m2-formal-data-plan.md docs/m2-train-text-guide.md docs/m2-hash-sampling-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git --no-pager diff --cached
git commit -m "feat: sample bounded BPE text by deterministic document hash"
```

add暂存源码、测试和文档；两条diff检查暂存范围/内容；commit保存本地快照。
不提交data/processed样本或根目录无关文件，助手不执行Git写操作。
复核后下一小步核对正式下载资源并接入下载清单；正文提取、正式16K产物和uint16仍待验收。
