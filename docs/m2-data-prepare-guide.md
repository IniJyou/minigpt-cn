# M2 第五小步：JSONL 处理入口与清单

第四小步已验收。本轮只接文件入口与manifest，不下载、训练tokenizer、导出训练纯文本或编码token。

## 1. 代码在哪里

- `src/minigpt_cn/data/prepare.py`：参数核对、逐行读取、清洗/去重/划分、保留元数据、写处理清单及CLI。
- `tests/test_prepare_data.py`：来源保留、跨文件去重、计数与哈希、失败处理、不覆盖和独立进程入口。

原cleaning.py和splitting.py算法未修改。入口调用normalize_text/choose_split，按同一正文set规则去重，
以便保留原记录的其他字段；没有搬入原脚本固定的输出路径。

## 2. 激活项目环境并检查

在WSL执行，不使用作业环境，本轮没有新依赖：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_prepare_data.py tests/test_splitting.py tests/test_cleaning.py -q
python -m ruff check .
```

WSL预期71 passed、Ruff通过。Windows没有创建符号链接权限时，该组为70 passed、1 skipped。

## 3. 用原pilot输入生成一份新的项目输出

输入是已经提取正文的JSONL，每行有text字符串；不是Wikipedia XML dump，也不是token二进制。
原作业文件只读，产物写项目忽略目录：

```bash
python -m minigpt_cn.data.prepare \
  --input /mnt/d/aiworkspace/cs336-assignment1-basics/data/raw/zh_wikipedia_pilot.jsonl \
  --output-dir data/processed/m2_user_pilot_jsonl \
  --seed 42
```

输出目录必须不存在；若已存在，换新名称，并同步改下一段检查脚本的output路径，不删除旧数据。
预期read=saved=1000，blank_lines/empty/duplicates/length_filtered均为0，train/valid/test=986/7/7。
目录应有train.jsonl、valid.jsonl、test.jsonl、manifest.json，暂时没有train.txt或bin文件。

manifest记录输入/输出哈希、处理规则、OpenCC版本、长度阈值和数量；终端报告另包含output_dir。
标题、URL、来源、许可等字段原样保留，仅text规范化；同正文只保留第一次出现的整条记录。
多个输入可重复传--input，所有输入共用去重集合；输入顺序决定哪份来源先保留。

## 4. 检查清单和三个文件

以下小样本检查会读取文件内容，只用于本轮1000篇pilot验收：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

output = Path('data/processed/m2_user_pilot_jsonl')
original = Path('/mnt/d/aiworkspace/cs336-assignment1-basics/data/processed')
manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
source = manifest['inputs'][0]
raw = Path(source['path']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == source['sha256']
assert len(raw) == source['bytes']
assert manifest['counts']['saved'] == 1000
groups = {}
for name, info in manifest['files'].items():
    contents = (output / info['path']).read_bytes()
    assert hashlib.sha256(contents).hexdigest() == info['sha256']
    assert len(contents) == info['bytes']
    assert contents == (original / f'{name}.jsonl').read_bytes()
    records = [json.loads(line) for line in contents.decode('utf-8').splitlines()]
    assert len(records) == info['documents']
    assert sum(len(record['text']) for record in records) == info['characters']
    groups[name] = {record['text'] for record in records}
    print(name, info['documents'], '篇；哈希和原pilot一致')
assert groups['train'].isdisjoint(groups['valid'])
assert groups['train'].isdisjoint(groups['test'])
assert groups['valid'].isdisjoint(groups['test'])
print('PASS：清单、元数据/正文文件一致，无跨集合重复')
PY
```

应显示train 986、valid 7、test 7，最后PASS。同一环境、相同输入/规则再写一个新目录，产物和manifest可逐字节复现；
若改变输入顺序，文档归属仍固定，但文件内顺序以及重复正文保留的来源可能改变。

## 5. 复习与失败处理

计数关系：`read = empty + duplicates + length_filtered + saved`，blank_lines单独计数。
长度过滤在去重之后，边界包含在内；默认只丢空正文，不自行选择正式长度阈值。
`--min-chars`/`--max-chars`可修改范围，`--keep-traditional`可关闭繁转简；设置记录在manifest中。
空split直接报错，不挪文档凑比例。坏JSON、非字符串text、NaN/Infinity或坏UTF-8会明确失败。
本轮不实现目录级原子事务；中断/失败可能留下部分输出，缺少完整manifest不可视为成功，换新目录重跑。
tokenizer以后仍只使用train；这里的JSONL不能直接喂现有的纯文本BPE CLI或uint16训练入口。
seen仍保存唯一正文，正式大语料内存未验收；来源/许可仍须人工核实，不由manifest自动证明。

## 6. 停在本模块提交点

先复核CLI计数和PASS，再由用户在项目PowerShell保存代码：

```powershell
git add src/minigpt_cn/data/prepare.py src/minigpt_cn/data/__init__.py tests/test_prepare_data.py README.md data/README.md docs/m2-splitting-guide.md docs/m2-data-prepare-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git commit -m "feat: add JSONL data preparation CLI and manifest"
```

add暂存本轮源码/测试/说明；diff核对整个暂存区；commit保存本地快照。
不提交data/processed产物或根目录无关文件，助手不执行Git写操作。
本轮验收后下一小步见 [仅train的BPE训练文本导出](m2-train-text-guide.md)，不同时训练正式16K词表或编码token。
