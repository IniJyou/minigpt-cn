# M2 第四小步：正文精确去重与固定 hash 划分

第三小步的清洗已验收。本轮只增加两个内存正文函数，不接全量处理CLI、下载或token编码。

## 1. 代码在哪里

- `src/minigpt_cn/data/splitting.py`：deduplicate_texts与choose_split，关键规则有中文注释。
- `src/minigpt_cn/data/__init__.py`：公开导入两个函数。
- `tests/test_splitting.py`：固定桶边界、重复正文、清洗后去重、顺序独立、跨进程一致和非法输入。

去重逻辑保留原prepare_pilot.py的正文set语义；划分保留split_pilot.py的SHA-256公式。
没有搬入原脚本的文件路径/main，不会改写旧pilot的train/valid/test。

## 2. 激活项目环境并运行测试

在WSL运行，仍使用项目环境：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_splitting.py tests/test_cleaning.py -q
python -m ruff check .
```

预期47 passed、Ruff通过。本轮无新依赖，不需要重建或更新环境。

## 3. 手动检查：先清洗，再去重，再划分

```bash
python - <<'PY'
from minigpt_cn.data import clean_document, deduplicate_texts, choose_split

raw = ['  边界文档91\r\n', '边界文档5', '边界文档28', '边界文档5', ' \t\n']
cleaned = (clean_document(text) for text in raw)
unique = list(deduplicate_texts(text for text in cleaned if text is not None))
assert unique == ['边界文档91', '边界文档5', '边界文档28']

groups = {name: [] for name in ('train', 'valid', 'test')}
for text in unique:
    groups[choose_split(text, seed=42)].append(text)
assert groups == {
    'train': ['边界文档91'],
    'valid': ['边界文档5'],
    'test': ['边界文档28'],
}
for name, texts in groups.items():
    print(name, texts)
assert {text: choose_split(text) for text in reversed(unique)} == {
    text: choose_split(text) for text in unique
}
print('PASS：空文档过滤、重复去除、固定归属与重排一致')
PY
```

这三个样本特意覆盖三个集合，1/1/1不是一般语料的预期比例。
只按正文去重；行内空格不同等正文差异不会被当作近似重复。
本步输入/输出是正文字符串，不包含标题/URL；文件管线稍后再接入并保留来源信息。

## 4. 简短复习与限制

公式：`bucket = int(SHA256(UTF8(f"{seed}\0{text}")), 16) % 100`。
0～97→train，98→valid，99→test；种子42、NUL分隔、UTF-8和完整正文都不能随意换。
不依赖随机采样或输入序号，不使用会受PYTHONHASHSEED影响的内置hash；同文不会跨集合。
98/1/1是期望比例，有限样本未必精确满足，也不是token比例；不能为凑比例临时挪文档。
tokenizer仍只能用train训练。去重只遍历输入一次，但seen保存全部唯一正文，尚未测正式语料内存。

助手只读复现原pilot：1000篇唯一文档，train/valid/test=986/7/7；正文与顺序逐项一致，文件哈希未变。
完整回归：Windows266 passed、1 skipped；WSL CPU262 passed、5 skipped；两环境Ruff通过。

## 5. 停在本模块验收点

先复核上述测试和手工样例，再由用户在项目PowerShell保存本模块：

```powershell
git add src/minigpt_cn/data/splitting.py src/minigpt_cn/data/__init__.py tests/test_splitting.py README.md data/README.md docs/m2-cleaning-guide.md docs/m2-splitting-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git commit -m "feat: add exact text deduplication and deterministic splits"
```

add只暂存本轮源码/测试/文档；diff检查整个暂存区；commit保存本地快照。
不要暂存根目录无关文件或语料；助手不执行Git写操作。
下一小步见 [JSONL处理入口与清单教程](m2-data-prepare-guide.md)，不同时完成正式数据下载与token编码。
