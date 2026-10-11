# M2 第三小步：单篇文档清洗与过滤

这一轮只处理一个正文字符串；不下载数据、不读写 JSONL、不做去重/划分/编码。

## 1. 代码在哪里、与作业是什么关系

- `src/minigpt_cn/data/cleaning.py`：保留作业 `prepare_pilot.py::normalize_text` 的处理顺序和中文注释；新增可选转换开关与长度过滤。
- `src/minigpt_cn/data/__init__.py`：公开导入 `normalize_text`、`clean_document`。
- `tests/test_cleaning.py`：规范化、繁转简、格式保留、长度边界和错误输入。

没有搬入原脚本的顶层 mkdir、文件路径或 main，所以调用函数不会重写任何语料。
使用相同的官方 OpenCC 1.4.2 和 t2s.json 字典；不替换成另一个同名模块实现。

## 2. 激活项目环境、检查依赖与测试

在 WSL 执行，仍使用项目环境，不用作业环境：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
/home/luoyu/.local/bin/uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" "OpenCC==1.4.2"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/test_cleaning.py -q
python -m ruff check .
```

预期 26 passed、Ruff 通过。依赖已由助手装入项目环境，重复安装会复用，不重建环境或更换 PyTorch。
未来全新安装项目时，pyproject.toml 已声明此依赖，CI 安装项目也会自动安装。

## 3. 手动看一篇文档如何变化

```bash
python - <<'PY'
from minigpt_cn.data import clean_document, normalize_text

raw = '  臺灣的數學\r\n\r\n  cafe\u0301  😀  '
cleaned = normalize_text(raw)
print('规范化后：', repr(cleaned))
assert cleaned == '台湾的数学\n\ncafé  😀'
assert normalize_text(cleaned) == cleaned  # 重复清洗不继续改变这份正文。
assert normalize_text('臺灣', to_simplified=False) == '臺灣'
assert clean_document(' \t\r\n') is None
assert clean_document('中', min_chars=2) is None
assert clean_document('数学百科', max_chars=3) is None  # 过滤，不截断。
assert clean_document('中\n😀', min_chars=3, max_chars=3) == '中\n😀'
print('PASS：繁转简、换行、空文档与长度边界正确')
PY
```

复习：NFC 合并 `e + 重音符` 为 `é`，不使用会改变兼容形式的 NFKC。
只清理行首尾空白，保留正文中的空格和段落；不删除英文、标点或 emoji。
长度条件为 `min_chars ≤ len(规范化正文) ≤ max_chars`，没有上限时不检查右边。
这里的长度是 Unicode 码点数，不是 UTF-8 字节数、视觉字符数或模型 token 数。
默认 min_chars=1、max_chars=None，仍只丢弃空文档；正式阈值以后决定。

## 4. 停在这个验收点

复核上述结果即可，本轮不运行全量清洗，也不改旧 pilot 文件。
确认后由用户在项目 PowerShell 保存本模块：

```powershell
git add pyproject.toml src/minigpt_cn/data/cleaning.py src/minigpt_cn/data/__init__.py tests/test_cleaning.py data/README.md README.md docs/m2-cleaning-guide.md docs/m2-tokenizer-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git commit -m "feat: add document normalization and length filtering"
```

add 只暂存本轮源码/依赖/测试/说明；diff 核对整个暂存区；commit 保存本地快照。
不提交语料、缓存或根目录无关文件。下一小步再接文档级精确去重和固定 hash 划分。
