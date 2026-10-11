# M2 第七小步：增量 BPE 优化与复核

本轮只优化训练算法并核对输出，不下载数据、不保存新 tokenizer、不开始正式16K或token编码。
公开 train_bpe(input_path, vocab_size, special_tokens) 和既有训练CLI、JSON文件格式不变。
作业源码、旧2048 tokenizer、数据和checkpoint不修改，无新依赖。

## 为什么官方测试1.1秒仍有优化空间

作业 tests/test_train_bpe.py 的速度用例是134042字节英文、500词表，阈值1.5秒。
通过阈值不等于达到最快速度，也不是中文大语料或16K词表的资源验收。
同一环境本次作业函数约0.97秒、项目旧版1.72秒、优化版0.12秒，词表和merge顺序完全一致。
项目旧版迁移时用了Counter/any等不同写法，并非逐字复制，速度也并非严格等于作业版。
以上是训练函数计时，不是整场pytest耗时；不能把1.1秒直接外推到正式语料。

## 实现在哪里、改了什么

- src/minigpt_cn/tokenizer/train_bpe.py：生产增量版，关键规则有中文注释。
- src/minigpt_cn/tokenizer/train_bpe_reference.py：保留旧全扫描版，只给小样本回归/对照用。
- tests/test_train_bpe_incremental.py：34项新测试；不依赖作业目录或真实语料。

初始化仍用原PAT和最长优先的特殊token隔离。词块是变长一维整数元组，长度为当前token数；
不是[B,S]张量，word_id是训练器内部的词块编号，不能混同token ID。

频次定义保持为 `C(pair) = Σ词块 权重 × 词块内相邻出现次数`。优化如下：

1. 倒排索引 pair_words：找到含本轮最佳pair的词块，其他词块不访问。
2. 缓存局部计数：x,a,b,y → x,new,y，仅删除(x,a)/(a,b)/(b,y)，加入(x,new)/(new,y)。
   全局变化为 `ΔC = Σ受影响词块 权重 × 局部变化`，不每轮重数全部pair。
3. 优先队列：按(频次,左bytes,右bytes)取最大，不按ID平局；旧频次懒删除，节点过多时重建堆。

注意：aaaa中aa有3个相邻出现，但替换是非重叠的两次。邻边计数与实际替换必须分别处理。
相邻连续合并要用已更新的左邻居；Counter减法会丢负数，不能用它算有符号频次变化。
索引只记录是否存在，局部计数降到零才移除词块。特殊token不参与任何pair或跨文档合并。
不同词块即使后来相同，也可保留固定编号和原权重分别计数，结果等于参考版的聚合计数。

## 本次实测与边界

WSL项目环境Python3.12.3、CPU；独立子进程单次测量，计时包含读文件/预分词/训练，
不含导入、SHA-256、保存和重载。耗时会随机器负载波动，不作为CI硬阈值。

| 输入与目标 | 项目旧全扫描版 | 优化版 | 优化进程峰值RSS |
| --- | ---: | ---: | ---: |
| 英文134042字节，500词表 | 1.72秒 | 0.12秒 | 约33MiB |
| 中文1010020字节，300词表 | 7.83秒 | 1.93秒 | 约172MiB |
| 同一中文，512词表 | 35秒超时未完成 | 3.75秒 | 约188MiB |
| 同一中文，2048词表 | 本轮未重跑 | 5.99秒 | 约219MiB |
| 全train17677160字节，300词表 | 35秒超时未完成 | 35.59秒 | 约2461MiB |
| 同一全train，2048词表 | 本轮未重跑 | 109.45秒 | 约2824MiB |

1MiB中文的2048结果逐ID词表及全部1791次merge与已验收旧pilot tokenizer完全一致。
全train的2048结果使用更多正文，当然不是旧pilot的词表；本轮只计时，未保存或替换它。
1MiB中文SHA-256为3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba。
全部输入、作业源码/速度测试、旧tokenizer哈希前后不变；未写新词表。
超时只是测试上限，不是完整训练耗时，也不能用于算精确加速比。

这是空间换时间：1MiB/300的旧版进程峰值约40MiB，优化版约172MiB。
仍整读文本，且缓存唯一词块、局部pair计数与索引；合并时仍扫描受影响词块构造新序列。
不宣称整体线性复杂度，不宣称无限大语料可训练，不等同于正式16K或全Wikipedia资源验收。

## 用户复核：只读样本，不保存词表

在WSL项目环境执行：

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_train_bpe.py tests/test_train_bpe_incremental.py \
  tests/test_train_tokenizer_cli.py tests/test_tokenizer.py -q
python -m ruff check .
```

WSL预期82 passed；Windows无符号链接权限时81 passed、1 skipped。
全部CPU回归：Windows342 passed、9 skipped；WSL346 passed、5 skipped。
CUDA测试在CPU模式跳过，Windows另有符号链接权限相关跳过；不是BPE测试失败。

```bash
python - <<'PY'
from pathlib import Path
from time import perf_counter
from minigpt_cn.tokenizer import BPETokenizer, train_bpe
from minigpt_cn.tokenizer.train_bpe_reference import train_bpe_reference

source = Path('data/processed/m2_user_bpe_text_1m/train.txt')
before = source.read_bytes()
results = []
for trainer in (train_bpe_reference, train_bpe):
    start = perf_counter()
    results.append(trainer(source, 300, ['<|endoftext|>']))
    print(trainer.__name__, 'seconds=', round(perf_counter() - start, 3))
assert results[0] == results[1]
vocab, merges = train_bpe(source, 2048, ['<|endoftext|>'])
old = BPETokenizer.load('/mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/tokenizer_2048.json')
assert (vocab, merges) == (old.vocab, old.merges)
assert len(vocab) == 2048 and len(merges) == 1791
assert source.read_bytes() == before
print('PASS：优化前后完全一致；旧2048 pilot词表/merge完全一致；输入未变')
PY
```

若你导出时换了目录名，同步修改source。这个样本仅用于复核，仍不能替换正式16K tokenizer。
完整回归可再执行 `CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q`。

## 停在本模块提交点

复核后由用户在项目PowerShell执行；助手不执行Git写操作：

```powershell
git add src/minigpt_cn/tokenizer/train_bpe.py src/minigpt_cn/tokenizer/train_bpe_reference.py tests/test_train_bpe_incremental.py README.md docs/experiment-log.md docs/m2-tokenizer-guide.md docs/m2-train-text-guide.md docs/m2-bpe-optimization-guide.md
git --no-pager diff --cached --stat
git commit -m "perf: train BPE with incremental counts and pair index"
```

add仅暂存本轮源码/测试/文档；diff核对整个暂存区；commit保存本地快照。
不提交数据/词表/日志，也不暂存根目录无关文件。不自动push或修改GitHub。
本轮验收后再确定正式BPE的train采样预算和16K资源测试，不同时完成下载、编码或模型训练。
