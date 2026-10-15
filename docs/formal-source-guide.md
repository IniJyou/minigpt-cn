# M3正式来源固定版本与有限样本验收

本轮唯一产物是来源版本/许可证据及有限样本验收包；下一任务以[current-work.md](current-work.md)为准。
不在本轮清洗正式三集合、编码token或启动训练。50元总预算包含实例保留时间，用户继续保留AutoDL实例。

## 1. 已固定的来源与本轮范围

- 数据集：wikimedia/wikipedia，中文配置20231101.zh。
- 仓库revision：b04c8d1ceb2f5cd4588862100d08de323dccfbaa。
- 元数据配置：configs/wikipedia_20231101_zh_source.json，六个中文分片大小与SHA-256均来自固定revision的官方API。
- 本轮只获取train-00000-of-00006.parquet，587109027字节（约560MiB），不下载另外五个分片。
- 该文件SHA-256：853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56。
- 原始数据集声明cc-by-sa-3.0/gfdl；许可版本原样记录，具体来源/例外及有限样本仍需审阅。

出处：[固定数据集卡](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)、
[中文分片列表](https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.zh)。
上游train是发布集合名，不是项目正式train划分。上游全中文1384748篇也不是本项目已验收规模。
许可依据和样本检查标准在包内license_notes.md；不把当前Wikipedia一般文本4.0声明套用到数据集卡的3.0声明。

## 2. 上传检查包

本地文件：logs/m3_source_review_20261013_v1/minigpt-cn-m3-source-review-v1.zip，7690字节。
包SHA-256：2c46d088b083a196c52633b2bc59aa9d8c0654481809cfbbfd5aad13cbbad9fc。
同目录delivery_manifest.json记有包内逐文件摘要。
包仅含source.json、inspect_source.py、license_notes.md、CODE_LICENSE和transfer_manifest.json。
已完成ZIP读回、语法/Ruff和缺文件/错来源/已有输出拒绝检查；实际Parquet读取尚未在本地执行，缺PyArrow且未下载分片。

用户将ZIP上传至云端/root/autodl-tmp后，在Terminal执行：

```bash
bash -eu <<'SH'
cd /root/autodl-tmp
printf '%s  %s\n' \
  '2c46d088b083a196c52633b2bc59aa9d8c0654481809cfbbfd5aad13cbbad9fc' \
  'minigpt-cn-m3-source-review-v1.zip' | sha256sum -c -
python - <<'PY'
from pathlib import Path
import hashlib
import json
import zipfile

root = Path('m3_source_review_v1')
if root.exists() or root.is_symlink():
    raise FileExistsError('已有来源检查目录，先检查，不覆盖')
with zipfile.ZipFile('minigpt-cn-m3-source-review-v1.zip') as archive:
    expected = {
        'm3_source_review_v1/source.json',
        'm3_source_review_v1/inspect_source.py',
        'm3_source_review_v1/license_notes.md',
        'm3_source_review_v1/CODE_LICENSE',
        'm3_source_review_v1/transfer_manifest.json',
    }
    if set(archive.namelist()) != expected or archive.testzip() is not None:
        raise ValueError('检查包内容不一致')
    root.mkdir(exist_ok=False)
    for name in expected:
        with (root / Path(name).name).open('xb') as handle:
            handle.write(archive.read(name))
manifest = json.loads((root / 'transfer_manifest.json').read_text(encoding='utf-8'))
for name, info in manifest['files'].items():
    data = (root / name).read_bytes()
    if len(data) != info['bytes'] or hashlib.sha256(data).hexdigest() != info['sha256']:
        raise ValueError(f'上传包文件变化：{name}')
print('上传检查包 PASS')
PY
SH
```

新目录独立于旧500步项目，旧transfer manifest和checkpoint不改。

## 3. 用户获取唯一分片

先核对控制台累计费用仍在50元总预算内，实际传输时间也计入费用；不沿用3.09元作为实时账单。
下载由用户执行，下面固定URL和输出名。已有最终文件或.part先检查，不覆盖或自动重下。

```bash
bash -eu <<'SH'
cd /root/autodl-tmp/m3_source_review_v1
mkdir -p raw
MINIGPT_SOURCE_FILE=raw/train-00000-of-00006.parquet
if [ -e "$MINIGPT_SOURCE_FILE" ] || [ -L "$MINIGPT_SOURCE_FILE" ] || \
   [ -e "$MINIGPT_SOURCE_FILE.part" ] || [ -L "$MINIGPT_SOURCE_FILE.part" ]; then
  echo '已有分片或部分下载，先核对，不覆盖、不盲目重下'
  exit 1
fi
curl --fail --location --retry 2 --connect-timeout 20 \
  --speed-time 120 --speed-limit 1024 \
  --output "$MINIGPT_SOURCE_FILE.part" \
  'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.zh/train-00000-of-00006.parquet'
printf '%s  %s\n' \
  '853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56' \
  "$MINIGPT_SOURCE_FILE.part" | sha256sum -c -
test "$(stat -c %s "$MINIGPT_SOURCE_FILE.part")" -eq 587109027
mv --no-clobber -- "$MINIGPT_SOURCE_FILE.part" "$MINIGPT_SOURCE_FILE"
echo '固定分片下载与摘要核对 PASS'
SH
```

不设曾中断正常下载的180秒总上限；有进度增长时继续观察。
中断后保留.part，未通过完整摘要不当成功文件。下载失败时回传具体curl错误及已有文件大小，
再根据实际阻塞指导恢复；不自动切镜像或把任意同名Parquet当同版本。

## 4. 用户安装读取依赖并检查有限样本

云端沿用500步的项目.venv，复用已有Torch；仅安装读取Parquet的PyArrow26.0.0。
已核对PyPI有Python3.12/Linux glibc2.28+ x86_64轮子，云端Ubuntu22.04/glibc2.35匹配。
出处：[PyArrow发行文件](https://pypi.org/project/pyarrow/26.0.0/#files)、
[Apache Arrow安装说明](https://arrow.apache.org/docs/python/install.html)。

```bash
bash -eu <<'SH'
source /root/autodl-tmp/minigpt-cn-autodl-500-v2/.venv/bin/activate
python -m pip install --disable-pip-version-check \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 30 --retries 1 --only-binary=:all: 'pyarrow==26.0.0'
cd /root/autodl-tmp/m3_source_review_v1
python inspect_source.py
SH
```

脚本先复核分片大小与SHA-256，检查实际四个字符串字段，再从首/中/末不同row group各取首8行。
最多24篇，若row group不足则更少；不完整扫描全分片正文，也不改变原始正文。
会保留实际分片总行数、文章id/url/title、正文摘要/长度、首1800字及长文末600字供审阅。
检查后再次核对输入摘要；已有review目录先检查，不覆盖。

## 5. 返回本轮验收材料

云端review目录输出source_acceptance.json、samples.jsonl、sample_review.md。
用户下载这三个小文件至PC的logs/m3_source_review_v1，并贴回source_acceptance.json。
无需把587MB分片再下载到PC，也无需改变旧checkpoint目录权限。
助手读取有限样本并核对来源/模式/摘要及许可证据后记录本轮结论，再进入正式三集合准备。

PASS仅表示身份/模式/有限样本可读取。许可与语义审阅待完成，ready_for_training固定为false，
这些摘录不能充当正式训练正文，不把有限样本的观察冒称整个分片或1亿tokens已验收。
本轮不新增src数据平台或放宽旧quarantine/审批门槛；不提交数据、日志或检查包。

## 6. 本轮实际验收结论

用户经AutoDL学术加速及断点续传完成固定分片下载，SHA-256通过；PyArrow26.0.0实际读取230792行、231个row group。
三份原始报告实际下载到logs/m3_source_review_20261013_v1；无需为了路径一致移动或覆盖。
本轮15项本地核对通过：原始报告与贴文一致，样本输出字节数/摘要正确，Markdown可从JSONL精确重建。
24篇首尾摘录已审阅，结果记录在同目录source_review_acceptance.json和source_review_conclusions.md。

质量发现：11篇有空括号、9篇有空学名、1篇有未渲染语言转换标记；还看到专辑/人员名称和行政列表缺项。
这些计数重叠，不代表全分片质量比例。数学（id13）有全文标记提示，具体位置未在所给摘录中出现；暂不接纳该篇。
来源身份与有限审阅任务已完成；该分片保留为候选，整批直接训练条件未通过，ready_for_training仍为false。
后续按预算筛选完整正文，损坏记录保守排除，不猜词、删括号后假称修好，也不重建全站渲染/审批系统。

本次网络问题的恢复依据：[AutoDL学术加速](https://www.autodl.com/docs/network_turbo/)、
[curl续传说明](https://curl.se/docs/manpage.html)。Connection refused时先启用source /etc/network_turbo并检查最终响应；
curl错误18时保留.part，从实际大小续传（--continue-at -），完成后仍核对原固定大小和SHA-256。
