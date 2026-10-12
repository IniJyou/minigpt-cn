# M2 第十一小步：下载入口、校验与来源清单

当前进度（2026-10-12）：两份本地压缩包已独立复核，最新摘要见data/README.md；
流式XML小样本见 [解析教程](m2-xml-sample-guide.md)。以下保留下载入口与续传过程的历史验收说明。
后续任何文件下载均由用户亲自执行，助手只提供命令、代码与本地复核，不自行启动下载。

本轮只接入入口，用小型离线响应验证传输；实际联网仅查询官方状态JSON和HTTP响应头。
**本轮没有下载两个大型dump，也没有提取正文、训练tokenizer、编码uint16或训练模型。**
用户这次只执行第2～3节复核，不加 `--download`；实际下载放到下一小步。

以上为第十一小步的历史验收范围。2026-10-11实际下载时遇到响应提前结束，
现已补上显式断点续传与有限网络重试；当前操作见第6节，不再要求网络中断后换新目录。

## 1. 新增文件与行为

- `configs/wikimedia_20261001.yaml`：固定两个站点的版本、官方URL、大小、SHA-1和许可依据。
- `src/minigpt_cn/data/download.py`：预览、元数据检查、分块下载、校验和完整下载清单。
- `tests/test_download_dump.py`：50项离线测试，不访问公网；不改变已有模型/BPE/数据处理接口。

沿用现有PyYAML、tqdm和Python标准库，无新依赖。一次只处理一个source；
首次下载必须新目录，只有显式`--download --resume`才允许接着已有的.part下载。

| 选项 | 联网 | 创建输出目录/大文件 | 返回状态 |
| --- | --- | --- | --- |
| 默认，无额外选项 | 否 | 否 | planned |
| `--check-remote` | 仅状态JSON，最多2MiB | 否 | planned，附带官方状态 |
| `--download` | 先状态JSON，再压缩包 | 是 | 校验/完整清单写出后为verified |

下载前对照固定配置：对应job必须done，官方文件大小/SHA-1/URL必须完全一致。
有变化就报错，不自动追随latest或更新配置。默认TLS验证保持开启，不自动跟随HTTP重定向。
未获授权的镜像、手动换URL和全历史dump不在首版入口范围内。

Wikipedia：3430912430字节，SHA-1为04d5eec3c249fdf8604b97f4435e019c60a4ec94；
Wikisource：9565159689字节，SHA-1为b1a6e0c5ae0b55692c41010a47f7eb02e8087e2a。
2026-10-11再次核对的 [Wikipedia状态](https://dumps.wikimedia.org/zhwiki/20261001/dumpstatus.json)、
[Wikisource状态](https://dumps.wikimedia.org/zhwikisource/20261001/dumpstatus.json)均匹配配置。
压缩包约3.195/8.908GiB；HEAD响应均为200，Content-Length与固定大小一致，无额外Content-Encoding。
HEAD检查不等于已下载并校验内容；当时本地dump SHA-256未知，当前Wikipedia结果见第6节。

磁盘预检查要求：`可用字节 >= 本次压缩文件字节数 + 1GiB下载余量`。
项目D盘本次检查可用约375GiB；这个条件不保证正式正文提取/去重/编码所需空间足够，
也不预占磁盘，其他进程写入或配额仍可能造成后续失败。下载完一份再检查下一份。

## 2. 在WSL项目环境执行离线测试

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_download_dump.py tests/test_download_resume.py -q --tb=short
python -m ruff check .
```

第十一小步原入口为50项测试；新增续传测试后本模块预期91 passed，两边均无需跳过。
第十一小步历史CPU回归：Windows435 passed、9 skipped；WSL439 passed、5 skipped，
Ruff两边通过。完整回归的跳过包括CUDA和Windows符号链接权限测试。
测试用约101字节的小型bz2响应，绝不是正式中文语料效果或大型传输性能验收。

## 3. 先只预览，不下载

先离线预览Wikipedia的配置、磁盘空间和拟写目录：

```bash
python -m minigpt_cn.data.download \
  --source zhwiki --output-dir data/raw/zhwiki_20261001_v1
```

应看到status=planned、official_status=null、file=null、ready_for_training=false；不会创建目录。
接着实际查询两个小型官方状态文件，仍不创建目录、不传输压缩包：

```bash
python -m minigpt_cn.data.download \
  --source zhwiki --output-dir data/raw/zhwiki_20261001_v1 --check-remote
python -m minigpt_cn.data.download \
  --source zhwikisource --output-dir data/raw/zhwikisource_20261001_v1 --check-remote
```

两份都应保持status=planned、file=null，official_status.status=done，并给出元数据SHA-256。
此处SHA-256是状态JSON的摘要，不是dump文件摘要，不能混填。
预览不创建输出目录，所以以上同名目录可以连续预览；若目录原本就存在，换新名，不删旧数据。
网络错误会报错，不能把离线测试通过当作网络检查通过；网络恢复后可重新执行预览。
`--timeout`默认30秒，指单次阻塞网络操作，不是整个下载的总时限。
参见 [Python urllib.request](https://docs.python.org/3.12/library/urllib.request.html)。

## 4. 后续真正下载时的约定（本轮不执行）

下一小步才在确认的命令上添加`--download`。它始终先重新检查官方元数据，
然后在新目录保存配置/状态快照，按最多1MiB一块写压缩文件.part；进度条在stderr，结果JSON在stdout。
传输字节超过预期立即停止，不将多余内容写盘；首次响应必须200，续传必须206并匹配Content-Range。
短响应/连接中断可以重试；错误响应大小/位置/编码、磁盘或摘要错误直接停止。

传输完成后从本地文件重算大小/SHA-1/SHA-256：SHA-1核对官方，SHA-256供后续本地溯源。
哈希支持分块更新：`H(block1 + ... + blockN)`与直接对完整文件求H一致；不需要整读大文件进内存。
这里是变长bytes，不是模型的[batch, sequence]张量；SHA-1与SHA-256不能相互换算。
校验通过再不覆盖定名文件，完整清单也先写.part，再不覆盖定名；不使用可能覆盖旧文件的os.replace。
Windows使用遇到已有目标就失败的rename，POSIX用硬链接再移除临时名字，均不复制一份大型文件。
不支持硬链接的POSIX文件系统会明确报错；依据见 [Python os.rename说明](https://docs.python.org/3.12/library/os.html#os.rename)。

成功目录只有四个文件：压缩dump、source_config.yaml、dumpstatus.json、download_manifest.json。
清单记录来源/版本/URL/许可待核对状态、配置与状态快照摘要、本地文件大小/两个摘要、UTC时间和耗时。
status=verified只代表压缩文件传输和完整性已验收；ready_for_training始终false。
它不能证明正文已提取、逐篇许可已核对或训练数据已通过验收。

首版原来无续传；现支持第6节的显式续传。失败/Ctrl-C保留本次诊断文件，不自动删除，
无完整download_manifest.json不能进入后续处理；网络中断后保留同目录，加`--resume`接着下载。
即使失败目录有完整大小的bz2，也不能把手动改名当作校验成功。正式目录需下一小步实际下载验收。

## 5. 本小步复核后提交入口

本轮验收条件是50项测试、Ruff和两份小型元数据预览通过，不要求大型dump已经下载。
在项目PowerShell由你执行：

```powershell
git add configs/wikimedia_20261001.yaml src/minigpt_cn/data/download.py tests/test_download_dump.py README.md data/README.md docs/m2-formal-data-plan.md docs/m2-download-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git --no-pager diff --cached
git commit -m "feat: add pinned Wikimedia download and verified provenance manifest"
```

add仅暂存入口/配置/测试/说明；两条diff检查暂存范围和内容；commit保存本地快照。
不提交data/raw下载文件、快照/清单、缓存或无关根目录文件，助手不执行Git写操作。
本模块未改作业环境/源码、旧pilot样本/词表/权重或CS336_note.md。M2尚未全部完成。

## 6. 接着已有172 MB续传（当前步骤）

**本次已续传完成并通过独立整文件复核，不要对已完成的v1目录再次执行下载/续传命令。**
本段命令保留作续传操作示例；其他未完成目录出现网络中断时才使用它。

用户的Wikipedia首次传输在171835020/3430912430字节处提前EOF。
现有目录的配置快照与当前配置逐字节一致，状态快照done且大小/SHA-1一致；旧.part不能删除或改名。
一次只运行一个下载进程；若助手已经启动续传，不要同时复制执行下面命令。

在项目WSL环境，从原目录接着下载（单行，避免遗漏续行反斜杠）：

```bash
python -m minigpt_cn.data.download --source zhwiki --output-dir data/raw/zhwiki_20261001_v1 --download --resume --retries 20
```

`--resume`需要`--download`，只接受配置/状态快照和对应.part三项普通文件。
先重新核对官方元数据，保留旧快照；从实际文件长度n发送`Range: bytes=n-`，
只接受`Content-Range: bytes n-(N-1)/N`，不接受忽略Range返回200或不同起点/总大小。
依据见 [HTTP Content-Range标准](https://www.rfc-editor.org/rfc/rfc9110.html#name-content-range)。

默认最多3次网络重试，可显式设0～20次（不含初次请求）；此处为易断链路设20次。
重试等待1/2/4/8秒、最多8秒，耗尽就非零退出；再次运行同一续传命令可接着新进度。
不会并行多连接、切镜像、跳过TLS/校验或从头覆盖。磁盘预检查只计剩余字节加1GiB。
进度与重试提示写stderr，成功JSON写stdout。100%后仍需重读完整文件核对SHA-1并计算SHA-256。

下载期间排他创建.download.lock，正常结束/异常/Ctrl-C移除本进程自己的锁；
若进程被强杀可能留锁，需要先确认没有下载进程，再人工处理，不自动抢锁。
已存在完整dump、成功清单或未完成清单定名的目录不自动覆盖/修补，需要单独排查。
旧前缀若损坏也会被完整SHA-1校验拦住，不写成功清单。

仅Wikipedia验收后再下载Wikisource，第一次无需`--resume`，中断后才加它。
status=verified代表压缩文件通过，ready_for_training=false仍正常；本步不提取XML正文或训练BPE。

本次真实结果：总计3430912430字节，SHA-1匹配官方，SHA-256为
5db9052efe9ca97aa3da05ffe5b2f12c8ffe5e040f0a9d1ced82ec2fda5d7e2f。
续传含校验耗时658.30秒（约11分钟），本次没有再次中断或实际使用自动重试。
助手独立重算整文件两个摘要、旧171835020字节前缀摘要以及两个快照摘要，全部匹配。
目录恰好四个正式文件，无.part/锁残留；download_manifest.json的transfer.resume_bytes为171835020。

用户可在WSL只读复核清单和摘要（第二条会读取整个压缩包，不修改文件）：

```bash
python -m json.tool data/raw/zhwiki_20261001_v1/download_manifest.json
sha256sum data/raw/zhwiki_20261001_v1/zhwiki-20261001-pages-articles.xml.bz2
```

检查清单status=verified、file.bytes=3430912430、file.sha256与上面的摘要一致；
ready_for_training=false正常，不能改成true来代替正文/许可验收。

复核后由用户提交，三条命令分别暂存改动、检查暂存范围、保存本地快照：

```powershell
git add src/minigpt_cn/data/download.py tests/test_download_dump.py tests/test_download_resume.py README.md data/README.md docs/m2-download-guide.md docs/experiment-log.md
git --no-pager diff --cached --stat
git commit -m "fix: resume interrupted dump downloads with bounded retries"
```

只提交代码/测试/说明，不提交data/raw中的压缩包、.part、快照或清单。
