# 实验日志

每次实验都复制下面的模板。失败实验也要记录，因为失败原因是项目复盘的重要部分。

## 2026-10-09：中文 pilot 迁移一致性

- 分支：`m1/transformer-core`；本次工作未执行 Git 提交。
- 来源：作业 pilot 第5000步 checkpoint；项目配置 `configs/pilot_3m.yaml`。
- 参数：3,737,856；tokenizer：2048 BPE；完整验证：37,826 个目标。
- WSL PyTorch：2.11.0+cu130；GPU：RTX 4060 Laptop。
- CUDA/BF16：原实现与项目 loss 均为3.8759773192，PPL=48.2298111859。
- CPU/FP32：原实现与项目 loss 均为3.8759228132，PPL=48.2271824449。
- 两种模式均逐块检查完整 logits 与 loss，最大误差为0；共享 Parameter 重载后保持。
- 7篇验证文档重编码与原 uint16 文件逐 ID 相同；3条 prompt 的 greedy 输出逐 token 相同。
- 原始报告：`logs/pilot-migration-cuda-bf16.json`、`logs/pilot-migration-cpu-fp32.json`，不提交。
- 复现、SHA-256、接口和数学复习见 `docs/pilot-migration.md`。
- 结论：迁移行为一致；生成仍重复/不连贯，不证明正式33M资源或质量。
- 下一步：建立独立项目 WSL 环境，再接入配置 checkpoint、训练/评估/生成入口和 KV Cache。

## 2026-10-10：步骤2C 配置与 checkpoint

- 新增统一运行配置与版本1 checkpoint；本次未执行 Git 提交。
- 训练/数据/模型配置、实际优化器参数组、真实步数、RNG 和采样状态可保存重载。
- CPU/FP32 与 CUDA/BF16 恢复测试：下一批 ID、dropout 后 loss、下一步参数逐元素相同。
- 原作业 AdamW 未在本段迁移；测试用原生 AdamW 单独验收 checkpoint 机制。
- 旧 pilot 显式导入为推理格式，iteration=5000、共享 Parameter 保持，不伪造精确续训状态。
- 新格式完整验证：37,826 个目标，loss=3.8759773192，PPL=48.2298111859，logits/loss 差异为0。
- 原 checkpoint SHA-256 仍为 `505fafd02381b8a92dc359a6c7f0801db9f444d23a1e11c7f7123b7eacc61ce0`。
- 忽略的本地产物：`checkpoints/c_verification/pilot_3m_v1.pt`、`logs/checkpoint-c-verification.json`。
- 复跑命令见 `docs/step2-guide.md` C段；下一段接入训练与完整评估 CLI。

## 2026-10-10：步骤2D 训练与完整评估入口

- 接入原作业 AdamW、余弦调度、全局梯度裁剪和 get_batch，保留公式与中文注释。
- 原作业 AdamW 对照20次更新：参数和一/二阶矩逐元素完全一致。
- 训练参数/梯度/矩状态 FP32，CUDA BF16 autocast；完整验证不再只采一个 batch。
- Windows、独立 WSL：95 passed，Ruff通过；CPU-only CI 跳过2个CUDA测试。
- CPU/FP32 固定小 batch 60步过拟合：最终 loss 小于初始的10%。
- CPU/FP32、CUDA/BF16 的真实训练循环：含 dropout 时连续/恢复状态完全一致。
- 独立评估旧 pilot：全部37,826个目标，loss=3.8759773192044915，PPL=48.2298111859。
- 新中文短运行：原2048 tokenizer和数据，3,737,856参数、batch8/context256、100步。
  从随机权重开始，完整验证 loss：0步7.6813722416 -> 50步6.7380504432 -> 100步6.6604861100。
- 真实CUDA/BF16对照：连续100步 vs 50步保存+恢复到100步，权重/AdamW/RNG/采样状态完全相同；
  恢复后记录的训练loss和lr也相同。不是原5000步pilot的续训或长期质量实验。
- 新片段目录拒绝覆盖；恢复只允许改训练终点，语料哈希/PyTorch/其他配置必须相同。
- loss、梯度、参数、矩状态非有限时退出，无正常最终 checkpoint。
- 本地产物均忽略：`checkpoints/d_verification/`、`logs/d-pilot-evaluation.json`，
  完整对照摘要 `logs/d-training-verification.json`；原作业checkpoint校验和未变。
- 未实现生成CLI、KV Cache、正式配置测速、W&B与best checkpoint策略，未执行Git提交。

## 2026-10-11：步骤2E 中文生成 CLI

- 接入原generate.py逻辑，增加中文公式/形状注释、ModelOutput、greedy和独立采样Generator。
- checkpoint配置/tokenizer自动加载，单条、非流式、无KV Cache；BF16前向、FP32概率。
- 新增51个生成测试；覆盖原temperature/top-p/EOS/裁剪/no_grad以及CLI/UTF-8/seed/错误路径。
- 最终Windows/独立WSL：146 passed、Ruff通过；隐藏GPU后CPU测试143 passed、3 skipped。
- 三条prompt的CUDA/FP32：采样64token和greedy32token与原作业参考逐ID相同，原checkpoint哈希未变。
- 实际CLI读取原5000步pilot和项目100步checkpoint均可生成；不import作业包、不重训。
- 旧pilot greedy64token存在“他的数学家”等重复；100步模型greedy64token主要输出逗号。
- 旧pilot FP32采样128token（T=0.8、p=0.9、seed42）出现英文碎片和非法UTF-8，保留1个替换字符。
  明确记录这些失败样例，不把接口验收当作中文语义质量验收。
- 生成库推理阶段eval/no_grad，结束恢复原模式；NaN/+Inf/全屏蔽候选明确退出。
- 原始样例忽略目录 `logs/e_verification/`，生成迁移摘要 `logs/e-generation-verification.json`。
- 使用独立项目环境，未改作业源码/学习笔记/模型权重，未执行Git写操作。下一步F实现KV Cache。

## 2026-10-11：步骤2F KV Cache 与窗口重建

- 在现有模型上增加可选缓存路径，保留原作业/项目注释与公开forward接口，不改权重键名。
- 每层保存已做RoPE的K与未旋转的V；新位置offset=历史长度，mask为[new,past+new]。
- 缓存只用于eval推理，由调用者持有；不进入checkpoint，不跨请求复用、不原地修改旧缓存。
- 生成默认仍无缓存；`--use-cache` 开启prefill/单token decode，满窗口重建并重置位置。
- Windows/独立WSL有CUDA：181 passed，Ruff通过；隐藏GPU后CPU：176 passed、5 skipped。
  本段新增35个用例（33个缓存/基准用例与2个生成入口回归用例）。
- 真实旧pilot与用户项目100步checkpoint：三条prompt，FP32/BF16，各新增64token，
  共12组cached/uncached greedy逐ID一致。两份缓存CLI报告也与用户E阶段报告逐ID一致。
- 旧pilot超长prompt300token/窗口256，在FP32/BF16下各新增8token，重建结果逐ID一致。
  满窗口基准记录incremental_calls=0、window_rebuilds=7，持久KV=1,048,576 bytes。
  此时两种模式都计算整个窗口，没有算法上的增量加速，短测量波动不能解释为持续缓存收益。
- 无缓存完整迁移回归：37,826目标，loss=3.8759773192044915、PPL=48.22981118588135，
  与作业完整logits/loss差异仍为0，共享参数保持，原作业checkpoint校验和未变。
- 独立短prompt基准：RTX4060 Laptop、PyTorch2.11.0+cu130、BF16、prompt3token、输出64token，
  每种模式预热2次/测5次，计时仅覆盖greedy生成、禁用EOS，不含加载/文本编解码。
  无缓存/缓存总时间中位数0.641453/0.596017秒，约1.076倍；后续token约99.53/107.42 tokens/s，
  首token约9.09/9.01ms，CUDA allocated峰值31,783,936/31,625,216 bytes；持久KV270,336 bytes。
  笔记本未锁频，原始trials全部保留；小pilot提升有限，不代表33M吞吐、长期速度或训练预算。
- 单条非流式生成仍会重复/不连贯；100步模型仍主要输出逗号，缓存不改变语言质量。
- 报告均在忽略目录 `logs/f_verification/`：`pilot_bf16_isolated.json`、
  `pilot_bf16_full_window.json`、`uncached_regression.json`、两份cached CLI报告。
  `pilot_bf16_short.json` 是与测试并行的早期smoke，不用于上述独立基准结论。
- 用户逐步命令见 `docs/step2-guide.md` F段。未执行Git写操作，未改作业源码/学习笔记/原权重。
- 下一步由用户复核F，再讨论正式16K tokenizer及33M/512的资源与稳定性试跑；未开始正式训练。

## 2026-10-11：M2 第一小步——BPE 训练函数

- 分支：`m2/tokenizer-data`；只接入基础函数，不接入CLI、数据处理或正式16K训练。
- 原作业来源：`cs336_basics/tokenizer.py::train_bpe`；项目文件：
  `src/minigpt_cn/tokenizer/train_bpe.py`，公开导入 `from minigpt_cn.tokenizer import train_bpe`。
- 保留按词块频次加权的pair统计、字节平局规则、非重叠替换、特殊token隔离和merge顺序。
  复用编码器PAT，省掉中间chunks列表，补输入校验与中文注释；输入语料只读。
- 新增22个独立小语料用例；与原编解码测试合计31 passed，测试不依赖作业仓库。
- 单独读取作业 `corpus.en`：目标257/260/500，实际词表与完整有序merges均逐项一致。
- Windows完整回归203 passed；独立WSL CPU-only 198 passed、5 skipped；两环境Ruff通过。
- 词表目标是上限，语料不足时提前停止、不填充；模型必须匹配实际词表大小。
- 限制：仍整读文本、每轮全量统计pair；没有正式16K语料的时间/内存基准，不宣称可扩展性已验收。
- 用户复核本模块后再接小语料训练/保存CLI；作业源码、学习笔记、现有词表/权重未修改，未执行Git写操作。

## 2026-10-11：M2 第二小步——小语料训练与保存 CLI

- 第一小步已由用户复核；本轮只新增 `python -m minigpt_cn.tokenizer.train`，不接数据处理或正式16K训练。
- 显式输入/目标/输出；只接受UTF-8纯文本，不自动提取JSONL，不读取token二进制。
- 默认EOS为256，可显式声明特殊token；目标是上限，不填充，报告实际词表、merge数和输入/输出SHA-256。
- 保存后核对词表/merge顺序/特殊token一致，并检查中文/emoji往返；失败不报告成功。
- save增加兼容的overwrite参数；CLI用独占创建防覆盖，包括并发同名文件与悬空符号链接。
- 新增17个CLI/保存用例；BPE相关共48项，Windows无符号链接权限时47 passed、1 skipped。
- 完整回归：Windows219 passed、1 skipped；独立WSL CPU-only215 passed、5 skipped；两环境Ruff通过。
- 真实WSL演练：读取原pilot训练文本1,010,020 bytes，目标/实际300、merge43、EOS256，
  reload_verified与round_trip_verified均为true；训练函数耗时7.6844秒，仅单次smoke，不是正式性能基准。
- 输入SHA-256：`3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba`。
  原2048 tokenizer哈希仍为 `22089eb3002b9c24509c5bc37cef710994c9ace095f600bb26a71877b6e1ea12`，输入未修改。
- 忽略产物：`data/processed/m2_verification/tokenizer_cli300.json`，
  SHA-256 `a369cf5f15f4fc203189c0747eb430eb8f89f8ab1427905c28319662a3d68392`。
- 新教程 `docs/m2-tokenizer-guide.md` 提供用户独立输出名与检查步骤；300词表不能替换旧2048或正式16K词表。
- 仍整读/逐轮扫描，未自动保存正式数据清单/训练配置；未下载语料、改模型配置/作业源码/学习笔记或执行Git写操作。
- 停在本模块用户验收点，下一轮再讨论清洗和划分的独立小模块。

## 2026-10-11：M2 第三小步——单篇文档清洗与过滤

- 第二小步已由用户复核并提交；本轮只接入独立函数，不下载/读写语料或实现去重、划分、编码。
- `src/minigpt_cn/data/cleaning.py` 保留原 `prepare_pilot.py::normalize_text` 的顺序：
  NFC、繁转简、CRLF/CR 统一 LF、逐行 strip、整体 strip；中文注释说明每一步。
- 只读核对作业环境实际依赖为官方 OpenCC 1.4.2；项目固定同版本与 t2s.json，
  仅更新 Windows/WSL 项目环境，作业环境未修改。繁转简可显式关闭。
- 保留行内空白、段落、全角字符、标点、英文和 emoji，不使用 NFKC。
  clean_document 默认只过滤空文档；可选长度阈值按规范化正文码点计，包含边界，整篇过滤不截断。
- 新增26个独立CPU用例；完整回归：Windows245 passed、1 skipped；WSL CPU241 passed、5 skipped；两环境Ruff通过。
- 真实pilot1000篇原始文档逐篇与原规范化函数完全一致。通过AST只编译原函数，
  不执行原文件的顶层mkdir或main；原源码与原始语料SHA-256前后相同。
- 新教程 `docs/m2-cleaning-guide.md` 提供项目环境、测试、单篇手工样例与用户Git命令。
  未重写旧pilot数据/词表/权重，未改学习笔记或执行Git写操作，无关根目录文件保留。
- 正式长度阈值、全量管线及处理配置清单尚未确定/接入；停在本模块验收点，后续再接精确去重和hash划分。

## 2026-10-11：M2 第四小步——正文精确去重与固定 hash 划分

- 用户已验收并提交清洗模块；本轮只新增splitting.py函数层，不接文件处理CLI、下载或编码。
- deduplicate_texts按完整清洗正文相等去重，保留首次出现顺序；输入只遍历一次，seen仍保存全部唯一正文。
- choose_split保留原split_pilot.py公式：种子42、NUL分隔、UTF-8、SHA-256转整数取模100；
  0～97=train、98=valid、99=test。比例是期望值，不是精确文档数/token数约束。
- 新增21项独立测试：桶边界、去重/格式差异、清洗组合、生成器单次遍历、重排/追加不变、
  三集合无交集、不同PYTHONHASHSEED的独立进程一致，以及非法正文/种子。
- 完整回归：Windows266 passed、1 skipped；WSL CPU262 passed、5 skipped；两环境Ruff通过。
- 只读真实pilot原始正文→项目清洗→去重→划分：1000篇唯一正文、986/7/7；
  与原清洗文件、三个split文件的正文及顺序完全一致；原choose_split逐篇结果一致。
  AST仅编译原函数，不执行main；原划分源码、原始/清洗/三份split数据SHA-256前后均未变。
- 新教程docs/m2-splitting-guide.md提供47项聚焦检查与三集合手工例子。本轮没有新依赖。
- 函数处理正文，不含JSONL元数据；正式管线需保留首次记录的来源信息并检查空split。
  不承诺正式大语料内存表现；未改作业环境、词表/权重、学习笔记，未执行Git写操作，无关文件保留。
- 停在本模块验收点；下一小步再接JSONL处理入口，不同时开始下载或token编码。

## 2026-10-11：M2 第五小步——JSONL 处理入口与清单

- 第四小步已由用户验收并提交；本轮新增data/prepare.py，未改清洗/划分算法或接下载、BPE训练/纯文本导出、token编码。
- 重复--input按顺序读取UTF-8 JSON对象行；text须为字符串，拒绝坏JSON/UTF-8、非有限JSON常量。
  保留首次记录元数据，仅规范化text；空正文过滤后全局正文set去重、长度过滤、固定hash划分。
- 输出目录必须新建，文件独占创建；成功后写manifest记录输入/输出SHA-256、字节数、规则/字典版本/阈值、计数。
  检查读取版本/最终输入版本、计数平衡与非空三集合；失败报行号或原因，不输出成功报告。
- 不是目录级事务；失败可能保留新目录中的部分文件，无完整清单不可用于训练，换目录重跑。
  seen仍保存唯一正文；输入来源/许可不自动核实，正式资源与下载尚未开始。
- 新增24项入口测试：Windows23 passed、1 skipped，WSL24 passed；含跨源去重/来源保留、
  可选转换与长度、字节确定性、输入变化、并发目标创建/悬空符号链接、不覆盖、独立入口及无Torch导入。
- 完整回归：Windows289 passed、2 skipped；WSL CPU286 passed、5 skipped；Ruff通过。
- 真实WSL CLI：原pilot1000篇，read=saved=1000，各类过滤计数为0；train/valid/test=986/7/7。
  新三个JSONL与原文件逐字节一致（含元数据），原输入及三个split文件SHA-256前后未变。
- 输入SHA-256：b29dc80174cd9f95229deaf8e6cb939ade7d44247261f08715353d146d9a642a。
  忽略产物：data/processed/m2_verification/pilot_jsonl_v1；用户教程使用独立目录m2_user_pilot_jsonl。
- 新教程docs/m2-data-prepare-guide.md含71项聚焦测试、CLI与清单/文件检查；
  未改旧数据/词表/权重、作业环境或学习笔记，未执行Git写操作，无关文件保留。
- 停在本模块验收点，后续再接仅train的BPE训练文本导出，不直接开始正式16K或token编码。

## 2026-10-11：M2 第六小步——仅 train 的 BPE 文本导出

- 用户已验收并提交JSONL入口；本轮新增data/export_text.py，不改既有清洗/划分/BPE算法，不训练tokenizer或编码。
- 从版本1处理清单固定读取同目录普通train.jsonl；校验格式、路径、SHA-256/字节数、完整文档/字符计数和逐篇train归属。
  测试通过禁止打开valid/test文件确认正文不会混入BPE；导出前/读取时/结束时核对来源版本。
- 正文不再清洗，按原pilot正文+EOS+LF拼接UTF-8；正文含保留EOS字面量时明确拒绝，来源/标题不写入文本。
- 可选max_bytes按EOS/LF计入预算，仅取完整文档前缀，不截断或跳过大文档挑后文；限量时仍校验完整train。
- 输出新目录train.txt与export_manifest.json，记录处理清单/训练文件哈希、选择/边界规则及实际输出统计；不覆盖，不改原清单。
  失败可能留下部分文本，没有完整导出清单不能视为成功；不宣称许可、前缀代表性或正式BPE资源已经验证。
- 新增26项测试：WSL26 passed；Windows24 passed、2 skipped。覆盖全量/限量、预算包含边界、
  来源计数/版本变化、路径重定向、错split、EOS字面量、防覆盖及符号链接、独立进程入口。
- 完整回归：Windows313 passed、4 skipped；WSL CPU312 passed、5 skipped；Ruff通过。
- 真实项目清单导出：1MiB预算得到50篇、382621正文码点、1010020字节；与作业bpe_train.txt逐字节一致，
  SHA-256为3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba。
- 全量导出986篇、6580357正文码点、17677160字节，与项目train逐篇正文+EOS/LF拼接完全一致，
  SHA-256为0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
- 忽略产物data/processed/m2_verification/bpe_text_prefix1m与bpe_text_full；用户教程用独立m2_user_bpe_text_1m。
  原项目清单/三份split和旧作业BPE样本SHA-256前后未变；未改作业环境、词表/权重、学习笔记或执行Git写操作。
- 新教程docs/m2-train-text-guide.md提供26项聚焦测试、1MiB CLI与完整来源/边界/旧样本校验；停在用户验收点。
  后续先讨论训练规模/性能，17.7MB文本导出不等于正式16K训练可用性验收。

## 2026-10-11：M2 第七小步——增量 BPE 与性能复核

- 用户已验收并提交train纯文本导出；本轮只优化项目BPE，不改作业源码、编码器/CLI接口、文件格式或模型配置。
- 将旧全扫描算法保留为tokenizer/train_bpe_reference.py；生产train_bpe使用pair→词块倒排索引、
  缓存词块局部频次、仅合并邻边的有符号增量、最大优先队列/懒删除/定期堆重建。
  固定词块编号和权重，不重建全部freq；预分词、特殊token隔离、非重叠替换、字节序平局均不变。
- 新增34项测试：20个随机种子、6种预算的逐ID词表/有序merge对照及round-trip；
  每轮重算检查全局频次和倒排索引，邻边增量与全文重算对照，旧堆节点/完全平局/强制重建测试。
- 聚焦WSL82 passed，Windows81 passed、1 skipped。全部CPU回归：Windows342 passed、9 skipped；
  WSL346 passed、5 skipped。CPU模式跳过CUDA，Windows另有符号链接权限跳过；两环境Ruff通过。
- WSL项目Python3.12.3、CPU、独立子进程单次计时，仅含读取/预分词/训练，不含导入/哈希/保存/重载。
  官方速度用例实际是134042字节英文/500词表、阈值1.5秒；不是正式16K或中文大语料测试。
  本次作业函数0.9715秒、项目旧参考1.7150秒、优化版0.1195秒，逐ID词表/merges摘要完全相同。
  迁移旧版用了Counter/any等不同写法，不能把作业速度直接当作项目速度。
- 中文1010020字节：300词表旧参考7.8283秒、优化1.9314秒，词表/merge完全相同；
  优化512词表3.7545秒（旧参考此前35秒超时），优化2048词表5.9879秒。
  2048逐ID词表及1791个有序merge与原pilot tokenizer完全相同，没有替换它。
- 全train17677160字节：优化300词表35.5896秒、峰值RSS2461MiB；
  优化2048词表109.4505秒、峰值RSS2824MiB，实际词表2048，未保存任何新词表。
  全train使用更多正文，结果不与旧50篇pilot词表混同。小样本每项限35秒，全train最终测量每项限120秒。
- 空间换时间：1MiB/300旧参考峰值约40MiB，优化约172MiB；1MiB/2048优化约219MiB。
  仍整读UTF-8文本并保存唯一词块/局部计数/倒排索引，仍扫描受影响词块来构造新序列。
  不宣称整体线性、无限大语料可训练或正式16K已验收；正式BPE采样预算需先单独确定。
- 真实输入、官方fixture/速度测试、原作业训练源码和旧2048 tokenizer的SHA-256前后不变。
  原pilot train文本SHA-256仍为3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba；
  全train仍为0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
- 新教程docs/m2-bpe-optimization-guide.md提供公式、边界、资源数字和只读一致性复核命令。
  未训练正式16K、下载/编码数据、改学习笔记/旧权重/作业环境或执行Git写操作；无关根目录文件保留。
  停在本模块验收点，不把BPE优化等同于M2全部完成。

## 2026-10-11：M2 第八小步——16K BPE 资源探测

- 用户已验收并提交BPE优化，算法commit为319a8c1；本轮只做只读CPU探测、教程和记录。
  未修改算法/模型配置，不保存新tokenizer，不下载正式语料、编码uint16或训练模型。
- 输入为已验收全train文本：986篇、6580357正文码点、17677160字节。
  来源data/processed/m2_verification/bpe_text_full，与m2_user_pilot_jsonl清单对应；
  SHA-256为0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
  启动前验证导出/处理清单、train文件哈希、边界/全量选择/统计；不使用valid/test训练BPE。
- WSL项目Python3.12.3、CPU，探测前总内存约15.27GiB、可用约14.27GiB。
  单独进程设置300秒训练/检查限时、RLIMIT_AS=6GiB虚拟地址空间上限；这不是RSS或显存限制。
  仅进程内临时包装选择函数，每2000次完成合并报告状态；磁盘源码未改。
- 实际完成16000词表、15743次merge；训练148.118507466秒，峰值RSS3210.609375MiB（约3.14GiB）。
  计时包含读文件/预分词/合并和少量阶段日志，不含来源哈希、构建编码器或往返检查。
  前2000次合并约116.62秒，后续更快；不能按merge次数或文件字节数直接线性外推。
- 检查ID为连续0..15999、256个初始字节正确、EOS ID=256、词条bytes全部唯一、
  每个新词条等于有序merge左右拼接，普通词条不含EOS字面量，六组样例ID范围和往返正确。
  样例覆盖简繁中文、英文/数字、emoji/肤色修饰、组合字符、tab/LF、空文本和多个EOS。
  这些检查不证明随机生成字节永远合法UTF-8，也不代表正式语料质量已经验收。
- 文本、导出/处理清单、源train JSONL、旧2048 tokenizer、正式模型配置、训练/编码器源码
  SHA-256前后均相同；探测词表随进程退出释放，tokenizer_saved=false，旧checkpoint不改。
- 复跑WSL CPU tokenizer聚焦测试82 passed；Windows及WSL Ruff通过。
  本轮只改文档，不新增测试/依赖；此前完整CPU回归342/9（Windows）、346/5（WSL）未重跑。
- 新教程docs/m2-bpe-16k-probe-guide.md提供独立train导出、同上限只读复核和仅文档的提交命令。
  正式来源/许可、采样预算、16K保存交付和uint16编码仍待独立验收；不把本次probe等同M2完成。
  未改作业仓库/环境/学习笔记、未执行Git写操作，无关根目录文件保留。

## 2026-10-11：M2 第九小步——正式来源核对与 BPE 采样方案

- 用户已提交第八小步，HEAD=0b679ec，仍在m2/tokenizer-data；本轮只改四份文档，不改源码/测试。
- 只读查询官方目录、dumpstatus.json和许可页面，保留Wikipedia + Wikisource双来源路线。
  候选版本均为20261001，两份状态文件全部已列出job为done；使用有日期的pages-articles.xml.bz2地址。
- Wikipedia压缩文件3430912430字节，官方SHA-1=04d5eec3c249fdf8604b97f4435e019c60a4ec94；
  Wikisource压缩文件9565159689字节，官方SHA-1=b1a6e0c5ae0b55692c41010a47f7eb02e8087e2a。
  合计约12.10GiB，不含提取/处理/编码文件；未下载，本地SHA-256未知，不伪造为已验收清单。
- 官方许可说明/使用条款及Wikisource版权说明已查阅；不把全部文库作品标为公有领域，
  计划保留页面/修订/URL及许可依据。逐篇许可、重复记录溯源、正文范围和章节跨split风险尚未验收。
- 正式BPE建议起点16MiB=16777216字节，包含EOS/LF，仅train，seed42；独立域采样哈希排序后
  取完整文档前缀，记录实际来源占比。该模式本轮未实现，现有export_text不改变行为。
  预算不是模型100M训练tokens，不保证正式文本沿用pilot耗时/内存；初测继续300秒/6GiB虚拟地址边界。
- 新docs/m2-formal-data-plan.md区分候选元数据、待核对许可、待实现采样和后续独立验收步骤。
  本轮未运行pytest；不把前次通过数算成本次重跑，不下载/保存词表/编码/训练。
  未改作业仓库/环境/学习笔记，未执行Git写操作，根目录无关文件保留，停在方案复核点。

## 2026-10-11：M2 第十小步——有预算的确定性哈希采样

- 用户已提交方案，HEAD=777e6c2，仍在m2/tokenizer-data；本轮实现采样，不下载正式数据或训练模型。
- 新data/sampling.py使用独立域SHA-256和UTF-8正文平局排序、候选最大堆；每次添加后仅保留
  累计字节首次覆盖预算的最短哈希前缀，末篇越界时不写。预算包含EOS/LF，不截断、不跳选。
  候选正文不缓存全量，但有Python对象/单篇文档开销；上游prepare的seen和正式规模内存仍待验收。
- export_text新增selection=hash、sample_seed默认42；hash必须显式给正整数max_bytes，
  prefix拒绝sample_seed，默认prefix的输出顺序/字节和清单字段不变，不改原处理/划分规则。
  校验完整train的哈希/计数/归属和EOS字面量，不读取valid/test正文；结束时再核对来源版本。
- hash清单增加算法版本/seed/规则及入选来源统计；source字符串或source.name可识别，
  其他/缺失为null，不猜测来源/许可，字节数包含EOS/LF，不声称来源均衡。
- 新增43项测试：独立排序/预算等号、10个随机seed逐次堆不变量、输入重排、采样seed、
  UTF-8平局、首篇超预算/不可跳选、全train错误/计数、输入变化、防覆盖、CLI与旧模式兼容。
  Windows聚焦67 passed、2 skipped；完整CPU回归385 passed、9 skipped。
  WSL完整CPU回归389 passed、5 skipped。Windows Ruff通过；跳过含CUDA和Windows符号链接权限。
- 真实pilot来源train共986篇，SHA-256=1250226874b95e47ef9f900140a83c315e420886558902cb23c4ca61346b3699。
  16MiB预算/采样seed42输出942篇、6244484正文码点、16742929字节；
  SHA-256=5a322b2d46ceb5081f853ac3c4fb1ba5dca0ca19c79756a87485a66c13567f1f。
  所有入选来源为null，符合该pilot缺少可识别source元数据的事实，不能充当正式来源验收。
- 单次WSL CPU采样约1.5709秒，截至首次导出进程峰值RSS43.625MiB；计时不含后续独立排序。
  不把采样速度当作BPE训练速度，不外推正式大语料。独立全排序和重复导出逐字节一致。
  原默认1MiB前缀仍50篇/1010020字节，SHA-256=3c7c547d9a705e23ebb23f23960b59cd31adb1d1de5fe405269f13d9a005fcba。
- 三份split/处理清单、旧词表/BPE文本/pilot权重、模型配置/BPE源码等10个受检文件哈希前后不变。
  助手忽略产物仅位于m2_verification/bpe_hash16m_v1、bpe_hash16m_repeat_v1、bpe_prefix_compat_v1；
  用户教程使用独立m2_user_bpe_hash16m。未改作业/环境/学习笔记或执行Git写操作，根目录无关文件保留。
  新docs/m2-hash-sampling-guide.md提供测试、CLI、独立排序PASS与提交命令；停在模块复核点。

## 2026-10-11：M2 第十一小步——固定版本下载入口与来源清单

- 用户已提交哈希采样，HEAD=45515b0，仍在m2/tokenizer-data；本轮只接下载，不改模型/BPE/处理规则。
- 新configs/wikimedia_20261001.yaml固定两站点URL/任务/大小/SHA-1和许可依据，许可仍需逐篇核对。
  新data/download.py默认离线预览；check_remote只读最多2MiB官方状态；download必须显式开启，
  始终检查官方job=done、URL/大小/SHA-1与固定配置一致，不追latest，不关闭TLS或自动跟随重定向。
- 下载要求新目录和压缩大小加1GiB余量；1MiB分块写.part，本地重读核对大小/SHA-1并计算SHA-256。
  配置/官方状态快照和完整下载清单保留溯源；成功status=verified但ready_for_training=false。
  Windows不覆盖rename、POSIX不覆盖硬链接定名，不复制第二份大型dump；无续传/自动重试，失败保留诊断文件。
- 初次Windows测试发现沙箱不允许硬链接；依Python官方平台语义改为Windows原生不覆盖rename。
  已在两平台成功路径与目标冲突路径复跑，无旧文件被覆盖；不跳过下载成功测试来掩盖权限限制。
- 新50项小型离线测试：预览无网络/无目录、两来源传输和快照/摘要、非法配置/超时/元数据、
  短/长/错误摘要/编码/206/超时、空间不足、目录竞争、配置变动、清单定名失败、CLI与拒绝重定向。
  WSL聚焦50 passed；完整CPU回归Windows435 passed、9 skipped，WSL439 passed、5 skipped；两边Ruff通过。
  小型bz2响应约101字节，不访问公网；完整回归跳过CUDA及Windows符号链接权限测试。
- 实际只读查询两份官方dumpstatus.json，配置的固定大小/SHA-1/URL全部匹配，WSL项目入口的两份
  check_remote预览均planned、job=done、file=null、输出目录未创建。状态JSON摘要不混作dump摘要。
  PowerShell HEAD两份压缩文件均HTTP200，Content-Length为3430912430/9565159689，无额外Content-Encoding。
- 项目D盘本次可用约375.03GiB；压缩包合计约12.10GiB，不估计提取/去重/编码总空间或下载性能。
  本轮未传输大型dump，本地dump SHA-256未知；正式下载和正文提取待后续用户验收。
- 新docs/m2-download-guide.md提供测试/无大型传输的预览和提交命令；本轮未新增依赖，
  未改作业/环境/学习笔记/旧数据/词表/权重，未执行Git写操作，根目录无关文件保留，停在入口复核点。

## 2026-10-11：M2 实际下载中断后的显式续传

- 用户Wikipedia下载在171835020/3430912430字节处提前EOF；快照匹配、磁盘约374.72GiB可用。
  只读1字节Range探测返回206及正确位置，不能据此确定断开的具体网络环节。
- 在data/download.py补resume/retries，保留旧快照和前缀，重新核对官方状态；续传只接受匹配的
  Content-Range，服务器忽略Range/版本不一致/摘要错误直接失败，不覆盖或自动换镜像。
- 默认3次有限网络重试，允许0～20次，等待1/2/4/8秒（上限8秒）；磁盘写入错误不重试。
  排他锁防止并发追加；完成后仍重读整个文件核对SHA-1并计算SHA-256，检查快照未变再写成功清单。
- 新tests/test_download_resume.py覆盖已有/空/完整.part、错位响应、断流/超时/连接失败重试、
  重试耗尽、损坏前缀、配置/快照改变、非法目录、TLS/HTTP错误分类与Ctrl-C锁清理。
  加上原入口共91项下载测试；CPU全量Windows476 passed、9 skipped，WSL480 passed、5 skipped；Ruff两边通过。
- 已启动原目录续传，resume_bytes=171835020，retries=20；未启动Wikisource、正文提取或训练。
  启动时旧前缀SHA-256=004191d34d56755e37cd2aadce0f681097921ea07362f7e0c0ec2a38aca7e980。
  最终退出码0，status=verified，总计3430912430字节，续传含校验耗时658.300458秒。
  本次真实传输未再次中断，重试路径仅由离线测试验收，不能宣称实测网络重试性能。
- Wikipedia最终SHA-1匹配官方，SHA-256=5db9052efe9ca97aa3da05ffe5b2f12c8ffe5e040f0a9d1ced82ec2fda5d7e2f；
  助手独立重算整文件SHA-1/SHA-256、旧前缀和快照摘要，全部匹配，目录仅四项正式文件，无.part/锁残留。
  清单路径data/raw/zhwiki_20261001_v1/download_manifest.json，ready_for_training=false。
  压缩文件已验收，正文/逐篇许可/正式双来源语料仍未验收；Wikisource本轮未下载。
- 未修改作业环境/学习笔记/旧pilot或执行Git写操作；复核与提交命令见下载教程第6节。

## 2026-10-12：M2 流式XML解析与原始wikitext小样本

- 当前HEAD=81a7267，分支m2/tokenizer-data；上一轮续传补丁未提交，保留已有改动与根目录无关文件。
  用户已完成Wikisource下载；助手本地独立重算两份压缩包摘要和快照，均匹配，未联网下载。
  Wikisource SHA-256=8433426aee0ad8c00121a0bd057182543fef9bb12e2fb4fbb8f06258e091b703。
  后续下载全部由用户执行；更新README/data说明中的双来源完成状态。
- 新data/parse_xml.py在bz2流上以16KiB块供给ElementTree，页结束后clear+remove，
  校验中文export-0.11和siteinfo/dbname；仅保存主命名空间非XML重定向的主槽位wikitext。
  默认扫描100页/64MiB解压预算，最多10000页，仅用于小样本；不保存完整解压XML。
- 来源清单/两份快照与整压缩包在解析前后校验；新目录输出wikitext.jsonl和sample_manifest.json。
  保留标题、两种ID、时间、固定修订/历史URL、来源版本、许可待核对状态与原始文本摘要。
  不输出编辑者用户名/IP；不跟随外部内容，不展开DTD/自定义实体，不清理维基标记或做繁转简。
  使用wikitext而非text字段，ready_for_training=false，不能直接接prepare/BPE/模型。
- 37项离线测试覆盖双来源、实体/繁体/emoji/元数据、主槽位、所有过滤计数、扫描页数/字节预算、
  1500页根节点不累积、坏XML/压缩/清单/摘要、输入改变、不覆盖和CLI。Windows/WSL聚焦均37 passed；
  全量CPU回归Windows513 passed、9 skipped，WSL517 passed、5 skipped；Ruff两边通过。
- 两份真实dump各扫描前100页：Wikipedia保留67、非0命名空间30、重定向3、读取解压XML5210112字节；
  Wikisource保留81、非0命名空间6、重定向13、读取1458176字节；其余跳过0，stop_reason=page_limit。
  样本分别位于data/processed/m2_verification/xml_zhwiki100_v1和xml_zhwikisource100_v1，均被Git忽略。
- JSONL输出字节数4242482/1435883；SHA-256分别为
  2e4d24749ceafe3b9eb1f94466b750cdc990247e23e0b06ed6b5a7463ed03331 /
  51d7fc4c8deeb63ffbd8afdadbd47143c2c78b84ba0b27a6d933805ae9b47632。
  独立XMLPullParser逐项比对所有保存记录的标题/ID/时间/wikitext，完全一致，计数与输出摘要匹配。
- WSL端到端（含解析前后整份压缩输入摘要）34.66/96.78秒，同一进程累计峰值RSS约36.26MiB。
  仅小样本资源观测，不推算全量正文/外存去重/BPE成本；Wikisource目录/章节与逐篇许可仍待核对。
- 新docs/m2-xml-sample-guide.md提供用户测试/新目录复跑/复核/提交步骤；未新增依赖，未修改
  原dump/旧pilot/词表/权重/作业环境/CS336_note.md，未执行Git写操作，未训练或全量提取。

## 实验模板

### 基本信息

- 实验名称：
- 日期：
- Git commit：
- 运行设备：
- 配置文件：
- 随机种子：

### 数据

- 数据版本与校验和：
- 训练 / 验证 token 数：
- tokenizer 版本与校验和：

### 训练

- 模型参数量：
- precision：
- batch size / gradient accumulation：
- 学习率与 scheduler：
- tokens/s：
- 运行时长与费用：

### 结果

- 最佳 validation loss：
- BPB / 其他指标：
- 生成样例位置：

### 观察与下一步

- 哪些设置有效：
- 哪些设置失败：
- 下一次只改变哪个变量：
