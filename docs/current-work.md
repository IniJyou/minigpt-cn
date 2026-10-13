# 当前工作与交接

更新：2026-10-13。**新会话先读本页；不要按旧逐小步教程继续支线。**

## 主线与唯一下一任务

16K tokenizer、同版本pilot三集合uint16及33M本地短训练/恢复均已保存验收，数据工具扩建继续冻结。
AutoDL 500步、下载原始产物和费用/实例状态记录均已收尾。
正式语料与首轮预训练的50元准备方案已保存：[正式预训练方案](formal-pretraining-plan.md)。
唯一下一任务：**固定正式语料来源版本、许可与样本验收包**；本轮不启动正式训练。
优先候选是Wikimedia已清洗中文Wikipedia Parquet（20231101.zh），固定revision/逐文件摘要后由用户获取；
复用既有清洗/划分和16K编码，不重新建设维基渲染或审批平台，旧20261001 raw与pilot保留。
用户总预算50元，保守按包含已报告3.09元解释；余额基准46.91元，保留实例费用也计入总额。
2026-10-13用户明确要求暂不关机以保留稀缺机器，实例继续运行；此决定保留到用户另行改变。
正式训练仍使用configs/minigpt_33m.yaml、已验收16K tokenizer、新正式三集合和同一云端框架。
用户已租用单卡vGPU-32GB实例，页面报价1.58元/时，数据盘50GB；不再沿用此前无库存的4090候选。
2026-10-13用户回传硬件核对通过：实际GPU为RTX4080 SUPER、31.48GiB、计算能力8.9；
Python3.12.3、PyTorch2.8.0+cu128、CUDA runtime12.8，BF16支持与小矩阵运算通过。
已准备configs/minigpt_33m_autodl_500.yaml（batch16/context512/BF16、500步）与
logs/autodl_500_prepare_v2上传包；操作见 [AutoDL资源验证](autodl-resource-guide.md)。
用户已回传云端上传预检查PASS：配置SHA-256为
c6f26c512959ce1e472d2ad02272c2ffb7eef30d0fe74f901bf069f6c3b227fb，
包清单/tokenizer/token数据清单摘要均与本地一致；batch16/context512/max_steps500。
云端定向CPU检查43 passed/2 skipped（3.53秒），Ruff通过；两项为隐藏CUDA后的GPU测试跳过。
项目依赖经清华PyPI镜像安装成功：regex2026.9.29、OpenCC1.4.2、pytest9.1.1、Ruff0.17.0。
用户已回传500步completed报告，配置/包清单/tokenizer/token数据清单摘要经本轮本地重新核对一致；
33104384参数、4096000训练tokens，完整valid覆盖25271目标，loss从9.7841降至6.7017。
程序总调用117.915秒；训练区间37969 tokens/s，含验证/最终保存的循环35139 tokens/s；
PyTorch峰值分配/保留显存5.804/6.242GiB（不含驱动或其他进程）。
报告核对PASS记录于logs/autodl_500_report_review_v1；其中resources.user_report.json是用户贴文的规范化副本，
不是下载的云端原始文件。用户随后已取回五份原始文件至logs/autodl_500_v1；
本轮19项独立核对PASS，记录于该目录artifact_acceptance.json。
包括五份文件摘要、原始报告/配置/输入版本、50条train及6次完整valid、两种吞吐复算、
stdout与metrics一致、checkpoint CPU重载/共享权重、500步及有限FP32优化器矩状态。
Windows本地PyTorch2.14.0+cu126仅用于CPU读取验收，未重跑训练/验证或宣称跨版本精确续训。
用户已报告累计费用3.09元，并明确保留实例运行继续主线；该金额不是最终账单，关机不是本轮验收条件。
收尾记录在logs/autodl_500_v1/closeout.json，原始验收记录的pending字段为当时快照，由收尾记录补充。
云端报告checkpoint SHA-256：4d976c1407fa1d9b0fd9914bc6dfe93bcfac7d6203b2ab82a11fc6ddeb595540。
本地checkpoint实际摘要已与上述云端摘要一致；500步资源任务完整验收，已有运行不重复。
本轮已完成预算/来源准备方案，未新下载、处理正式语料或启动训练；下一项只做正式来源验收包。
正式语料扩充/1亿token预训练与展示随后开展；独立0.6B LoRA最后开展。

## 已完成与未完成

- 已完成：模型/权重共享/dropout、BPE增量训练与编解码、训练/完整评估、checkpoint恢复、
  生成和KV Cache；模型与旧中文pilot的完整logits/loss/共享关系一致。
- 已完成但冻结：Wikimedia下载/流式XML、正文提取v2、预算子集、质量筛选、作品级去重划分。
  代码/测试留在项目供后续复用，不要求当前重做或扩建。
- 已完成：16000词表tokenizer保存/重载/往返验收，15743 merges、EOS ID256，训练122.11秒。
  同版本三集合uint16编码51.57秒；全部1000篇已用项目读取器从磁盘解码核对正文/EOS/LF。
- 已完成：33M实际参数33104384，WSL/RTX4060/BF16/batch2/context512短训练100步。
  50步保存→100步恢复与连续100步的权重/优化器/采样器/RNG完全一致；新进程重载验证相同。
  完整valid loss=9.7909→8.1517→8.0524，全部25271个目标；训练日志区间吞吐约7637 tokens/s。
  PyTorch最大分配/保留显存1.175/1.227GiB；范围与复现见 [33M短验证](33m-short-validation.md)。
- 已验收：AutoDL/RTX4080 SUPER、BF16/batch16/context512的500步资源验证与本地原始产物；费用3.09元时点及继续运行决定已记录。
- 已准备：50元总预算、首轮1亿训练tokens的预算/来源方案；正文与许可/编码未验收，不直接训练。
- 未完成：正式来源验收、正式三集合/16K编码、正式预训练与展示。
- 旧32MiB正文仍有信息损失，所有自动候选未审批，不直接训练；
  修提取器不会自动修复旧文本。未来需要新正式语料时另定页面/正文预算，不全站扫描。
- M2基础实现、pilot数据和本地训练闭环已验收；正式语料规模/许可验收和M3仍未完成。

## 当前可用输入

data/processed/m2_user_bpe_16k_probe_text/export_manifest.json与train.txt：
986篇train、17677160字节，SHA-256
0e116f4eeb82d56713dadceddc1d50b51f2917a26656ac33762a64010744ed7c。
来源为data/processed/m2_user_pilot_jsonl/manifest.json，固定986/7/7三集合。
新会话重新核对实际文件及清单；本批用于验证闭环，不是正式语料规模/许可验收。
旧2048 tokenizer与checkpoint不替换，不与新16K token ID混用。

已验收tokenizer：data/processed/pilot_tokenizer_16k_v1.json，SHA-256
21c99525eb12462f6ac26bc78a1487c61c8cbdc2c98c6709cd5b1b9988374470。
已验收token目录：data/processed/pilot_tokens_16k_v1，含三份bin、manifest.json与acceptance.json。
train/valid/test token数：4168196/25272/42263；文档/EOS数仍为986/7/7。
重载步骤见 [Tokenizer操作](tokenizer-guide.md)，编码与独立读回见 [Token编码验收](token-encoding-guide.md)。
已有输出不覆盖、不盲目重训/重编码；模型数据以新token目录为准，旧pilot仅保留。

33M短运行配置：configs/minigpt_33m_short_run.yaml，引用唯一模型配置，100步调度、50步退出恢复。
验收目录：checkpoints/m3_33m_short_v1，含first50/resumed100/continuous100三段及acceptance.json。
同一WSL项目环境PyTorch2.11.0+cu130完成训练/恢复/独立验证；Windows2.14.0+cu126不能当同版本精确续训。
这些新checkpoint是pilot短验证产物，旧checkpoint不替换。下一阶段实测资源与费用，不直接开始正式预训练。

## 环境与边界

项目：D:\aiworkspace\transformer；WSL：/mnt/d/aiworkspace/transformer。
已有项目环境：/home/luoyu/.venvs/minigpt-cn；本地RTX4060 Laptop 8GB、约31GB内存。
用户执行所有下载/安装/Git写操作；不改作业仓库、学习笔记、旧权重与数据。
缺少真实输入或依赖时只报告具体阻塞，不重启整个数据平台建设。
接口按需查 [运行参考](runtime-reference.md)、[数据参考](data-reference.md)；
历史数字见 [实验结论](experiment-log.md)，不会变成新的任务指令。

## 已有暂存改动与用户Git操作

本次Git收尾开始时分支m2/tokenizer-data、HEAD=cf1c7ea，包含以前已暂存的源码/测试/文档。
用户2026-10-13明确授权“提交一下M2，完成GitHub的M2”，本次授权允许助手提交/推送该分支，
通过PR与CI完成合并，并关闭M2里程碑；其他会话仍按AGENTS.md的默认Git约定执行。
GitHub M2是第3号里程碑，M3正式预训练仍未完成；不因已有500步资源结果关闭M3。
提交包含M2实现/编码测试、已冻结的数据工具、活动文档收尾及33M/云端验证配置和结果说明。
tokenizer、语料、checkpoint、日志、历史备份仍Git忽略；那份无关未跟踪文件原样保留。
不切换/直接改写本地main，不清空暂存区、不强推、不删除分支；实际远端完成状态以PR及里程碑为准。

本次Git提交前重新运行完整CPU回归：Windows724通过/10跳过（28.77秒），
WSL729通过/5跳过（73.95秒），两边Ruff通过；不是复述下面的历史702/707结果。
原始检查记录保存到Git忽略的logs/m2_git_closeout_20261013。

旧教程及两份误建终端输出已备份到logs/docs_closeout_20261012_v1，Git忽略、不提交。
原始下载/中断文件/旧pilot/学习笔记未删除。以后只读活动文档，不读旧备份继续任务。
文档收尾轮历史完整CPU检查：Windows702通过/10跳过，WSL707通过/5跳过，两边Ruff通过；
活动文档28个本地链接均有效。跳过项为CUDA及Windows符号链接权限，不是数据训练验收。
本次编码轮实际定向检查：Windows53通过/1跳过，WSL54通过，两边Ruff通过；
未重跑完整回归。新token数据独立全量读回验收PASS，详细结果在新token目录acceptance.json。
本次33M轮实际定向CPU检查：Windows34通过/2跳过、WSL34通过/2跳过，两边Ruff通过，未重跑完整回归。
真实33M三段GPU运行及独立重载完整验证均通过，详情在checkpoints/m3_33m_short_v1/acceptance.json。
