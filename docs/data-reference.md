# 数据工具参考（扩建冻结）

这些是已验证、可复用的接口，不是要求继续完成的步骤列表。
当前任务见 [current-work.md](current-work.md)；16K保存和pilot编码已验收，
操作见 [Token编码验收](token-encoding-guide.md)，输入/产物状态见 [data/README.md](../data/README.md)。
本页不提供全站扫描、下载或批量审批操作命令；当前不要开展新模板/章节/审批系统开发。

## pilot与模型闭环可复用工具

| 模块 | 用途与边界 |
| --- | --- |
| cleaning.py | NFC、可选OpenCC t2s、统一换行/逐行strip；按码点过滤，整篇超限不截断 |
| splitting.py | 已清洗全文精确去重；SHA-256正文键，seed42，桶98/1/1，重排不改归属 |
| prepare.py | 已验收正文JSONL→固定三集合/manifest；保留首次来源，拒绝空集合，不处理XML |
| export_text.py | 只读train并核对来源/归属/摘要；正文+EOS+LF；新目录输出文本与导出清单 |
| sampling.py | 独立哈希域的完整文档前缀采样；预算含EOS/LF，不截断、不跳选大文档 |
| encode.py | 已验收pilot三集合按篇编码；绑定tokenizer摘要、EOS/LF、无头小端uint16；不覆盖 |
| tokens.py | 无头小端uint16数据的memory map/get_batch；新16K产物已通过完整读回 |

prepare不是绕过许可/质量的入口，不将quarantine/未完成提取或原始wikitext伪装为已验收正文。
BPE训练使用train纯文本；模型训练使用同版本tokenizer编码的整数ID，二者文件不可混用。
原pilot正文级划分不重写，后续作品/章节语料按共同work_id避免跨集合泄漏。

## Wikimedia工具保留，但不占当前主线

| 模块 | 已实现与限制 |
| --- | --- |
| download.py | 固定版本/来源快照、完整摘要、显式续传/有限重试；下载仍只能由用户执行 |
| parse_xml.py | 有页面/解压预算的流式原始wikitext；不写全解压XML，不产训练正文 |
| wikitext.py / extract.py | AST白名单提取、来源/修订/摘要/审阅绑定；v2显示字段/空节/局部转换 |
| candidate_subset.py | 复用中断候选、32MiB完整文档哈希前缀；声明中断范围，不假冒全量完成 |
| quality.py | 自动eligible/held/excluded与分层抽检；不做语义保证或自动许可批准 |
| corpus.py | 只接完整提取清单中的approved；SQLite精确去重/全来源保留、work_id划分及泄漏复核 |

v2保留已知显示参数和Zy基字，移除无正文的空节；可见math、行内未知模板、缺显示字段、
空括号/重复逗号或未支持的转换明确拒绝，rejected管线不输出text。
不执行Lua/跨页转录/完整MediaWiki渲染；未知孤立模板仍需审阅，不能保证没有所有缺词。
表格/公式覆盖有限，保守启发式可能排除正常注释；修代码不恢复旧文本，旧审批摘要不可套用。
机器过筛不是人工批准，不放宽现有审批门槛，也不开发批次审批平台。

两个划分模式不可混用：pilot以正文键为准；corpus以确认work_id为准。
corpus遇到同正文来源跨集合时排除整个正文，不挪作品；不宣称检测全部近似转载。
数据工具的成功manifest仅代表对应阶段成功，不自动意味着ready_for_training或正式语料完成。

## 按需查看参数与检查

各CLI用python -m minigpt_cn.data.<模块> --help查看，不把帮助页当自动运行授权。
下载/安装、扩大页面/字节预算、正式数据操作均需用户授权；输出不覆盖，数据目录不提交。
来源文件/版本不一致时报具体错误，禁止补写成功清单、改状态字符串或放宽门槛来“通过”。
正式数据扩充另定预算后复用这些工具，不以两站全量解析为前置条件。
