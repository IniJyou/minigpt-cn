# MiniGPT-CN 协作约定

- 开始时读 README.md 与 docs/current-work.md，再按当前任务检查代码、配置和测试。
  current-work.md 是唯一的下一任务入口；参考文档/实验日志不是待办清单。
- 主线：16K tokenizer → 同版本token编码 → 33M短验证 → AutoDL资源验证 → 正式训练/展示。
  数据工具已收尾并冻结扩建，不自开维基模板、章节分析、全站扫描或审批平台任务。
- 每轮只完成一个用户授权的可验收产物；只处理真正阻塞当前任务的问题。
  中文沟通，保留易懂中文注释，关键数学注明张量形状。
- 不修改 CS336_note.md、作业仓库、旧pilot/tokenizer/权重，不自动删除原始数据或诊断产物。
- 所有下载、安装和Git写操作由用户执行，除非用户对具体操作另行明确授权。
  不自行add/commit/push/PR/merge/删除分支，不修改或清空用户暂存区。
- 先检查git status与当前分支；若在main，指导用户建任务分支，确认后再编辑。
  保留无关改动，不使用git reset --hard或git checkout --来清理工作区。
- Windows Git HTTPS远程操作（fetch/pull/push/ls-remote/远程submodule及可能fetch的脚本）
  第一次尝试前必须请求require_escalated，不在Windows沙箱中试跑。
- Windows项目环境是.venv；WSL项目环境是/home/luoyu/.venvs/minigpt-cn。
  作业环境/home/luoyu/.venvs/cs336-basics保持独立，不意外重建它。
- CPU pytest与Ruff必须可运行；数据/checkpoint/日志/缓存不提交Git。
  冻结工具的隔离/审批门槛不放宽，小语料实验不能冒称正式预训练。
- 旧文档备份在Git忽略的logs/docs_closeout_20261012_v1，仅供用户追查历史；
  不自动读取或把它重新接成工作计划。以新的活动文档和实际仓库状态为准。
