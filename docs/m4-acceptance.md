# M4展示交付与用户验收

状态：展示材料已完成，等待用户收尾验收和Git发布。此处验收展示内容、复现入口与证据一致性；生成质量不足继续作为实验结论保留。

## 从这里开始

1. 打开[项目展示](showcase.md)：查看真实曲线，展开全部12条固定续写，查看有界续训未改善的结论。
2. 打开[模型卡](../MODEL_CARD.md)：核对模型身份、训练规模、评估指标、适用范围与发布边界。
3. 打开[本地复现](local-reproduction.md)：确认本地同版产物、CPU最小命令及结果读回。CUDA长续写为可选，不是验收必须重跑的项目。

图片可直接打开：[PNG](assets/training-loss.png) / [SVG](assets/training-loss.svg)。这套展示材料可在GitHub浏览，无需部署服务或购买云实例。

## 验收清单

- [ ] README能直接进入展示页、模型卡和本地复现说明。
- [ ] 曲线显示原训练611条train记录、26次完整valid及续训6次valid；标注未平滑，并区分云端训练过程与本地独立指标。
- [ ] 展示页包含预定6个提示词×greedy/sampling全部12条续写，完整保留重复、离题、EOS与截断，不将失败结果包装成可靠能力。
- [ ] 模型卡固定原12208步final；14208步续训只作为未通过连贯性门槛的对照。实际train规模、处理tokens和曝光比例没有混淆。
- [ ] 本地复现明确需要已有准确版本的权重/tokenizer，代码新克隆不假装自带模型；不自动下载、重训或换用其他词表。
- [ ] 代码、配置、文档和派生图片可提交；原始数据、日志、tokenizer、checkpoint、缓存和无关文件不在提交清单。

M4验收通过表示项目展示和复现说明交付完成，不改变“生成连贯性未通过”的M3实验结论。没有新增模型训练、完整valid/test或解码参数搜索，也不启动独立0.6B LoRA。

## 已执行的技术检查

模型卡轮已经实际执行文档的Windows PowerShell5.1核对与CPU/FP32/greedy最小命令：12208步、8个新token，输出在logs/m4_local_reproduction_v1/cpu_smoke.json。已有文件不要覆盖；需要自己重放时在命令中换一个新--output文件名。
本轮只读回该产物，核对当前文档命令块与当时实跑版本一致，不重复生成或把短命令通过当作质量证明。四个PowerShell块已做5.1/7语法检查，CUDA长续写未执行。
本轮实际用已有Matplotlib生成并视觉检查PNG，交付SVG；核对图中源点数量、全部12条样例全文、指标、SHA、文档/章节链接、包含未跟踪文件的全文空白格式和Ruff。最终记录在logs/m4_showcase_v1/acceptance.json。

M3收尾的完整CPU回归Windows736 passed/10 skipped、WSL741 passed/5 skipped是历史结果。本轮模型源码与测试没有改变，不冒称重新跑过完整回归；绘图脚本不导入模型或执行训练。
本轮绘图使用现有本地依赖，未下载/安装，也未修改项目.venv。新代码克隆阅读已提交图文无需绘图库；复建图需要原始日志和完整可用的Matplotlib环境。

## 用户执行Git发布

用户已确认M3合并，本地当前仍为codex/m3-closeout。本轮助手没有执行Git写操作、访问远端或改写main，M4文件仍在工作区，暂存区保持为空。
通过上面的内容验收后，由用户在项目PowerShell中执行；先获取已合并main作为M4分支起点，保留工作区改动：

```powershell
Set-Location -LiteralPath 'D:\aiworkspace\transformer'
git fetch origin
git switch -c codex/m4-showcase origin/main
```

每条命令成功后再执行下一条。如果分支已存在或Git提示本地改动会被覆盖，保留现场并核对，不用reset/clean或清空暂存区解决。助手若以后代执行Windows Git HTTPS操作，第一次必须require_escalated；上面是交给用户手动执行的命令。

仅暂存以下10份展示文件：

```powershell
git add -- README.md MODEL_CARD.md docs/learning-roadmap.md docs/current-work.md docs/local-reproduction.md docs/showcase.md docs/m4-acceptance.md docs/assets/training-loss.svg docs/assets/training-loss.png scripts/build_showcase.py
git diff --cached --check
git diff --cached --stat
git diff --cached
```

确认只有这10份文件，格式检查无输出、原始数据/日志/权重和无关文件未暂存后：

```powershell
git commit -m "docs: complete M4 experiment showcase and reproduction guide"
git push -u origin codex/m4-showcase
```

创建PR至main，正文可用logs/m4_showcase_v1/pr-description.md。远端CI通过后由用户合并，核对M4对应issue/验收项，再关闭M4里程碑。没有公开权重或数据，不把展示页发布当成权重许可批准。具体远端状态以用户实际操作/核验为准。

## M4之后

本轮停在整体验收点；验收和Git收尾完成前，不自动进入其他任务。独立0.6B LoRA仍单独立项，使用其自身基座与tokenizer，不能直接套用33M模型或16K token ID。
