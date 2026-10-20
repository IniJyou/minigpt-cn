# M5 Qwen3-0.6B 本地基座运行协议 v1.1

日期：2026-10-18。用户同意暂用Transformers直接运行，独立环境和固定模型已通过[只读验收](m5-base-acceptance.md)。本轮交付本地入口、固定协议及首次小输入/54块校准结果；不安装、下载模型、训练LoRA或推进Note2Lecture。
本文件与源码/测试在首次运行前绑定SHA；结果写入logs和current-work，不在看到结果后修改本协议。
v1首次启动在实例化模型前被依赖检查误拦：PYTHONPATH里的项目源码egg-info被算入已安装发行包。v1.1只读取M5解释器purelib/platlib的包元数据，源码另行SHA绑定；提示、输入、采样和预算不变。原v1协议及源码/测试/文档快照与失败记录保留在logs/m5_qwen_checks_20261018_v1，没有v1模型输出或校准分数。

## 固定条件

| 项目 | 固定值 |
| --- | --- |
| 模型 | Qwen/Qwen3-0.6B，revision=c1899de289a04d12100db370d81485cdf75e47ca |
| 本地模型目录 | data/raw/m5_models/qwen3_0_6b_c1899de，9份准确文件逐份绑定SHA |
| 环境 | WSL /home/luoyu/.venvs/minigpt-cn-m5；Python3.12.3；全部实际发行包版本写入协议 |
| 推理 | 单GPU cuda:0，BF16，PyTorch SDPA，eval+inference_mode，无量化/CPU卸载/自动降级 |
| 输入 | 完整冻结系统提示 + input对象的紧凑JSON；仅blocks，不传review_id/来源/参考标注 |
| 模板 | 同revision tokenizer的apply_chat_template，add_generation_prompt=True、enable_thinking=False |
| 预算 | 完整模板输入最多4096 tokens，最多新增1024，合计最多5120；不截断、不滑窗 |
| 采样 | do_sample=True，temperature=0.7，top_p=0.8，top_k=20，min_p=0.0 |
| 其他解码 | repetition_penalty=1.0，num_beams=1，num_return_sequences=1，动态KV缓存；disable_compile=True |
| 特殊token | EOS=[151645,151643]，PAD/BOS=151643 |
| 随机性 | 每条输入重新设置CPU/CUDA种子42；TF32关闭，cudnn benchmark关闭，CUBLAS_WORKSPACE_CONFIG=:4096:8 |
| 模式 | smoke：固定合成1片段2块；calibration：原16片段54块，顺序保持JSONL原序 |

采样设置采用[Qwen官方非思考模式建议](https://huggingface.co/Qwen/Qwen3-0.6B#enable_thinkingfalse)。这是一次预先固定种子的采样，不做种子搜索、失败后补抽或选更好回复；不承诺不同CUDA/硬件逐token一致。
预算根据已验收输入1352–1940 tokens预检给出余量；它是当前实验边界，不代表全部6000字符规范输入都能放入4096 tokens，也不声称支持tokenizer标称的131072上下文。任何输入超限则在加载权重前停止整次运行。

输入采用json.dumps(input, ensure_ascii=False, separators=(",", ":"))；提示按UTF-8文本读取并统一换行。GPU输入input_ids与attention_mask形状为[1, input_tokens]，无padding；生成只保留下一位置logits以节省显存。
生成使用新建GenerationConfig，显式设置上述项，完整有效配置记入run_manifest，不继承模型文件中的thinking采样默认值。[Transformers生成配置](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation)

## 入口与离线边界

代码为[src/minigpt_cn/m5_qwen_run.py](../src/minigpt_cn/m5_qwen_run.py)，固定协议为[configs/m5_qwen_baseline_v1_1.json](../configs/m5_qwen_baseline_v1_1.json)，启动脚本为[scripts/m5_qwen_wsl.sh](../scripts/m5_qwen_wsl.sh)。
后者指定已准备的M5解释器与项目src，不激活或安装包；Python入口在导入HF前固定D盘缓存、HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1，两个from_pretrained均为本地路径、local_files_only=True、trust_remote_code=False。不启动HTTP服务或依赖Ollama。
所有模型文件必须存在且SHA一致；模型目录中出现额外文件（例如adapter_config.json）也停止，不自动导入适配器、远程模型或替代权重。

在WSL终端执行，输出必须为项目logs下尚不存在的新目录：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_qwen_wsl.sh \
  --mode smoke --output-dir /mnt/d/aiworkspace/transformer/logs/m5_qwen_smoke_replay_v1
bash /mnt/d/aiworkspace/transformer/scripts/m5_qwen_wsl.sh \
  --mode calibration --output-dir /mnt/d/aiworkspace/transformer/logs/m5_qwen_calibration_replay_v1
```

首次小输入只检查加载、生成和输出保存链路；它不是选择提示词/参数的调优集。即使出现模型格式/字段错误，仍按原协议运行校准并如实记分；基础设施失败则保留现场并说明，不能悄悄替换版本后沿用同次结果。

## 响应保留和评分

生成序列先核对原输入token前缀，随后仅取新增token，剥离最后一个已配置终止EOS。保存新增token ID、含EOS的原始解码、用于评分的响应、耗时和停止原因。除该终止EOS外，不删特殊token、空白、思考、围栏或说明，不修JSON、补块、重排、重写标题或使用规则补答案。
仅在生成结束后使用参考标注，沿用原[严格评分](m5-rule-baseline.md)：JSON/schema/编号覆盖、类别、块全部字段、整片段、标题/reason的全体及非空子集、固定八类macro/micro-F1和混淆矩阵。格式坏整条没有部分TP，不排除失败分母。
每条最多尝试一次。生成异常保留错误，空响应进入原有全部分母；不因为失败自动重试、降级或只报告成功子集。触及token上限另报，截断JSON按原有格式失败计分；若刚好在上限输出完整合法JSON，照常评分并仍记录上限标志。

输出包含protocol.json、input_tokens.jsonl、predictions.jsonl、errors.jsonl、metrics.json、report.md和run_manifest.json。逐条生成后立即写入并flush；权重加载或运行完整性失败记录failure.json。运行前后复核协议/绑定产物SHA，完整性变化不接纳该次得分。峰值显存和单请求耗时仅为本机本次实测，不是跨框架吞吐基准或LoRA显存测试。

## 校准限制与停止点

规范、系统提示、16片段54块候选标签及评分器均保持原样。样本已参与规范制定，候选由助手核定，非独立人工金标、未见来源测试或正式SFT；来源/家族/稀有类别不足继续存在。
固定规则校准54/54只是同材料参照。Qwen低分也不自动证明LoRA值得做，高分也不证明任务泛化；先分析格式错误、规范执行错误和标注缺口，再决定独立来源评测及后续训练范围。不会按本次分数回改提示、候选或解码设置来刷新同一协议。
