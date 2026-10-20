# M5 独立环境与本地模型只读验收

日期：2026-10-17。结论：**75项只读检查通过，独立M5依赖、固定模型文件和本地tokenizer已就绪；尚未加载模型权重或执行生成。**
2026-10-18后续已完成[首次本地GPU推理及校准](m5-qwen-results.md)；本页保留首次推理前的只读验收状态。
用户完成安装/下载，助手在断网模式下复核，只向本轮logs目录写验收记录及更新活动文档。下一任务入口仍为[current-work.md](current-work.md)。

## 实际通过的范围

| 检查 | 实际结果 |
| --- | --- |
| 独立解释器 | `/home/luoyu/.venvs/minigpt-cn-m5`，Python3.12.3，sys.prefix符合预期 |
| 固定顶层依赖 | torch2.8.0+cu126 / transformers4.57.6 / accelerate1.10.1 |
| 解析后的辅助依赖 | huggingface-hub0.36.2 / tokenizers0.22.2 / safetensors0.8.0 / Jinja2 3.1.6 / numpy2.5.4 |
| 包完整性与导入 | 全部发行包版本与用户下载回执一致，uv pip check通过，8个主要模块均从M5环境导入 |
| GPU能力查询 | RTX4060 Laptop，8188MiB，CUDA12.6可用，原生BF16支持，compute capability8.9 |
| 模型/缓存路径 | 模型与本次HF、uv缓存设置均在D盘；虚拟环境和安装依赖本体仍在WSL文件系统 |
| 模型来源 | Qwen/Qwen3-0.6B，固定revision=`c1899de289a04d12100db370d81485cdf75e47ca` |
| 文件核对 | 9个文件重新计算大小/SHA256，与下载回执一致；其中5个核心文件另对照前次官方元数据SHA；9份本地下载元数据revision/ETag也一致 |
| 权重文件头 | 只读取safetensors头；311个BF16张量的形状/数据偏移/大小一致，embedding和层编号符合config |
| 本地配置与分词器 | AutoConfig和Qwen3ForCausalLM类导入成功；Qwen2TokenizerFast离线加载成功，合成中文/公式/代码往返一致 |
| 聊天模板 | 与固定tokenizer_config中的模板一致，SHA与前次官方记录一致；enable_thinking=False实际改变生成前缀 |
| 冻结规则产物 | 10份规范、提示、schema、规则代码/测试、校准数据及来源manifest的SHA均未变 |

本地模型位于 `D:\aiworkspace\transformer\data\raw\m5_models\qwen3_0_6b_c1899de`；权重1503300328字节，SHA256为`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
配置仍为28层、hidden_size1024、vocab_size151936、tie_word_embeddings=true。文件头统计的存储元素数不是加载后去重的独立参数统计，本轮不创建模型来测参数或显存。

依赖清单在 `logs/m5_dependencies_RROCXytJ.txt`，下载回执在 `logs/m5_model_prepared_4u71uj49/receipt.json`；这记录实际解析版本，不是含wheel摘要/来源的完整安装lock。PEFT、TRL、bitsandbytes等尚未准备，不阻塞下一步原始基座推理。

## 完整提示的token预检

只用真实本地tokenizer对现有16片段54块做计数，无权重、前向或预测。用户输入序列化为 `json.dumps(input, ensure_ascii=False, separators=(",", ":"))`，系统提示按UTF-8文本读入（统一换行），模板设add_generation_prompt=True、enable_thinking=False。

| 项目 | 结果 |
| --- | --- |
| 最短完整输入 | 1352 tokens，n05 |
| 最长完整输入 | 1940 tokens，r05 |
| 是否截断 | 否 |
| 是否设置正式输入/输出预算 | 否，仅预检；后续模型协议另行冻结 |

tokenizer的model_max_length为131072，而模型config的max_position_embeddings为40960，二者不能直接当成本项目运行预算。此次16条完整输入都低于config上限；这不代表可使用131072上下文，也不代替实际输入加输出预算和GPU显存验收。
实际模板关闭思考模式后的前缀含空的`<think>…</think>`结构，由模板提供。后续运行器只评分输入长度之后的新生成文本，不对模板前缀做字符串拼接来伪装关闭思考。

## 证据和边界

本轮证据保存在 `logs/m5_base_acceptance_20261017_v1/`：

- `audit_readonly.py`：本次离线验收脚本，只写本验收目录。
- `inventory.json`：实际包版本、模块来源、文件摘要、配置、模板、文件头及逐片段token计数。
- `acceptance.json`：75项检查逐项通过记录。

读取时设置HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1、UV_OFFLINE=1和禁止写pyc，并使用local_files_only=True、trust_remote_code=False。缓存变量只在验收进程内绑定到D盘；新终端仍需显式设置，未修改全局代理、shell配置、旧缓存或旧环境。

没有实例化Qwen模型、将权重加载到CPU/GPU、分配CUDA张量、做矩阵/神经前向、生成或训练；没有重跑pytest。文件头核对不验证张量内容的数值性质，CUDA/BF16能力查询不等于模型能加载、推理速度或LoRA显存通过。
规则54/54结果保持为校准参照，仍无Qwen得分、独立人工金标或未见来源评测。Note2Lecture继续暂缓。

唯一下一任务是实现本地Qwen基座运行入口，冻结模型/模板/序列化、输入与输出token预算、解码/EOS等协议，再做小输入和54块校准推理，保留完整原始响应并沿用严格评分；不因本轮就绪自动进入LoRA训练。
