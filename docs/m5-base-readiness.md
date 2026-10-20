# M5 独立环境与Qwen基座就绪盘点

日期：2026-10-16。结论：**GPU侧检测通过，独立M5环境和模型尚未准备，当前不能运行Qwen任务基线。**
本轮只读取环境/缓存信息和公开发行元数据，保存此报告及盘点记录；没有安装、下载模型、加载权重、生成或训练。项目入口仍为[current-work.md](current-work.md)。

**2026-10-17后续状态：用户已完成准备，[独立环境与本地模型只读验收](m5-base-acceptance.md)75项通过。下文保留10月16日盘点当时的缺项，不作为当前未安装/未下载的判断。仍未加载权重或生成。**

## 1. 本地实际状态

| 项目 | Windows | WSL Ubuntu |
| --- | --- | --- |
| 已有项目环境 | `D:\aiworkspace\transformer\.venv` | `/home/luoyu/.venvs/minigpt-cn` |
| Python | 3.12.4 | 3.12.3 |
| Torch（已有33M环境） | 2.14.0+cu126 | 2.11.0+cu130 |
| CUDA运行时 | 12.6 | 13.0 |
| CUDA可用 / BF16支持 | true / true | true / true |
| GPU | RTX4060 Laptop，8188MiB | 同一物理GPU可见 |
| 独立M5环境 | 检查的常用位置未发现 | `.venvs`中仅原项目与独立作业环境，无M5环境 |
| transformers / tokenizers / safetensors / huggingface-hub | 均未安装 | 均未安装 |
| accelerate / peft / datasets / trl / bitsandbytes | 均未安装 | 均未安装 |

显卡驱动本轮读到617.14，查询当时空闲7947MiB。只查询CUDA/BF16能力，没有做BF16矩阵运行、权重加载、峰值显存或速度实测，不能把它当LoRA显存验收。
这两份Torch版本是本次实际读数，不是本轮安装，也不是原云端精确resume环境。

Windows原始解释器`D:\python\python.exe`为3.12.4，venv/ensurepip/pip可用；uv位于`D:\python\Scripts\uv.EXE`。
WSL的`/home/luoyu/.local/bin/uv`实际存在，版本0.12.21；它不在本次非交互WSL调用的PATH中，应使用绝对路径。WSL基础Python有venv但无ensurepip，原项目环境也没有pip发行包；使用已有uv可避免先改系统Python或原环境。

WSL最初被Windows沙箱拒绝访问，随后通过已授权的只读系统调用完成盘点；没有读取令牌、密钥、笔记或作业环境内容。作业环境只在目录名列表中出现，未运行其解释器。

## 2. 本地模型检查范围

- Windows：默认`C:\Users\luoyu\.cache\huggingface`没有hub模型目录；未设置HF_HOME、HF_HUB_CACHE、HUGGINGFACE_HUB_CACHE、TRANSFORMERS_CACHE或XDG_CACHE_HOME。
- WSL：`/home/luoyu/.cache/huggingface/hub`的模型仓库数为0，上述缓存环境变量也未设置。
- 两端检查的ModelScope默认缓存、常用独立models目录不存在；检查的WindowsOllama/LM Studio目录也不存在。
- 项目data/raw、data/processed、checkpoints、logs仅查看直接条目的Qwen/M5/model名称，没有发现Qwen模型目录；旧33M权重、tokenizer和日志不当作Qwen产物。

因此是**在本轮明确检查的缓存及常用位置未发现Qwen**，不是全盘搜索后断言整机任何地方都没有。若已有自定义位置，只需按准确路径追加核对，不重下替代品。
磁盘读数：C盘约75.62GiB、D盘约368.73GiB空闲；WSL根文件系统报告约934.49GiB虚拟可用空间，它不等于承载VHD的宿主盘实际余量。

## 3. 应准备的准确模型

选定仓库仍是 **`Qwen/Qwen3-0.6B`**。项目所说“基座”指未加本项目LoRA的起点，不能自行换成`Qwen/Qwen3-0.6B-Base`或量化衍生版；官方将本仓库列为经过预训练和后训练的模型。[官方模型卡](https://huggingface.co/Qwen/Qwen3-0.6B)

本轮只读查询得到的固定revision为：

```text
c1899de289a04d12100db370d81485cdf75e47ca
```

官方文件清单含单个`model.safetensors`（1503300328字节），连同配置/tokenizer等仓库文件约1.52GB；本地一个也未下载。[固定版本文件列表](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
准备范围为model.safetensors、config.json、generation_config.json、tokenizer.json、tokenizer_config.json、merges.txt、vocab.json、README.md、LICENSE。分词器与权重必须同revision，不使用项目33M的16K token ID。

已读取的少量公开配置仅用于核对元数据：model_type=qwen3、BF16、vocab_size=151936；tokenizer_class为Qwen2Tokenizer，聊天模板嵌在tokenizer_config.json，不能因类名含Qwen2就更换分词器。模板SHA256=a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8。[固定版本配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json)、[tokenizer配置](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer_config.json)

后续须明确使用`enable_thinking=False`来符合纯JSON任务，并固定模板和完整提示；官方提供该硬开关。不能只在文字提示中要求不思考，就声称已关闭模板默认思考模式。[模型卡的非思考模式](https://huggingface.co/Qwen/Qwen3-0.6B#enable_thinkingfalse)

## 4. 还缺什么

| 缺项 | 是否阻塞原始Qwen基线 | 当前处理 |
| --- | --- | --- |
| 独立M5解释器及依赖 | 是 | 建议新建WSL `/home/luoyu/.venvs/minigpt-cn-m5`，该位置本轮确认不存在 |
| Transformers及tokenizers/safetensors/Hub/Jinja2等推理依赖 | 是 | 在新环境安装，保持两份33M环境不变 |
| 同revision的完整模型、分词器、聊天模板与SHA记录 | 是 | 下载到Git忽略的数据目录，下载后核对文件和摘要 |
| Qwen任务运行入口与模型协议 | 是 | 当前src/scripts没有AutoTokenizer/AutoModelForCausalLM调用，只有规则运行器；需后续实现 |
| 输入与输出token预算、解码设置、EOS/模板处理 | 是 | 用真实Qwen tokenizer预检完整提示加输入，冻结后运行，不静默截断 |
| PEFT、LoRA配置、训练数据及优化器 | 否，属训练阶段 | 暂不安装/实现；基座运行不能被它们无故阻塞 |
| datasets、TRL、bitsandbytes、FlashAttention、推理服务 | 当前不需要 | JSONL可直接读，先做BF16单GPU基线，不顺带扩服务或量化任务 |

## 5. 供用户后续执行的准备命令（本轮未运行）

**2026-10-17更新：用户已选择模型和缓存放D盘，当前执行入口为[M5环境与D盘准备](m5-environment-setup.md)及其WSL脚本。下面是10月16日盘点时的候选命令，保留供追查；实际准备执行新入口，缓存路径由脚本统一设置。**

按[协作约定](../AGENTS.md)，下载和安装由用户执行。本次只给出具体候选准备方案；以下命令会新建环境/下载安装，**本轮没有执行**。推荐WSL，使用已存在的uv绝对路径，无需修改原33M环境或额外安装pip。

建议的顶层版本为torch2.8.0/CUDA12.6、transformers4.57.6、accelerate1.10.1：PyTorch官方提供该CUDA wheel，后两者发行元数据支持Python3.12，Transformers版本也高于Qwen3要求的4.51.0。这只是可准备的固定候选组合，尚未在本机新环境进行依赖解析和加载验收；传递依赖需在准备后锁定，不能当作已有完整lock。[PyTorch发行命令](https://pytorch.org/get-started/previous-versions/#v280)、[Transformers发行元数据](https://pypi.org/pypi/transformers/4.57.6/json)、[Accelerate发行元数据](https://pypi.org/pypi/accelerate/1.10.1/json)、[Qwen兼容要求](https://huggingface.co/Qwen/Qwen3-0.6B#quickstart)

在WSL中执行，目标路径应仍不存在；若已经准备过同名环境，先核对，不重建：

```bash
/home/luoyu/.local/bin/uv venv --python /usr/bin/python3 /home/luoyu/.venvs/minigpt-cn-m5
/home/luoyu/.local/bin/uv pip install --python /home/luoyu/.venvs/minigpt-cn-m5/bin/python 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu126
/home/luoyu/.local/bin/uv pip install --python /home/luoyu/.venvs/minigpt-cn-m5/bin/python 'transformers==4.57.6' 'accelerate==1.10.1'
```

Jinja2由Torch依赖、tokenizers/safetensors/Hub等由上述依赖带入，准备后仍须实际import并记录解析版本；不安装torchvision/torchaudio等文本任务无关包。
随后用新解释器下载固定版本，目录位于项目Git忽略的data/raw；这段也**未执行**：

```bash
/home/luoyu/.venvs/minigpt-cn-m5/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
    local_dir="/mnt/d/aiworkspace/transformer/data/raw/m5_models/qwen3_0_6b_c1899de",
    allow_patterns=[
        "model.safetensors", "config.json", "generation_config.json",
        "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json",
        "README.md", "LICENSE",
    ],
)
PY
```

公开元数据给出的权重SHA256为f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b。下载完成后要对真实本地文件核对，并保存新环境完整版本清单；仅下载命令退出0不算模型验收。

## 6. 准备后的验收顺序

先核对新环境路径、包版本、CUDA/BF16、模型文件/摘要和模板，再实现只读本地模型的Qwen运行入口（local_files_only=True、trust_remote_code=False）。随后用已冻结[完整提示](../configs/m5_structure_prompt_v1_1.txt)测token长度，固定模型/解码协议，做小输入及54块校准基线，沿用严格评分。
Qwen首次运行的结果与规则54/54分别报告；正式未见来源评测仍未就绪，不因环境和模型就绪直接启动LoRA。
下一步是用户按具体范围准备新环境和固定模型，助手再做只读验收及基座运行入口；若另行授权助手执行具体安装/下载，才按该授权操作。本报告不扩展先前GitHub或其他任务的授权。
