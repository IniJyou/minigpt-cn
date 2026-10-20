# M5 独立环境与 D 盘模型准备

日期：2026-10-17。交付为用户执行的[WSL 准备脚本](../scripts/m5_prepare_wsl.sh)，本轮助手未执行创建环境、安装或模型下载。
用户现已完成准备，后续[只读验收](m5-base-acceptance.md)75项通过；本页保留准备与网络中断后的操作说明，当前下一任务以current-work.md为准，不重复创建环境或下载。
依据[协作约定](../AGENTS.md)，这些操作由用户执行；本页不授权助手代装。原始盘点见[M5就绪报告](m5-base-readiness.md)，下一任务仍以[current-work.md](current-work.md)为准。

## 路径与范围

| 用途 | 固定位置 |
| --- | --- |
| 独立 M5 环境 | `/home/luoyu/.venvs/minigpt-cn-m5` |
| Qwen 模型 | `D:\aiworkspace\transformer\data\raw\m5_models\qwen3_0_6b_c1899de` |
| Hugging Face 缓存 | `D:\aiworkspace\transformer\data\raw\m5_hf_cache` |
| uv 下载缓存 | `D:\aiworkspace\transformer\data\raw\m5_uv_cache` |
| 准备记录 | `D:\aiworkspace\transformer\logs` |

WSL 使用 `/mnt/d/aiworkspace/transformer/…` 访问上述 D 盘路径；data/raw 和 logs 均被项目 Git 忽略。虚拟环境及安装后的依赖本体仍在 WSL Linux 文件系统，磁盘用量不全在 D 盘。
脚本在子进程启动前设置 HF_HOME，并显式绑定 Hub、Xet、assets、datasets 子缓存到 D 盘，避免继承其他缓存位置；uv 通过 UV_CACHE_DIR 指向 D 盘。环境与缓存跨文件系统，UV_LINK_MODE=copy。设置只作用于脚本及其子进程，不改 `.bashrc`、Windows 全局环境或旧缓存。[Hugging Face 环境变量](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables)、[uv 缓存设置](https://docs.astral.sh/uv/concepts/cache/)

准备组合为 Python3.12、torch2.8.0/cu126、transformers4.57.6、accelerate1.10.1；候选版本依据和限制沿用盘点报告。安装后记录实际全部依赖版本并运行 uv pip check；这不是预先解析并验证的完整 lock，也不代替 GPU/模型加载验收。
只下载 `Qwen/Qwen3-0.6B` 的 `c1899de289a04d12100db370d81485cdf75e47ca`，共9个指定文件，权重约1.50GB。使用公开匿名下载，不读取或保存登录凭据。

## 用户执行

打开 **WSL Ubuntu 终端**，运行：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_prepare_wsl.sh
```

脚本依次创建新环境、安装依赖、下载固定版本并核对文件摘要。遇错立即停止；创建时若目标环境已存在则停止，不重建或清理它。
命令使用已存在的 `/home/luoyu/.local/bin/uv`，不需要先激活环境或安装 pip；不改已有两个33M环境、作业环境、Git或shell配置，不运行模型推理或训练。

安装步骤完成后打印 `logs/m5_dependencies_*.txt`，内容为完整实际版本清单；下载结束打印 `logs/m5_model_prepared_*/receipt.json`，含全部9个文件的大小/SHA256、5个核心文件与聊天模板的预期摘要核对、实际包版本和本次缓存路径。只有这些步骤实际成功后，才有准备记录；本轮没有预生成“安装成功”或“模型已就绪”的回执。

## 中断后的继续方式

环境已由本脚本创建、依赖安装中断时，先保留终端错误，重新安装到这份独立环境：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_prepare_wsl.sh deps
```

依赖安装成功、模型下载中断时，单独继续下载：

```bash
bash /mnt/d/aiworkspace/transformer/scripts/m5_prepare_wsl.sh model
```

`deps` 和 `model` 会校验解释器的 sys.prefix 为固定 M5 虚拟环境；`model` 还核对三个顶层包版本。下载使用同一个 local_dir 和固定 revision，保留下载元数据供重试使用；不自动清缓存或切换镜像/模型。[官方 local_dir 下载说明](https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder)
如果创建环境本身失败，或脚本开始前就存在同名环境，先只读核对，不删除重建。SHA不符时也保留现场。

## 本机网络超时后的继续方式

2026-10-17用户执行全流程：独立环境创建成功，首次访问PyTorch cu126索引在3次重试后连接超时。随后只读确认该M5环境中尚无已安装发行包；不需要重建。
当次WSL没有HTTP(S)代理环境变量，直连PyTorch HEAD为200、PyPI和Hugging Face HEAD超时，说明直连测试并非全部稳定通过。本机WSL配置为Mirrored，可连接localhost的7897端口；显式经 `http://127.0.0.1:7897` 后，PyTorch索引、PyPI索引、固定revision配置及权重链接的HEAD均为200。没有下载权重、运行安装或测大文件吞吐。

在同一个 **WSL终端** 执行以下命令。代理仅对当前终端及子进程生效，7897为本机本次实测端口，使用时须保持对应代理程序运行：

```bash
export http_proxy=http://127.0.0.1:7897
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
export UV_HTTP_CONNECT_TIMEOUT=30
export UV_HTTP_TIMEOUT=120
bash /mnt/d/aiworkspace/transformer/scripts/m5_prepare_wsl.sh deps &&
bash /mnt/d/aiworkspace/transformer/scripts/m5_prepare_wsl.sh model
```

`deps`成功后才开始`model`。不再从`all`创建环境，不清理缓存或替换官方源。uv支持HTTP_PROXY/HTTPS_PROXY和上述连接/请求超时设置；Mirrored模式支持从WSL通过127.0.0.1连接宿主服务。[uv环境变量](https://docs.astral.sh/uv/reference/environment/)、[WSL网络说明](https://learn.microsoft.com/en-us/windows/wsl/networking#mirrored-mode-networking)
HEAD成功只确认当时的元数据链路可访问；实际依赖安装和模型下载仍由用户运行并以实际结果验收。

## 准备后验收

用户完成后，助手只读核对新环境、依赖、CUDA/BF16、固定文件和模板；随后交付只加载本地文件的 Qwen 运行入口及固定 token/解码协议。文件准备成功不等于模型能加载、生成质量通过或 LoRA 可训练。
规则基线、冻结提示、54块校准样本和旧实验均保持原样，Note2Lecture继续暂缓。

本轮已通过 bash -n、两段内嵌Python编译/Ruff、全仓库Ruff、git diff --check及活动文档本地链接检查；10份规则协议冻结产物摘要未变。没有执行安装/下载脚本或重跑pytest。
