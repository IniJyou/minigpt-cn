# 步骤2：项目运行入口的逐步教程

步骤1已验收。这里先准备独立环境，再分段接入运行入口；不开始正式33M训练。
下方 A～F 的命令现在都能执行。C/D/E 已完成用户演练，接下来执行 F 的缓存验收。

## A. 在 WSL 建立独立项目环境

PowerShell 和 WSL 是两套 Python 环境，不能把 Windows 的 `.venv/Scripts/python.exe`
当作 Linux Python 使用。本项目 WSL 环境放在 `/home/luoyu/.venvs/minigpt-cn`；
作业仍使用 `/home/luoyu/.venvs/cs336-basics`。

先在 PowerShell 进入 WSL：

```powershell
wsl
```

以下全部在 WSL 执行。先确认目录和分支：

```bash
cd /mnt/d/aiworkspace/transformer
pwd
git branch --show-current
git status --short
```

应为项目路径、`m1/transformer-core`。未提交迁移文件是正常的。
不要切回 main，也不要在这里操作作业仓库。

创建或复用项目环境，不清空已有目录：

```bash
MINIGPT_ENV=/home/luoyu/.venvs/minigpt-cn
if [ ! -e "$MINIGPT_ENV" ]; then
  /home/luoyu/.local/bin/uv venv --no-project --python /usr/bin/python3.12 "$MINIGPT_ENV"
fi
test -x "$MINIGPT_ENV/bin/python"
```

最后一条若失败，停止并把报错发给助手，不使用 `--clear`，不删除环境。
`--no-project` 避免 uv 发现别的项目；这里只创建指定的 Linux 环境。

固定到本次已验证的 WSL PyTorch 版本，不同时升级模型框架：

```bash
/home/luoyu/.local/bin/uv pip install --python "$MINIGPT_ENV/bin/python" \
  "torch==2.11.0" --default-index https://download.pytorch.org/whl/cu130
/home/luoyu/.local/bin/uv pip install --python "$MINIGPT_ENV/bin/python" \
  -e ".[dev,migration]" "torch==2.11.0"
```

CUDA 13.0 的版本组合来自 [PyTorch 官方历史版本安装说明](https://pytorch.org/get-started/previous-versions/)。
不需要 torchvision 或 torchaudio。下载较大；任一命令报错时停止，不继续运行后面的检查。
`dev` 安装测试工具，`migration` 安装参考模型所需 einops；生产模型不用 einops。
`-e` 为 editable 安装，之后源码修改无需每次重新安装。

激活并检查：

```bash
export UV_PROJECT_ENVIRONMENT="$MINIGPT_ENV"
source "$MINIGPT_ENV/bin/activate"
python -c 'import sys,torch,minigpt_cn; print(sys.executable); print(minigpt_cn.__file__); print(torch.__version__); print("CUDA:",torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU")'
```

预期解释器在 `minigpt-cn/bin/python`，包路径在本项目 `src/minigpt_cn`，
torch 为 `2.11.0+cu130`，CUDA=True，设备为 RTX 4060 Laptop。
若显示作业环境、CPU-only torch 或 CUDA=False，先排查，不能开始 GPU 训练。
激活只对当前终端有效；每次打开新终端重新设置项目的环境变量和 `source`。
回到作业时则改回作业环境，不混用。

## B. 独立环境重新验收步骤1

```bash
cd /mnt/d/aiworkspace/transformer
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.evaluation.verify_pilot \
  --assignment-root /mnt/d/aiworkspace/cs336-assignment1-basics \
  --device cuda --precision bfloat16 \
  --output logs/pilot-migration-cuda-bf16.json
```

应全部通过，报告 `status=passed`、checkpoint_iteration=5000、targets=37826、
max_logits_abs_error=0、loss_abs_error=0。
CUDA/BF16 的 project_loss 应为约3.8759773；参数量为3,737,856。
不要用 FP32 的结果强行对齐 BF16 最后几位小数或 greedy 文本。

A、B 完成后，将环境检查、pytest/Ruff 和验收报告中的 validation 段发给助手。
随后按下面四个小阶段一起实现，而不是你手工复制一份新的训练系统。

## C. 统一配置与 checkpoint（2026-10-10 已实现）

这段新增 `TrainingConfig`、`DataConfig`、`RunConfig` 和版本1 checkpoint。
模型代码不变；`configs/pilot_run.yaml` 是下一段的100步短运行配置，
不是原5000步 pilot 的历史训练配置。它引用 `pilot_3m.yaml`，不复制模型结构。

### C1. 激活项目环境并复跑测试

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
```

截至C阶段有 CUDA 时为81 passed；D接入后为95 passed；E接入后为146 passed。
F接入后有CUDA为181 passed；当前CPU-only CI 为176 passed、5 skipped。
新增22项测试包括 CPU/FP32 和 CUDA/BF16 的随机采样、dropout 及下一步参数完全一致。
该恢复机制测试暂用 PyTorch AdamW 作独立测试对象，作业 AdamW 和训练循环在D阶段迁移。

### C2. 显式导入原 pilot（不重训、不覆盖原文件）

```bash
python -m minigpt_cn.training.checkpoint import-legacy \
  --source /mnt/d/aiworkspace/cs336-assignment1-basics/out/pilot_3m_bpe2048/ckpt_final.pt \
  --model-config configs/pilot_3m.yaml \
  --tokenizer /mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/tokenizer_2048.json \
  --output checkpoints/pilot_3m_v1.pt
```

预期提示“已保存”，原作业文件不变。目标已存在时拒绝覆盖；不要删除原文件，
直接用下一条 inspect 查看现有目标。助手的实测文件另存于
`checkpoints/c_verification/pilot_3m_v1.pt`，不会占用上述用户演练文件名。

旧 pilot 没有历史 RNG/采样状态和训练配置，所以这里只导入推理权重，
不带旧优化器，也不用当前 RNG 冒充原第5000步状态。原优化器仍在源文件中保留。
后续若以它初始化新训练，属于 warm start，不是精确续训。

### C3. 独立加载、核对实际配置

```bash
python -m minigpt_cn.training.checkpoint inspect checkpoints/pilot_3m_v1.pt
```

预期：format_version=1、iteration=5000、vocab_size=2048、
unique_parameters=3737856、shared_parameter=true、resume_supported=false。
最后一项为false是旧格式的真实限制，不是导入失败。
打印的 model_config 来自 checkpoint，不必再提供 D/layers/heads。

tokenizer 校验和被保存；路径相对于 checkpoint 目录记录。若迁移文件位置，
可以给 inspect 显式传 `--tokenizer <同一份tokenizer路径>`，但不能换词表或跳过校验。
只加载可信来源 checkpoint，即使启用了 weights_only=True 也不随意加载陌生文件。

### C4. 已验收结果和原理

- 新推理格式在 CUDA/BF16 下覆盖全部37,826个目标，logits/loss 与导入前差异均为0。
- loss=3.8759773192，PPL=48.2298111859，原文件 SHA-256 不变。
- 训练保存记录优化器实际参数组；加载会恢复旧组，并返回真正生效的设置。
- `iteration` 是已完成更新数；拒绝把同一个 checkpoint 的步数写小，默认也不覆盖已有文件。
- RNG 在模型和优化器加载之后恢复，因为模型初始化本身会消耗随机数。
- 数据采样使用独立 CPU `torch.Generator`，保存其 state 后下一批才能一致。
- 精确重放结论限于相同环境与相同数据；不承诺跨 PyTorch 版本、硬件的逐bit一致。

权重只描述当前模型；下一步还取决于优化器的一/二阶矩、更新计数、dropout 随机数
和采样位置。所以“只存权重”不等于“能够精确续训”。
本段 checkpoint 库已经具备这些恢复能力，实际训练 CLI 在D阶段接入。

原始验收报告为 `logs/checkpoint-c-verification.json`；报告、checkpoint 均被 Git 忽略。

### 后续训练 checkpoint 的约定

以上C1～C3完成后，检查输出配置和测试结果，再进入D，不启动正式33M训练。

约定：

- 训练、评估、生成都从同一份模型配置构建；新 checkpoint 自带配置和格式版本。
- 保存模型、优化器、真实完成步数、随机状态及数据采样发生器状态。
- tokenizer 路径/校验和与实际词表大小必须一致；恢复时拒绝结构配置不匹配。
- 恢复优化器后记录真正生效的参数组配置；允许修改的学习率/调度项须明确记录。
- 旧 pilot 没有配置，导入必须显式提供 `configs/pilot_3m.yaml` 和原 tokenizer。
- 新输出存入项目忽略目录，绝不覆盖作业的 ckpt_final.pt。

验收：保存重载的 logits 相同、共享关系不变；错误配置拒绝；恢复步数不倒退。

## D. 接入训练与完整评估入口（2026-10-10 已实现）

### D0. 本次代码放在哪里

- `src/minigpt_cn/training/optim.py`：从你原来的 adamw.py / utils.py 接入 AdamW、
  cosine schedule、全局梯度裁剪；保留关键公式、更新顺序和中文注释。
- `src/minigpt_cn/data/tokens.py`：原始小端 uint16 检查、memmap、原 get_batch。
  新增独立 CPU Generator，checkpoint 保存它的状态。
- `src/minigpt_cn/training/train.py`：统一配置的短训练入口，记录本地 JSONL、
  完整验证、周期保存、恢复及数值检查。不接入 W&B、不上传、不下载数据。
- `src/minigpt_cn/evaluation/evaluate.py`：从 checkpoint 配置重建模型，
  复用原 eval_pilot.py 的逐块完整验证，包括尾块。

只修改项目，作业源码和学习笔记不变。生产入口不 import 作业包。
本次训练是从随机权重开始的新100步实验，不是原5000步 pilot 的续训或质量提升实验。

### D1. 激活并复跑检查（WSL）

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
```

截至D阶段有 GPU 为95 passed；E接入后为146 passed，CPU-only 为143 passed、3 skipped。
D新增14个测试用例覆盖
固定 batch 过拟合、CPU/CUDA 恢复、完整尾块验证及 loss/梯度/参数/优化器异常退出。

### D2. 用独立评估入口复现原 pilot

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.evaluation.evaluate \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --data /mnt/d/aiworkspace/cs336-assignment1-basics/data/processed/valid.bin \
  --device cuda --precision bfloat16 \
  --output logs/d_user_pilot_eval.json
```

预期 iteration=5000、targets=37826、loss=3.8759773192、ppl=48.2298111859。
这里没有提供模型层数/头数：它们直接来自 checkpoint。
报告输出已存在时拒绝覆盖；可以换一个新文件名，或不传 `--output` 只看终端。

### D3. 新训练50步后正常保存退出

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.training.train \
  --config configs/pilot_run.yaml \
  --out-dir checkpoints/d_user_first50 \
  --stop-after 50
```

不要给这条命令加旧 pilot 的 `--resume`。配置仍是 max_steps=100，
`--stop-after 50` 只控制本次在完成50次更新后保存退出，不改变100步余弦调度。
这比修改配置的 max_steps 来模拟中断更清楚。

目录应出现 `run.json`、`metrics.jsonl`、`step_000050.pt` 和 `ckpt_final.pt`。
`run.json` 记录实际配置、数据校验和和优化器设置；JSONL 记录当前进程/累计耗时。
最后 status=stopped_at_boundary、iteration=50。实测完整验证 loss 约6.7381。

### D4. 恢复到100步，并检查新 checkpoint

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.training.train \
  --config configs/pilot_run.yaml \
  --resume checkpoints/d_user_first50/ckpt_final.pt \
  --out-dir checkpoints/d_user_resumed100

python -m minigpt_cn.training.checkpoint inspect \
  checkpoints/d_user_resumed100/ckpt_final.pt
```

恢复的第一行应显示“从已完成50步恢复”，随后进行第51～100次更新。
最后 status=completed、iteration=100；完整验证 loss 约6.6605。
inspect 应有非空 training_config/data_config/optimizer_settings/run_state，
并且 resume_supported=true、shared_parameter=true。

每个运行片段必须使用新目录；这些演练目录名没有被助手使用。
若目录已存在，不删除原产物，换一个新目录名。
续训只允许覆盖 `--max-steps` 训练终点，其他训练/模型/数据配置必须相同。
`lr_decay_steps` 独立于终点，延长训练不会隐式重画余弦曲线。
训练/验证/tokenizer 内容都有校验和；不能在同路径改文件后冒充同一实验。
达到或超过目标时明确退出，不创建新目录、不把 iteration 写小。

注意：新实验只训练100步，loss 比旧5000步 pilot 高是正常现象；原 pilot 文件未改变。
它不是正式33M训练，也不能据此判断长期生成质量。

### D5. 如需自己复核连续训练与恢复一致

另跑一次不暂停的100步：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.training.train \
  --config configs/pilot_run.yaml \
  --out-dir checkpoints/d_user_continuous100

python - <<'PY'
import torch

a = torch.load('checkpoints/d_user_continuous100/ckpt_final.pt', map_location='cpu', weights_only=True)
b = torch.load('checkpoints/d_user_resumed100/ckpt_final.pt', map_location='cpu', weights_only=True)

def same(x, y):
    if torch.is_tensor(x):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            same(x[key], y[key])
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y)
        for first, second in zip(x, y):
            same(first, second)
    else:
        assert x == y

assert a['iteration'] == b['iteration'] == 100
for key in ('model_state_dict', 'optimizer', 'rng_state', 'sampler_state'):
    same(a[key], b[key])
print('PASS：第100步的权重、AdamW状态、RNG和采样状态完全一致')
PY
```

比较时不要求耗时、父 checkpoint 路径或训练片段日志完全一样。
精确结论限于相同硬件、PyTorch/精度/运行条件；不承诺跨环境逐bit相同。
助手已用独立 `checkpoints/d_verification/` 目录实测这个100步对照，四类状态均完全相同，
恢复后所有记录的训练 loss / lr 也相同。原 checkpoint 校验和未变。

### D6. 简短复习与提交点

训练形状为 `X,Y:[B,S] -> logits:[B,S,V] -> loss:标量`，Y由采样函数右移一次，
模型不能再右移。验证平均值为 `sum(loss_i * targets_i)/sum(targets_i)`，
不是对长短块简单取均值。AdamW 的 m/v 与参数同形状并保持FP32；
全局裁剪是 `g <- g * min(1, max_norm/(||g||+1e-6))`。
优化器保留作业的 epsilon 位置，不宣称与原生 AdamW 在所有 epsilon 下数值相同。
恢复需要权重、矩估计/更新计数、dropout RNG 和采样 RNG 全部对上。

累计耗时从 checkpoint 继续计数，仅覆盖训练/验证/循环内保存，
不包含进程启动、读数据/构建模型、暂停期间及最终保存写盘的尾部时间。
它不是完整租机费用，也不是正式配置吞吐基准。尚未接入 W&B 或 best checkpoint 选择。

D1～D4完成后贴最终日志和 inspect；D5可以自行再复核，然后再进入E。
这是适合保存当前迁移成果的提交点。在项目 PowerShell 由用户执行：

```powershell
git add README.md pyproject.toml configs docs src tests
git diff --cached --stat
git commit -m "feat: integrate pilot model and reproducible training entries"
```

`git add` 暂存源码、配置、测试和文档；`git diff` 检查暂存范围；`git commit` 保存本地快照。
上述命令也包含此前尚未提交的迁移/C阶段成果。不要加 `-f`，不暂存数据、checkpoint、日志。
助手未执行这些命令，不执行 push/PR/merge。

## E. 项目生成 CLI（2026-10-11 已实现）

### E0. 代码与中文注释

代码在 `src/minigpt_cn/inference/generate.py`，测试在 `tests/test_generate.py`。
生成逻辑来源于你的作业 generate.py，保留温度、top-p 等号边界、上下文裁剪和EOS停止；
关键数学、张量形状、FP32/BF16区别与“项目适配”均有中文注释。

模型现在返回 ModelOutput，所以读取 `.logits`；层数/头数/词表直接从 checkpoint 加载，
tokenizer 也按 checkpoint 中的路径/校验和加载，不需要手写一套结构参数。
新加 greedy、独立采样 Generator、异常检查、原模式恢复和 JSON 样例报告。
CLI不 import 作业包，不下载模型、不改 checkpoint、不继续训练。

### E1. 激活和检查（WSL）

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
```

有CUDA为146 passed；CPU-only为143 passed、3 skipped。E新增51个用例覆盖
原作业边界、greedy、seed、CLI、共享配置加载、UTF-8报告和CUDA/BF16参数保持FP32。

### E2. 先用旧 pilot 做 greedy 续写

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.generate \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --prompt '数学是一门' \
  --greedy --max-new-tokens 64 \
  --device cuda --precision bfloat16 \
  --output logs/e_user_pilot_greedy.json
```

应显示 iteration=5000、模式=greedy、缓存=关闭，并打印续写。
greedy每次选最高分ID，不消耗采样随机数，也不应用temperature/top-p。
实测包含“他的数学家”等重复片段；这里只验收能正确生成，不承诺语言质量。

### E3. 再试温度与 top-p 采样

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.generate \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --prompt '数学是一门' \
  --max-new-tokens 128 --temperature 0.8 --top-p 0.9 --seed 42 \
  --device cuda --precision float32 \
  --output logs/e_user_pilot_sampled.json
```

这里刻意使用FP32，与原 generate_pilot.py 前向模式一致。
固定seed仅保证相同环境/模型/精度/解码参数下可复现；不要期待FP32与BF16样例逐token相同。
CUDA/BF16也支持采样，但概率统一在FP32计算，这是明确的项目适配。
同一份旧pilot的三条prompt：FP32采样64token、greedy32token均与原作业参考逐ID一致。

本次128token样例出现百科标题、英文碎片与一个 `�`，CLI会打印UTF-8警告。
它不是运行失败，也不是被修饰过的“优秀样例”；原始ID和失败情况全部保留。

### E4. 检查项目100步模型也能独立生成

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.generate \
  --checkpoint checkpoints/d_user_resumed100/ckpt_final.pt \
  --prompt '数学是一门' \
  --greedy --max-new-tokens 64 \
  --device cuda --precision bfloat16 \
  --output logs/e_user_short100_greedy.json
```

应显示iteration=100。实测主要输出逗号：100步新训练太短，不能视作生成接口错误。
旧pilot和100步新模型是不同训练阶段，本段不试图证明训练效果已经足够。

没有GPU时，把设备/精度改为 `--device cpu --precision float32`，模型参数仍为FP32。
要换prompt，改 `--prompt`，并为 `--output` 换一个新文件名。
文件存在时拒绝覆盖；不传 `--output` 只打印，不保存样例。

### E5. JSON 报告和数学复习

JSON记录checkpoint哈希/iteration/模型配置、设备/精度/seed、解码参数、prompt、
完整/新增IDs、完整/续写文本、新增token数和停止原因。默认EOS为特殊token
`<|endoftext|>`，新增EOS先写入输出再停止；需要固定生成长度时可加 `--no-eos`。
可以用 `--eos-token` 指定词表中其他特殊token，但不能替换tokenizer或绕过校验。
移动tokenizer后可显式传 `--tokenizer <同一份文件>`；校验和仍必须一致。

温度公式为 `P(i)=exp(z_i/T)/sum_j exp(z_j/T)`，T>0；温度分母没有额外epsilon。
top-p先排序，再保留累计概率首次达到/超过阈值的最小集合，至少一项；
`[0.5,0.25,0.25]` 在p=0.5时保留第一项，p=0.6时保留前两项。
输入 `[1,S] -> logits:[1,S,V] -> 最后位置:[1,V] -> next_id:[1,1]`。
完整ID输出保留所有prompt；模型每次只看最后context_length个ID，窗口位置重新从0开始。

token可能只有半个UTF-8字符，所以先拼接全部字节，再decode。
`full_utf8_invalid` / `continuation_utf8_invalid` 标注真正的非法/不完整字节，
`replacement_character_count` 记录续写中的U+FFFD数量。合法文本也可能含真正的U+FFFD，
因此不能仅凭这个数量断言非法字节。报告不删字符、不隐藏EOS、不美化样例。

E完成后贴三条命令的输出摘要；再进入F实现KV Cache，这时还不会有缓存加速。
实测产物在助手专用 `logs/e_verification/` 目录，上述用户演练文件名未占用。
本段不修改模型/训练数学，不重新下载或编码语料。

这是合适的提交点。在项目PowerShell由用户执行：

```powershell
git add README.md docs/step2-guide.md docs/experiment-log.md src/minigpt_cn/inference tests/test_generate.py
git diff --cached --stat
git commit -m "feat: add commented Chinese generation CLI"
```

第一条暂存E源码/测试/文档；第二条核对范围；第三条保存本地提交。
如果之前迁移/C/D尚未commit，它们仍在暂存区，应一并核对；不要以为这是只有E的小提交。
样例报告被logs/忽略，数据/checkpoint不提交。助手不执行Git写操作。

## F. KV Cache 与步骤2最终验收（2026-10-11 已实现）

### F0. 代码位置与数学复习

不是重新搬一份作业模型，而是在已验收的项目实现中加可选缓存路径，原中文注释保留。

| 文件 | 本段改动 |
| --- | --- |
| `src/minigpt_cn/model/attention.py` | 每层 K/V 校验、拼接、带历史长度的因果 mask |
| `src/minigpt_cn/model/transformer.py` | Block 传递本层缓存，MiniGPT 汇总所有层的缓存 |
| `src/minigpt_cn/inference/generate.py` | prefill、单 token decode、窗口滑动时重建；增加 `--use-cache` |
| `src/minigpt_cn/inference/benchmark.py` | 同一权重的 greedy 对照、预热、多次计时与显存报告 |
| `tests/test_kv_cache.py` | 分块 logits、offset、错误缓存、跨窗口与速度报告回归测试 |

公开模型接口没有改变：

```python
model.eval()
with torch.no_grad():
    first = model(prompt_ids, use_cache=True)                  # prefill
    next_id = first.logits[:, -1, :].argmax(-1, keepdim=True)
    next_output = model(next_id, use_cache=True, kv_cache=first.cache)
```

此示例要求 prompt 长度小于 context_length；满窗口处理由生成入口完成。
返回仍为 `ModelOutput(logits, loss, cache)`。增量输入 `[B,new]` 返回 `[B,new,V]` 的 logits，
不是包含历史位置的完整 logits。cache 是按层排列的 tuple，每层 `(K,V)` 形状为
`[B,H,past+new,Dh]`，K 已做 RoPE，V 未旋转；它不是新的模型参数，也不进入 checkpoint。
旧 pilot 权重无需转换或重新训练，embedding 与 LM head 仍共享同一个 Parameter。

公式仍为 `softmax(Q_new @ K_all.T / sqrt(Dh) + mask) @ V_all`：
`Q_new:[B,H,new,Dh]`，`K_all,V_all:[B,H,past+new,Dh]`，
scores 为 `[B,H,new,past+new]`。query位置为 `past+i`，只允许 key位置 `j <= past+i`。
因此不能对这个非方形 mask 直接使用从0开始的下三角。新 Q/K 的 RoPE offset=past；
旧 K 不能重复旋转。FFN 和残差只计算新位置，因果性保证未裁剪前的历史表示不变。

缓存仅用于 eval 推理，生成入口自动 eval/no_grad 并恢复原模式；模型会拒绝 training 下的缓存。
缓存由调用者持有，不能跨 prompt、权重、设备或前向精度混用；旧缓存不会被原地修改。
最初使用 torch.cat 拼接 K/V，尚未做预分配或内核优化。

窗口策略为 `rebuild_on_slide`：未满时只输入新 token；满后用最新完整窗口重建全部层，
RoPE 从0重新开始。深层历史表示仍可能包含已移除 token 的信息，所以只删旧 K/V 不等价。
这保持 E 阶段的无缓存窗口语义；代价是满窗口后每一步重建，没有持续的增量加速。

### F1. 激活并检查（WSL）

```bash
cd /mnt/d/aiworkspace/transformer
export UV_PROJECT_ENVIRONMENT=/home/luoyu/.venvs/minigpt-cn
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
python -m ruff check .
```

本次有CUDA为181 passed；只用CPU为176 passed、5 skipped。测试不依赖真实语料。
如需明确复核 CPU 路径：

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

最后一条仅对这个进程隐藏GPU，不改变你的CUDA安装或环境。

### F2. 开缓存生成，并与自己的 E 报告逐 ID 比较

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.generate \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --prompt '数学是一门' \
  --greedy --max-new-tokens 64 --use-cache \
  --device cuda --precision bfloat16 \
  --output logs/f_user_pilot_cached.json

python - <<'PY'
import json
from pathlib import Path

old = json.loads(Path('logs/e_user_pilot_greedy.json').read_text(encoding='utf-8'))
new = json.loads(Path('logs/f_user_pilot_cached.json').read_text(encoding='utf-8'))
for key in ('checkpoint_sha256', 'prompt_ids', 'precision', 'generated_tokens'):
    assert old[key] == new[key], key
assert not old['use_cache'] and new['use_cache']
assert old['full_ids'] == new['full_ids']
print('PASS：与E阶段无缓存结果逐token完全一致')
PY
```

应显示 iteration=5000、缓存开启，并且仍出现之前的重复续写。这是预期：缓存不改变语言质量。
比较同一权重、精度、prompt 和 greedy 参数；不能拿 BF16 缓存结果与 FP32 无缓存结果比较。
若没有 E 报告，先运行 E2 保存它；已存在则不重写。输出文件存在时换新名字并同步改比较脚本。

### F3. 测一致性、速度与显存

测速时暂停其他GPU任务，保持供电/运行环境一致，不同时运行pytest。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.benchmark \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --prompt '数学是一门' --max-new-tokens 64 \
  --device cuda --precision bfloat16 \
  --warmup 2 --repeats 5 \
  --output logs/f_user_benchmark.json
```

预期 `status=passed`、`greedy_ids_equal=true`，cached 的 incremental_calls=63、window_rebuilds=0。
基准固定新增64个 token、不用EOS，所有正式测量的完整ID必须相同，否则明确报错。
计时前后同步CUDA：first_token_seconds 是 prefill 到首个ID生成的时间；
decode_tokens_per_second 排除首token；generation_seconds 是全64token生成时间。
这些值不包含加载权重、prompt编码、文本解码。速度/延迟为多次中位数，trials保留每次原始结果。
peak_allocated_bytes 是测量阶段PyTorch分配的CUDA显存峰值（多次取最大），
不是nvidia-smi整个进程占用；kv_cache_bytes 只统计各层持久K/V，CPU显存字段为null。
小pilot/短prompt可能主要受Python、小算子和校验开销影响，不要求某个加速倍数。

可选复核：超长prompt使每次都重建窗口，报告应 incremental_calls=0、window_rebuilds=7：

```bash
MINIGPT_LONG_PROMPT=$(python -c "print('数学是一门' * 100)")
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m minigpt_cn.inference.benchmark \
  --checkpoint checkpoints/pilot_3m_v1.pt \
  --prompt "$MINIGPT_LONG_PROMPT" --max-new-tokens 8 \
  --device cuda --precision bfloat16 --warmup 1 --repeats 3 \
  --output logs/f_user_full_window.json
```

300个prompt token超过pilot的256窗口；完整输出仍保留prompt，但每次模型只看最新256个。
此时缓存模式也重算整个窗口，不能期待增量加速。测试已覆盖满窗口前后、超长输入与多token追加。

### F4. 验收与提交点

将 F1 的测试摘要、F2 的 PASS、F3 的 status/速度/显存摘要发给助手。
已知生成质量问题照实保留，不需要因文本重复重训或改变输出。
这时步骤2实现已齐备；完成你的本机复核后，再讨论步骤3的正式16K tokenizer、
33M/512配置约500步资源试跑与训练预算，不直接启动正式预训练。

在项目PowerShell由用户执行：

```powershell
git status --short
git add README.md docs/step2-guide.md docs/experiment-log.md docs/pilot-migration.md src/minigpt_cn/model src/minigpt_cn/inference src/minigpt_cn/evaluation/verify_pilot.py tests/test_model.py tests/test_generate.py tests/test_kv_cache.py
git diff --cached --stat
git commit -m "feat: add KV cache and sliding-window generation checks"
```

status先看范围，add暂存源码/测试/文档，diff核对全部暂存项，commit保存本地快照。
如果之前迁移/C/D/E未commit，它们也在这次暂存范围中；助手没有清空或修改你的暂存区。
不加-f、不提交数据/checkpoint/logs；push/PR/merge仍由用户执行。
