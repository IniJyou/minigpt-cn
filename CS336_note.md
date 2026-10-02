# CS336: Language Modeling from Scratch（从零构建语言模型）


[toc]


## Introduction（课程介绍）

资料：[课程主页](https://cs336.stanford.edu/) · [Spring 2026 第一讲](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)

### Why This Course?

- 课程理念是 **understanding via building**：从零搭建语言模型，以理解底层机制，而不只是调用现成模型。
- 前沿模型的训练成本和实现细节难以复现；用较小模型仍能学习可迁移的 **mechanics**（机制）和 **mindset**（资源与扩展性的思维方式），但小规模实验得出的效果直觉未必能直接推广到前沿规模。

### Course Roadmap

- 路线：**Basics**（分词、模型、训练）→ **Systems**（计算核、并行、推理）→ **Scaling laws** → **Data**（评估与数据处理）→ **Alignment**（对齐与强化学习）。

```python
    basics()         # Assignment 1: tokenization, model architecture, training
    systems()        # Assignment 2: kernels, parallelism, inference
    scaling_laws()   # Assignment 3: scaling laws
    data()           # Assignment 4: evaluation, curation, transformation, filtering, deduplication, mixing
    alignment()      # Assignment 5: RLHF, RL algorithms, RL systems

```

- 核心问题：在给定数据与硬件资源（算力、内存、通信带宽）时，如何训练出尽可能好的模型？课程用 `accuracy = efficiency × resources` 强调效率的重要性；这是讲义中的 **概念性口号，不是可直接计算的精确公式**。
- 历史脉络（只记主线）：N-gram → 神经语言模型与 Transformer → 预训练基础模型与规模化 → 开放模型及更复杂的应用形态。

MiniGPT-CN 对照（非讲义内容）：课程的 Basics 部分对应项目里的分词、Transformer、训练与评估。

### Personal Understanding

**核心判断：** 小模型上有效的做法，不保证在前沿规模同样有效。讲义强调：机制与效率思维较容易迁移，具体实验直觉只能部分迁移。

可能的原因（不是讲义逐条给出的结论）：

- **计算瓶颈**：模型规模、序列长度、硬件及并行方式变化时，计算、内存和通信的开销占比可能改变；不能仅凭模型大小断定瓶颈。
- **超参数**：学习率、批量大小等可能需要随规模调整，但并非必然失效；合适的参数化方法也可以促进超参数迁移。
- **能力评估**：某些任务的表现会随规模显著变化，但不能笼统断言“达到固定参数阈值才出现能力”；是否呈现突变也可能受 [评测指标](https://arxiv.org/abs/2304.15004) 影响。
- **数值精度**：大规模训练常采用 BF16，或在部分运算中采用 FP8，以提高效率；这不是硬性要求，低精度的数值问题也能在小模型上测试。
- **数据与泛化**：过拟合或欠拟合取决于模型容量、数据量与质量、训练时长等因素，不能只由模型大小判断。

**资源与效率：** 讲义明确讨论数据和硬件资源（算力、内存、通信带宽）。人力与电力是实际项目中也要考虑的成本；算法和系统优化主要是提高资源使用效率的方法，不宜与资源本身混为一类。

## Tokenization（分词）

### Introduction to Tokenization

原始文本一般是 Unicode string，比如：

```python
string = "Hello, 🌍! 你好!"
```

LM 处理的不是这个 string，而是一串整数（token IDs）。所以需要 tokenizer 做两件事：`encode` 把 string 转成 ID 序列，`decode` 再转回来。

```python
indices = tokenizer.encode(string)  # list[int]
assert tokenizer.decode(indices) == string
```

注意：`15496` 这种数字本身不代表固定的词；它对应什么，要看具体 tokenizer 的词表。不能拿一串 ID 去猜上面那句话是怎么切的。

### Tokenizer Examples

[交互页面](https://tiktokenizer.vercel.app/?encoder=gpt2)中的几个观察：

- 空格有时会和后面的词放在同一个 token 里，比如 `" world"`。
- 句首的 `"hello"` 和带空格的 `" hello"`，可能是不同的 token。
- `123456789` 也可能被拆成几段数字，而不是一个数字一个 token。

这些只是这个 tokenizer 的表现，不是所有 tokenizer 都这样。具体 ID 还会受前后空格和所选编码影响。

### Character Tokenizer

最直接的办法：一个 Unicode 字符（准确说是一个 code point）对应一个 token。Python 里用 `ord` 变成整数，用 `chr` 变回来：

```python
assert ord("a") == 97
assert ord("🌍") == 127757
assert chr(97) == "a"

string = "Hello, 🌍! 你好!"
indices = [ord(ch) for ch in string]
assert "".join(chr(i) for i in indices) == string
```

问题在于词表：Unicode 码点很多，而且有些字符很少出现。如果直接拿码点当 ID，embedding 表至少要有 `max(indices) + 1` 行（这不是这句话里不同字符的数量）。另一方面，一个字符一个 token，序列也不够短。讲义说它是“两头不讨好”：词表大，压缩率也不高。

### Byte Tokenizer

换一种切法：先把 string 编码为 UTF-8 bytes，再让每个 byte 对应一个 token。一个 byte 的值是 `0`～`255`，所以基础词表只有 256 个 ID。（byte 不是 Python 的 `char` 类型。）

```python
assert "a".encode("utf-8") == b"a"                  # 1 byte
assert "🌍".encode("utf-8") == b"\xf0\x9f\x8c\x8d" # 4 bytes

string = "Hello, 🌍! 你好!"
indices = list(string.encode("utf-8"))
assert bytes(indices).decode("utf-8") == string
```

它不会因为遇到没见过的字而只能输出 `UNK`，但一个汉字通常要占多个 bytes。比如 `"你好"` 的 UTF-8 编码一共是 6 bytes，纯 byte tokenizer 就要用 6 个 token。词表是小了，序列却变长了。Transformer 的标准 attention 计算量随序列长度 $T$ 近似按 $O(T^2)$ 增长，所以这不是免费的。

### Word Tokenizer

还有一种办法是按词切，更接近传统 NLP 的做法。讲义举的例子：

```python
string = "I'll say supercalifragilisticexpialidocious!"
chunks = regex.findall(r"\w+|.", string)
```

这个正则会把连续的 word characters 放在一起，其他字符（如这个例子中的空格、标点）分开。之后还要给每个 chunk 分配一个整数 ID；正则本身并没有完成分词器。这里是演示思路，不是完整的多语言分词规则。

好处是一个常见词可能只占一个 token。但词太多了：罕见词出现次数少，新词也会不断出现。如果词表只收录训练时见过的词、又没有 byte/subword 回退，那么新词只能变成 `UNK`，原词的信息就丢了。

Character tokenizer 词表可能很大；byte tokenizer 词表小但序列长；word tokenizer 序列短一些，却要处理罕见词和新词。下一段是 BPE。

压缩率：$r=\text{UTF-8 bytes}/\text{tokens}$；纯 byte tokenizer 的 $r=1$。

张量形状（补充）：单条样本是长为 $T$ 的 ID 序列；凑成 batch 后通常是 $(B,T)$，过 embedding 后是 $(B,T,d_{\text{model}})$。这部分不是讲义在此处展开的内容。

MiniGPT-CN 对照（非讲义内容）：tokenizer 决定词表大小 $V$，以及一段文本会占多少个 token。

### BPE (Byte-Pair Encoding) Tokenizer

BPE 算法由 Philip Gage 于 1994 年提出，用于数据压缩。[Article](http://www.pennelynn.com/Documents/CUJ/HTML/94HTML/19940045.HTM)
它被引入 NLP 用于 neural machine translation，此前论文一直使用 word-based tokenization，随后 BPE 被 GPT-2 采用。

**Basic idea**：*train* the tokenizer on raw text to construct a vocabulary tailored to the data（构建针对该数据定制的词表）。

**Intuition**：常见的字节序列用单个 token 表示，罕见的序列用多个 token 表示。

**Sketch**：从每个字节作为一个 token 开始，逐步合并最常见的相邻 token 对。

#### Training BPE

先把训练文本编码为 UTF-8 bytes。初始状态下，每个 byte 就是一个 token，ID 是 `0`～`255`：

```text
string
  ↓ UTF-8
byte sequence
  ↓ 每个 byte 转成整数
indices = [token_0, token_1, ..., token_(T-1)]
```

之后重复以下步骤：

1. 在**当前的** `indices` 中统计所有相邻 token 对 `(indices[i], indices[i+1])` 的出现次数。
2. 找出出现次数最多的 pair。不需要把所有 pair 完整排序，只需要找到最大值。
3. 为这个 pair 分配一个新 ID，例如第 $k$ 轮使用 `256 + k`。
4. 在 `vocab` 中记录新 token 对应的 bytes，在 `merges` 中记录这条合并规则。
5. 从左到右，把当前序列中这个 pair 的所有**不重叠**出现位置替换为新 ID。
6. 用替换后的序列重新统计 pair，进入下一轮。

这里统计的是当前 token 序列，不是一直统计原始字符串中的字符。第一轮以后，一个 token 可能已经表示多个 bytes。

例如当前序列是 `[a, a, a, a]`：相邻 pair `(a,a)` 会被统计 3 次，位置分别是 `(0,1)`、`(1,2)`、`(2,3)`；但真正合并时不能重叠，所以从左到右合并后得到 `[aa, aa]`。

训练过程的伪代码：

```text
TRAIN_BPE(text, num_merges):
    indices = UTF8_BYTES_AS_INTEGERS(text)

    vocab = {
        0: byte(0),
        1: byte(1),
        ...,
        255: byte(255)
    }
    merges = []

    repeat num_merges times:
        if LENGTH(indices) < 2:
            break

        counts = empty map
        for i = 0 to LENGTH(indices) - 2:
            pair = (indices[i], indices[i + 1])
            counts[pair] += 1

        best_pair = pair with the largest count
        new_id = next unused token ID

        vocab[new_id] = vocab[best_pair.left] + vocab[best_pair.right]
        APPEND(merges, (best_pair, new_id))

        indices = MERGE(indices, best_pair, new_id)

    return vocab, merges
```

其中 `MERGE` 的伪代码：

```text
MERGE(indices, pair, new_id):
    result = []
    i = 0

    while i < LENGTH(indices):
        if i + 1 < LENGTH(indices)
           and indices[i] == pair.left
           and indices[i + 1] == pair.right:
            APPEND(result, new_id)
            i = i + 2              # 两个旧 token 已经被合并
        else:
            APPEND(result, indices[i])
            i = i + 1

    return result
```

如果多个 pair 的次数相同，需要一个固定的 tie-breaking 规则，保证相同训练数据得到相同结果。具体规则要以作业要求为准。

训练完成后得到两个重要结果：

- `vocab: token_id -> bytes`，用于知道每个 token 实际表示哪些 bytes。
- `merges: (token_id_1, token_id_2) -> new_token_id`，并且需要保留学习顺序。

不考虑 special tokens 时，做 $M$ 次有效合并后，词表大小是：

$$
|V| = 256 + M
$$

#### Encoding with BPE

训练和编码是两件事。对新文本执行 `encode` 时，不再重新统计最高频 pair，而是使用训练阶段学到的 `merges`：

```text
ENCODE(text, merges):
    indices = UTF8_BYTES_AS_INTEGERS(text)

    for (pair, new_id) in merges 按训练顺序:
        indices = MERGE(indices, pair, new_id)

    return indices
```

这是讲义里的简单版本：它会遍历所有 merge rules，容易理解但比较慢。实际实现只处理当前文本中真正可能发生的合并，还需要考虑 pre-tokenization 和 special tokens。

#### Decoding with BPE

`decode` 不需要逆着执行 merge rules。只要查 `vocab`，把每个 ID 对应的 bytes 依次拼起来，再用 UTF-8 解码：

```text
DECODE(indices, vocab):
    byte_sequence = empty bytes

    for token_id in indices:
        byte_sequence += vocab[token_id]

    return UTF8_DECODE(byte_sequence)
```

BPE 的结果是：高频 byte sequence 逐渐变成较大的 token，低频内容仍可退回到底层 bytes，因此不会像纯 word tokenizer 那样依赖 `UNK`。随着合并进行，词表变大，训练文本中的 token 数通常减少，压缩率通常提高；但更大的词表也意味着更大的 embedding/output 参数矩阵。

---

## PyTorch（张量操作与资源核算）

资料：[Spring 2026 第二讲](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) · [Recording Version](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording)

本讲从 `motivating_questions()` 开始，核心是建立 **resource accounting mindset**：一切都是 tensor 上的操作（参数、梯度、激活、优化器状态、数据）。

下面的代码是分段示例，不是一个可以从上到下直接运行的脚本。常用导入：

```python
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum, reduce
```

`cuda_if_available()`、`benchmark()`、`get_num_parameters()` 是讲义中的辅助函数，不是 PyTorch 内置接口。

要带走三件事：

- **Mechanics**：直接对应 PyTorch 语义，比较直接。
- **Mindset**：资源核算（resource accounting），要养成习惯。
- **Intuitions**：感受资源花在哪里；本讲没有 ML 魔法。

### Resource Accounting

课程结构（对应函数调用）：

```text
motivating_questions()

    # Memory accounting
    tensors_basics()
    tensors_memory()
    tensors_on_gpus()

    # Compute accounting
    tensor_einops()
    tensor_operations_flops()

    arithmetic_intensity()

    # Memory and compute accounting for training
    deep_network()
    gradients_basics()
    gradients_flops()
    optimizer()
    train_loop()

    # More memory optimizations
    gradient_accumulation()
    activation_checkpointing()
```

**两个 back-of-the-envelope 估算（motivating questions）**:

Q1：70B 参数模型，15T tokens，1024 张 H100，要训多久？

```python
total_flops = 6 * 70e9 * 15e12          # = 6.3e24
h100_flop_per_sec = 1979e12 / 2         # = 9.895e14
mfu = 0.5
flops_per_day = h100_flop_per_sec * mfu * 1024 * 60 * 60 * 24   # ≈ 4.38e22
days = total_flops / flops_per_day      # ≈ 143.9
```

结论：**约 144 天**。

Q2：8 张 H100，用 AdamW，最多能训多大的模型？

```python
h100_bytes = 80e9
bytes_per_parameter = 2 + 2 + (4 + 4)   # 参数 2 + 梯度 2 + Adam 状态 8 = 12
num_parameters = (h100_bytes * 8) / bytes_per_parameter   # ≈ 53.3e9
```

结论：**约 53B 参数**。Caveat：假设参数、梯度用 BF16，Adam 状态用 FP32，且能把这些状态分散到 8 张卡上；未计 activations、临时缓冲等开销，这是理想上界。普通数据并行会在每张卡上复制模型，不能直接把显存相加。

### Tensor Basics

Tensor 是存储一切的基本单位：

- data
- parameters
- gradients
- optimizer state
- activations

**Rank** 是 tensor 的维度数：

```python
x = torch.zeros(4)        # rank 1 (vector)
x = torch.zeros(4, 8)     # rank 2 (matrix)
x = torch.zeros(4, 8, 2)  # rank 3
```

Transformer 中常见 rank 4：

```python
B = 32   # Batch size
S = 16   # Sequence length
H = 16   # Number of heads
D = 64   # Hidden dimension per head
x = torch.zeros(B, S, H, D)
```

MiniGPT-CN 对照（非讲义内容）：输入 token IDs 通常是 `(B, S)`，embedding 后是 `(B, S, d_model)`；多头 attention 的中间张量还会出现 head 维度。

### Tensor Memory

Tensor 数据的内存 = 元素数量 × 每个元素占用的字节数。不包含 allocator 缓存、临时缓冲等额外开销，也不能把共享 storage 的多个 view 重复相加。

```python
def get_memory_usage(x):
    return x.numel() * x.element_size()

x = torch.zeros(4, 8)
assert x.dtype == torch.float32
assert x.numel() == 4 * 8
assert x.element_size() == 4   # float32 是 4 个字节
assert get_memory_usage(x) == 4 * 8 * 4   # 128 bytes
```

例：GPT-3 feedforward 层的一个矩阵：

```python
assert get_memory_usage(torch.empty(12288 * 4, 12288)) == 2304 * 1024 * 1024
# 2304 MiB = 2.25 GiB ≈ 2.42 GB
```

#### FP32

- 默认类型，也叫 float32 / single precision。
- 科学计算里的 baseline，也可以用 fp64。
- 深度学习可以更“糙”一点。

#### FP16

- 也叫 float16 / half precision，每元素 2 bytes。
- 动态范围差，小数容易下溢：

```python
x = torch.tensor([1e-8], dtype=torch.float16)
assert x == 0   # Underflow!
```

- 训练时可能不稳定。

#### BF16

- Google Brain 2018 年提出（brain floating point）。
- 每元素 2 bytes，和 fp32 一样有 8 个 exponent bits，动态范围接近 fp32。
- fraction bits 比 fp16 更少，表示相近数值的精度更低；动态范围大不等于精度高，也不保证训练稳定。

```python
x = torch.tensor([1e-8], dtype=torch.bfloat16)
assert x != 0   # No underflow!
```

#### Mixed Precision

- 纯 fp32 能训，但内存开销大。
- 纯 fp16 / bf16 有风险，可能不稳定。
- 讲义给出的一种**混合精度配置**，也用于后面的内存估算：
  - 参数、激活、梯度用 bf16
  - optimizer state 用 fp32（稳定性）

PyTorch 提供 AMP，按操作选择计算 dtype。下面是补充示例，需要可用的 CUDA GPU：

```python
x = torch.ones(4, 8, device="cuda", dtype=torch.float32)
w = torch.ones(8, 4, device="cuda", dtype=torch.float32)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    y = x @ w

assert x.dtype == torch.float32
assert w.dtype == torch.float32
assert y.dtype == torch.bfloat16
```

AMP 的 matmul 等操作可用 BF16，exp 等操作按规则使用 FP32。它不会自动把模型参数、梯度和 optimizer state 全部变成上面的配置；在 autocast 里调用 `torch.zeros()` 也不会自动创建 BF16 tensor。

补充：[PyTorch AMP 文档](https://docs.pytorch.org/docs/2.11/amp.html)。

#### FP8 and FP4

- **FP8**：2022 年标准化，面向 ML 工作负载。H100 支持两种：
  - E4M3：范围 `[-448, 448]`
  - E5M2：范围 `[-57344, 57344]`
- **FP4 (nvfp4)**：2025 年 NVIDIA 推出，每值仅 4 bits。
  - 取值：`-6, -4, -3, -2, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2, 3, 4, 6`
  - 用 per-block scale factor 换取更大动态范围。
  - Nemotron 3 Super 已用 NVFP4 训练。

注意：部分低精度处理由 NVIDIA 库在用户控制之外完成。

### Tensors on GPUs

默认在 CPU 内存：

```python
x = torch.zeros(32, 32)
assert x.device == torch.device("cpu")
```

要利用 GPU 的并行能力，需要移动到 GPU：

```python
device = cuda_if_available()
x = x.to(device)
```

或者直接在 GPU 上创建：

```python
with torch.device(device):
    x = torch.zeros(32, 32)
```

### Tensor Operations with Einops

Einops 是一套「给维度命名」的张量操作库，灵感来自 Einstein summation notation。

动机：传统 PyTorch 代码容易搞混维度：

```python
x = torch.ones(2, 2, 3)      # batch seq hidden
y = torch.ones(2, 2, 3)      # batch seq hidden
z = x @ y.transpose(-2, -1)  # batch seq seq
```

`-2`、`-1` 是什么？一眼看不出来。

#### Einsum

```python
x = torch.ones(3, 4)   # seq1 hidden
y = torch.ones(4, 3)   # hidden seq2

# 传统
z = x @ y

# einops
z = einsum(x, y, "seq1 hidden, hidden seq2 -> seq1 seq2")  # (3, 3)
```

更复杂的例子：

```python
x = torch.ones(2, 3, 4)   # batch seq1 hidden
y = torch.ones(2, 3, 4)   # batch seq2 hidden

# 传统
z = x @ y.transpose(-2, -1)

# einops
z = einsum(x, y, "batch seq1 hidden, batch seq2 hidden -> batch seq1 seq2")  # (2, 3, 3)

# 用 ... 广播任意维
z = einsum(x, y, "... seq1 hidden, ... seq2 hidden -> ... seq1 seq2")
```

规则：**输出里没出现的维度会被求和掉。**

#### Reduce

```python
x = torch.ones(2, 3, 4)   # batch seq hidden

# 传统
y = x.sum(dim=-1)

# einops
y = reduce(x, "... hidden -> ...", "sum")  # (2, 3)
```

支持 `sum`、`mean`、`max`、`min` 等。

#### Rearrange

有时一个维度其实代表两个维度：

```python
x = torch.ones(3, 8)     # seq total_hidden，total_hidden = heads * hidden1
w = torch.ones(4, 4)     # hidden1 hidden2

# 拆开
x = rearrange(x, "... (heads hidden1) -> ... heads hidden1", heads=2)  # (3, 2, 4)

# 做变换
x = einsum(x, w, "... hidden1, hidden1 hidden2 -> ... hidden2")  # (3, 2, 4)

# 合回去
x = rearrange(x, "... heads hidden2 -> ... (heads hidden2)")  # (3, 8)
```

### FLOPs and FLOP/s

- **FLOPs**：floating-point operations，衡量计算量。
- **FLOP/s**：floating-point operations per second，衡量硬件速度（也写作 FLOPS）。

两者发音相同，注意区分。

直觉：

- GPT-3 训练约 `3.14e23` FLOPs。
- GPT-4 传闻约 `2e25` FLOPs。
- 讲义采用的 H100 SXM FP16/BF16 Tensor Core 峰值（含稀疏）1979 TFLOP/s，不含稀疏约一半：`1979e12 / 2`。这不是 FP32 的峰值。

8 张 H100 跑 2 周的理论计算量（假设一直达到峰值，未乘 MFU）：

```python
total_flops = 8 * 2 * (60 * 60 * 24 * 7) * h100_flop_per_sec
```

#### Matrix Multiplication

```python
x = torch.ones(B, D)   # B 个点，每个 D 维
w = torch.randn(D, K)  # D 维输入，K 维输出
y = x @ w   # (B, K)
```

FLOPs 计数：

$$
\text{FLOPs} \approx 2 \times B \times D \times K
$$

常规估算把一次乘法和一次加法计为 2 FLOPs。逐个点积数乘法与加法，精确计数是 `B * K * (2 * D - 1)`，大矩阵时忽略减去的那一项。

实际测：

```python
actual_num_flops = 2 * B * D * K
actual_time = benchmark(lambda: x @ w)
actual_flop_per_sec = actual_num_flops / actual_time
```

#### Model FLOPs Utilization (MFU)

$$
\text{MFU} = \frac{\text{actual FLOP/s}}{\text{promised FLOP/s}}
$$

这里用一次 matmul 的实际吞吐量作简化演示；完整训练的 MFU 要用模型计算量除以实际训练时间，再与所有 GPU 的理论峰值比较。通常 **MFU ≥ 0.5 已经很不错**，但取决于模型和硬件。GPU 峰值应与所用精度、稀疏性对应。

为什么 MFU 达不到 1？需要理解 GPU 的计算方式，见下一节。

### Arithmetic Intensity

计算一次操作的耗时取决于两件事：

1. **加速器速度**（FLOP/s）
2. **内存带宽**（bytes/s）

H100 参考值：

```python
h100_flop_per_sec = 1979e12 / 2    # 不含稀疏
h100_bytes_per_sec = 3.35e12
```

#### Memory-bound vs. Compute-bound

- **Memory-bound**：内存读写时间 > 计算时间
- **Compute-bound**：计算时间 > 内存读写时间

这里的通信指 GPU 内存与计算单元之间的数据搬运，不是多卡通信。令计算量为 `F`，搬运字节数为 `Q`，峰值算力为 `P`，内存带宽为 `W`：

$$
t_{\text{compute}} = F/P,\qquad t_{\text{memory}} = Q/W
$$

充分重叠时，理想耗时约为两者的最大值；实际耗时还会有额外开销。

用 intensity 判断：

- **Accelerator intensity** = `FLOP/s ÷ bytes/s`，硬件能提供多少计算/字节。
- **Arithmetic intensity** = `FLOPs ÷ bytes`，这个 workload 实际需要多少计算/字节。

判断：

- `arithmetic < accelerator` → **memory-bound**
- `arithmetic > accelerator` → **compute-bound**

#### ReLU and GELU

**ReLU**：

```python
n = 1024 * 1024
x = torch.ones(n, dtype=torch.bfloat16, device="cuda")
y = torch.relu(x)

bytes = (2 * n) + (2 * n)   # 读 x，写 y
flops = n                    # n 次比较
arithmetic_intensity = flops / bytes   # ≈ 1/4
```

`1/4 < accelerator intensity` → **memory-bound**。

**GELU**：每字节做更多工作，arithmetic intensity 更高，但**仍然 memory-bound**。

结论：在这里的带宽受限估算下，两者耗时可能接近；ReLU 运算更少，不保证明显更快。实际速度要测量。

#### Dot Product and Matrix-vector Product

**Dot product**：

```text
bytes = 2n + 2n + 2       # 读 x，读 w，写 y
flops = 2n - 1
arithmetic_intensity ≈ 1/2
```

**Matrix-vector product**（`x @ w`，`x` 是 `n` 维，`w` 是 `n×n`）：

```text
bytes = 2n + 2n² + 2n
flops = n * (2n - 1)
arithmetic_intensity ≈ 1
```

都是 **memory-bound**。

#### Matrix Multiplication

```python
n = 1024
x = torch.ones(n, n, dtype=torch.bfloat16, device="cuda")
w = torch.ones(n, n, dtype=torch.bfloat16, device="cuda")
y = x @ w

bytes = 2 * n**2 + 2 * n**2 + 2 * n**2
flops = n**2 * (2 * n - 1)
arithmetic_intensity = flops / bytes   # ≈ n/3
```

当 `n` 足够大时，`n/3 > accelerator intensity` → **compute-bound**。

结论：

- 在这里的理想估算下，**足够大的矩阵乘法是 compute-bound**；实际能否接近峰值还取决于实现。
- **训练 Transformer 主要是大矩阵乘法**。
- **小 batch 的逐 token decoding 常是 memory-bound**。补充：prefill 和较大 batch 的运算可有更高算术强度，不能把所有推理都归为 memory-bound。

注意：算术强度也取决于精度（bf16 vs fp32）。

推理阶段的区分见补充资料：[Transformer Math](https://jax-ml.github.io/scaling-book/transformers/)。

#### Roofline Model

用 roofline plot 可视化 arithmetic intensity 与性能的关系：

- x 轴：某个具体计算的 arithmetic intensity。
- 每段折线：一种硬件。
- Kink 点：accelerator intensity（memory-bound 与 compute-bound 的分界）。

讲义把 roofline 下的理想利用率写成等式。更准确地说，它给出的是单个操作实际吞吐量相对峰值的上界：

$$
\frac{P_{\text{actual}}}{P_{\text{peak}}}
\leq \min\left(1, \frac{\text{arithmetic intensity}}{\text{accelerator intensity}}\right)
$$

充分利用硬件、忽略额外开销时才可能接近右侧；不能把它当成实际训练 MFU 的恒等式。参考：[Roofline](https://jax-ml.github.io/scaling-book/roofline/)。

### Deep Networks

考虑 `L` 层、`D` 维的深层网络：

```python
D = 8   # 输入、激活、输出维度
L = 3   # 层数
model = DeepNetwork(dim=D, num_layers=L)

num_parameters = get_num_parameters(model)
assert num_parameters == (D * D) * L

B = 4
x = torch.randn(B, D)
y = model(x)
```

#### nn.Module and nn.Parameter

补充：`nn.Module` 组织模型；赋给它的 `nn.Parameter` 会注册为可学习参数。`nn.ModuleList` 注册其中的子模块，让参数能被 `model.parameters()` 找到，也能随模型一起移动设备。这里没有 bias，所以参数量是 `L * D * D`。

```python
class Block(nn.Module):
    """Linear + ReLU。"""
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.weight
        x = F.relu(x)
        return x


class DeepNetwork(nn.Module):
    """把 dim 维向量映射到 dim 维向量。"""
    def __init__(self, dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([Block(dim) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
```

### Gradients and Backpropagation

前面都是 forward；现在开始算 gradient（backward）。

#### Autograd

简单线性模型：

$$
y = 0.5 (x \cdot w - 5)^2
$$

```python
x = torch.tensor([1., 2, 3])
w = torch.tensor([1., 1, 1], requires_grad=True)
pred_y = x @ w
loss = 0.5 * (pred_y - 5).pow(2)

loss.backward()
assert torch.equal(w.grad, torch.tensor([1, 2, 3]))
```

`requires_grad=True` 告诉 PyTorch 追踪这个 tensor 的梯度。

#### Forward and Backward FLOPs

两层线性网络：

```python
B, D = 1024, 256
x = torch.ones(B, D)
w1 = torch.randn(D, D, requires_grad=True)
w2 = torch.randn(D, D, requires_grad=True)

h1 = einsum(x, w1, "batch in, in out -> batch out")
h2 = einsum(h1, w2, "batch in, in out -> batch out")
loss = (h2.mean() - 0)**2
h1.retain_grad()   # 非叶子 tensor，调试时显式保留 .grad
h2.retain_grad()
loss.backward()
```

聚焦第二层 `h2 = h1 @ w2`：

- **Forward FLOPs**：`2 * B * D * D`
- **Backward 需要算**：
  - `h1.grad = d loss / d h1` → `2 * B * D * D`
  - `w2.grad = d loss / d w2` → `2 * B * D * D`
- **Backward FLOPs**：`2 * (2 * B * D * D)`


---

```python
h1_grad = einsum(h2.grad, w2, "batch out, in out -> batch in")
assert torch.allclose(h1.grad, h1_grad)

h1.grad = h2.grad @ (d h2 / d h1)的证明如下:
```

**设定**：$B, C \in \mathbb{R}^{n\times n}$，$A = BC$，$L$ 是依赖 $A$ 的标量。记上游梯度 $G := \dfrac{\partial L}{\partial A}$（与 $A$ 同形，$G_{mk} = \partial L/\partial A_{mk}$​），后面少写几个字。

**最关键的一步，一个类比**：别把 $A$ 当"矩阵"看。把 $L$ 写成 $L(A_{11},\ \cdots,\ A_{ij},\ \cdots,\ A_{nn})$——**看作 $n^2$ 元多元函数**。思绪一下就打开了：矩阵求导根本不是新东西，就是普通的多元微积分，只不过自变量有 $n^2$ 个、刚好排成了一个方阵。既然 $L$ 是这 $n^2$ 个变量的函数，它对每个 $A_{mk}$ 都有普通的偏导数，多元链式法则也照常能用。

**要算的东西长什么样**：$\dfrac{\partial L}{\partial B}$ 按定义就是把每个偏导数排回原位置：

$$\frac{\partial L}{\partial B} = \begin{pmatrix} \dfrac{\partial L}{\partial B_{11}} & \cdots & \dfrac{\partial L}{\partial B_{1n}} \\[1ex] \vdots & \ddots & \vdots \\[1ex] \dfrac{\partial L}{\partial B_{n1}} & \cdots & \dfrac{\partial L}{\partial B_{nn}} \end{pmatrix}$$

所以整个问题归结为：算出任意一个位置上的 $\dfrac{\partial L}{\partial B_{ij}}$。

**对这个 $n^2$ 元函数用链式法则**：$B_{ij}$ 不是只影响一个 $A$，它影响多少个，就要把多少条路径的贡献加起来：

$$\frac{\partial L}{\partial B_{ij}} = \sum_{m=1}^n \sum_{k=1}^n \frac{\partial L}{\partial A_{mk}} \cdot \frac{\partial A_{mk}}{\partial B_{ij}}$$

**看谁真的被影响**：回到 $A_{mk} = \sum_{k'} B_{mk'}\, C_{k'k}$——**行数为 $i$ 的 $A_{mk}$ 才有 $B_{ij}$ 参与运算**（出现在 $k'=j$ 那一项，系数是 $C_{jk}$）；行号 $m \neq i$ 的元素里根本没有 $B_{ij}$，偏导为 $0$。双重求和塌缩成一层：

$$\frac{\partial L}{\partial B_{ij}} = \sum_{k=1}^n \frac{\partial L}{\partial A_{ik}} \cdot \frac{\partial A_{ik}}{\partial B_{ij}} = \sum_{k=1}^n \frac{\partial L}{\partial A_{ik}} \cdot C_{jk} = \sum_{k=1}^n G_{ik}\, C_{jk}$$

注意这个 $\sum_k$ 不能省：$B_{ij}$ 喂的是第 $i$ 行整整 $n$ 个元素，若只取 $k=j$ 一项（误写成 $G_{ij}C_{ji}$）就漏掉了 $n-1$ 条路径的贡献。

**认出矩阵乘法**：这个和，就是「$G$ 的第 $i$ 行」点乘「$C$ 的第 $j$ 行竖起来」：

$$= \begin{pmatrix} \dfrac{\partial L}{\partial A_{i1}}, & \cdots & \dfrac{\partial L}{\partial A_{in}} \end{pmatrix} \begin{pmatrix} C_{j1} \\ \vdots \\ C_{jn} \end{pmatrix}$$

行点列正是矩阵乘法的定义；而"把 $C$ 的第 $j$ 行竖成列"这件事，名字就叫 $C^T$ 的第 $j$ 列。

**收尾**：上式对每个位置 $(i,j)$ 都成立，按开头梯度矩阵的定义把这些数排回去：

$$\Rightarrow \quad \frac{\partial L}{\partial B} = \frac{\partial L}{\partial A} \cdot C^T = G\,C^T$$

维度自检：$(n\times n)(n\times n) = n\times n$，与 $B$ 同形；若忘了转置，$G\,C$ 虽然也乘得动，但元素和上面的求和式对不上。

**$C$ 那边完全对称**：看 $C_{ij}$ 影响谁——在 $A_{mk} = \sum_{k'} B_{mk'}\, C_{k'k}$ 里，$C_{ij}$ 只在列号 $k=j$、求和指标 $k'=i$ 时出现，系数是 $B_{mi}$，即**列数为 $j$ 的 $A_{mk}$ 才有 $C_{ij}$ 参与运算**。求和塌缩成对 $m$：

$$\frac{\partial L}{\partial C_{ij}} = \sum_{m=1}^n \frac{\partial L}{\partial A_{mj}} \cdot B_{mi} = \sum_{m=1}^n G_{mj}\, B_{mi} = \begin{pmatrix} B_{1i}, & \cdots & B_{ni} \end{pmatrix} \begin{pmatrix} G_{1j} \\ \vdots \\ G_{nj} \end{pmatrix}$$

「$B$ 的第 $i$ 列横过来」就是 $B^T$ 的第 $i$ 行，点乘「$G$ 的第 $j$ 列」= $(B^T G)_{ij}$。排回去：

$$\Rightarrow \quad \frac{\partial L}{\partial C} = B^T \cdot \frac{\partial L}{\partial A} = B^T G$$

**为什么总是转置**：前向里 $B_{ij}$ 沿系数 $C_{jk}$ 把影响分发到第 $i$ 行、$C_{ij}$ 沿系数 $B_{mi}$ 分发到第 $j$ 列；反向收回时走的还是同一批边、系数不变，只是行列角色互换——这就是 $C^T$ 和 $B^T$ 的全部来源。

结论：对需要计算输入梯度和权重梯度的线性层，**backward 的主要矩阵乘法开销约是 forward 的 2 倍**。首层若不需要输入梯度，可少算一次矩阵乘法。

---

对所有层求和：

- Forward：`2 × (# data points) × (# parameters)` FLOPs
- Backward：`4 × (# data points) × (# parameters)` FLOPs
- **Total：`6 × (# data points) × (# parameters)` FLOPs**

这是线性层计算占主导时的近似，忽略激活函数、loss、优化器更新等开销；对短上下文的 Transformer 也常用。语言模型每步处理的 token 数是 `B * S`，所以常写成 `6 * B * S * N`，其中 `N` 是参数量；长上下文时不能忽略 attention 的额外计算。

### Optimizers

回顾优化器家族：

- momentum = SGD + grad 的指数平均
- AdaGrad = SGD + grad² 的累积
- RMSProp = AdaGrad，但用 grad² 的指数平均
- Adam = RMSProp + momentum

讲义中的 AdaGrad 简化示例（保留原写法，不是通用优化器实现）：

```python
class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                state = self.state[p]
                grad = p.grad.data
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)
                state["g2"] = g2
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)
```

#### AdaGrad

- 维护二阶矩的累积 `g2 = sum g_i^2`。
- 若状态用 FP32，每参数 4 bytes。

上面代码的 `torch.zeros_like(grad)` 跟随梯度 dtype，不会自动创建 FP32 状态。`.data` 是讲义的简化写法，实际实现通常用 `torch.no_grad()` 更新参数，并处理 `grad is None` 的情况。

#### Training Memory and Compute

以 `D=4, L=3, B=2` 的小网络为例，假设参数、梯度、激活是 BF16，AdaGrad 状态是 FP32；这是估算配置，不是上面默认 FP32 代码的实际内存：

```python
num_parameters = D * D * L
parameter_memory = 2 * num_parameters            # bf16
gradient_memory = 2 * num_parameters             # bf16
optimizer_state_memory = 4 * num_parameters      # fp32（AdaGrad）
activation_memory = 2 * (B * D * L)              # bf16
total_memory = parameter_memory + activation_memory + gradient_memory + optimizer_state_memory
```

优化器状态内存（假设状态是 FP32）：

- AdaGrad：4 bytes/参数（二阶矩）
- **Adam：8 bytes/参数**（一阶 + 二阶矩）

一步训练的 FLOPs：

```python
flops = 6 * B * num_parameters
```

Transformer 的核算更复杂，但思路一样——Assignment 1 会要求做这件事。

### Training Loop

下面保留讲义的循环结构，但修正预测的 shape：讲义的 `model(x).mean()` 把整个 batch 变成标量，会和 `(B,)` 的目标发生广播；这里改为只对特征维度取 mean，每个样本得到一个预测。

```python
D = 16
true_w = torch.arange(D, dtype=torch.float32)

B = 4
def get_batch():
    x = torch.randn(B, D)
    true_y = x @ true_w
    return (x, true_y)

L = 2
model = DeepNetwork(dim=D, num_layers=L)
optimizer = AdaGrad(model.parameters(), lr=0.01)

num_train_steps = 3
for t in range(num_train_steps):
    x, y = get_batch()
    pred_y = model(x).mean(dim=-1)   # (B, D) -> (B,)，笔记中的修正
    loss = F.mse_loss(pred_y, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
```

标准训练循环：**取 batch → forward → loss → backward → step → zero_grad**。梯度默认会累加，更新后要清空，除非正在做 gradient accumulation。

### Gradient Accumulation

动机：

- 较大 batch 可降低梯度估计的噪声，但并非越大越好。
- 但 activation memory 随 batch size 线性增长，可能 OOM。

做法：

1. 在 micro batch 上计算梯度。
2. **累加**梯度（不 zero）。
3. 每 `batch_size / micro_batch_size` 步更新一次参数并清空梯度。

补充：若 `K` 个 micro batch 大小相同，且每个 loss 都取 mean，每次 backward 前要把 loss 除以 `K`，才能得到完整 batch 的平均梯度；大小不等时按样本数加权。参考：[PyTorch 梯度累积示例](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html#gradient-accumulation)。

效果：每个 micro batch 做完 backward 后释放其计算图，activation memory 按 micro batch size 计；参数、梯度和优化器状态仍需保留。

```python
activation_memory = 2 * B * D * L              # 完整 batch
activation_memory = 2 * micro_batch_size * D * L   # 用累积后
```

### Activation Checkpointing

普通训练的 autograd 会保存反向传播所需的中间 tensor，不是每个中间结果都要保存。无梯度推理不需要为 backward 保留这些 tensor；Transformer decoding 仍可能保留跨步使用的 KV cache。

```python
activation_memory = 2 * B * D * L
```

**Activation checkpointing = gradient checkpointing = rematerialization**：

- Forward：只保留部分层的 activations。
- Backward：从最近的 checkpoint 重新计算缺失的 activations。
- 哲学：**用计算换内存**。

对比：

```text
存所有激活：    x g1 h1 g2 h2 g3 h3 g4 h4
激活检查点：    x    h1    h2    h3    h4
```

PyTorch 实现（补充：当前文档推荐显式指定 `use_reentrant=False`）：

```python
from torch.utils.checkpoint import checkpoint

x = checkpoint(layer, x, use_reentrant=False)
```

参考：[PyTorch Checkpoint 文档](https://docs.pytorch.org/docs/2.11/checkpoint.html)。

**Checkpoint 频率权衡**（讲义中的简化链式网络估算，假设每层开销相近，只比较激活内存与重算开销）：

- 保存每层所需的中间激活：activation memory `O(L)`，无重算。
- 只留原始输入、不留中间 checkpoint：采用逐层从输入重算的策略，activation memory `O(1)`，重算 `O(L²)`。
- 每 `sqrt(L)` 层存一次 checkpoint，并分段重算：activation memory `O(sqrt(L))`，重算 `O(L)`。

“每层保存所需的中间激活”和“对每个 layer 调用 checkpoint”不是一回事；后者仍会重算层内部操作。

### Personal Understanding

本讲不是教你训练模型，而是教你**快速做资源估算**。拿到一个训练任务，先问：

- 需要多少 FLOPs？
- 需要多少显存？
- 是 compute-bound 还是 memory-bound？

四条核心结论：

1. **一切都是 tensor 上的操作**：参数、梯度、激活、优化器状态、数据。
2. **`einops`** 是更好的思考张量操作的方式。
3. **每训练步约 `6 × (# data points) × (# parameters)` FLOPs**。
4. **矩阵乘法 compute-bound，逐元素操作 memory-bound**；推理因为 matrix-vector 而 memory-bound。
5. **梯度累积和激活检查点**都是为了降低内存、以使用更大 batch size。

两个 motivating 估算都是 back-of-the-envelope，但能快速感受资源量级。实际数字会受 MFU、并行策略、batch size、sequence length、优化器实现等影响。

---

## Architectures and Hyperparameters（架构与超参数）

资料：[课程主页](https://cs336.stanford.edu/) · [Spring 2026 第三讲](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)

### Lecture Overview

本讲通过比较不同语言模型的架构与超参数，观察哪些选择已经比较一致、哪些仍有变化，以及这些选择背后的效果、效率和训练稳定性权衡。

讲义的学习方法：自己动手积累经验，也从其他模型的实验与论文中学习。模型采用某个配置，不等于这个配置在所有场景下都是最优的。

### Transformer Recap

#### Original Transformer

原始 Transformer 的几个选择：**Post-Norm + LayerNorm + ReLU FFN + 正弦/余弦位置编码**。原论文的完整模型是 encoder-decoder；本课作业实现的是用于自回归语言建模的 decoder-only 模型。

#### Modern Transformer

Assignment 1 使用的现代变体：**Pre-Norm + RMSNorm + RoPE + SwiGLU**，并去掉 Linear 的 bias。

不少现代模型还使用 GQA，但它不是上述架构的必选项，也不是 Assignment 1 标准多头注意力实现的要求。

#### Comparing Architectures

架构对比表的每一行是一个模型，各列记录词表大小、归一化方式、激活函数、Attention/FFN 的组织方式等配置。阅读时先问：**共同点是什么？差异在哪里？这些差异有什么依据？**

- **Norm**：表中许多较新的模型采用 RMSNorm，早期模型更多采用 LayerNorm；不能概括成所有模型都已改用 RMSNorm。
- **Layer**：多数采用 Serial，即同一个 block 内先 Attention、后 FFN。
- **Activation / Position**：SwiGLU、GeGLU 等门控变体和 RoPE 较常见，但存在其他选择。
- **Vocabulary size**：不同模型差异较大，需结合语言覆盖范围、序列长度和参数开销理解。

讲义比较的是一组 dense 模型，不能据此推断所有前沿模型都是稠密架构。MLP Factor 也需结合是否使用 GLU 来读，见后面的 Feedforward Dimension。

### Normalization

#### Pre-Norm vs Post-Norm

设 $x$ 是子层输入，$F$ 是 Attention 或 FFN，$N$ 是归一化操作。残差相加要求 $F$ 的输出与 $x$ 形状相同，通常都是 $(B,T,d_{\text{model}})$。

**Post-Norm**：先经过子层，与输入相加，再归一化。
$$
y=N(x+F(x))
$$

**Pre-Norm**：先归一化子层的输入，原始输入沿残差路径直接相加。

$$
y=x+F(N(x))
$$

直觉：Pre-Norm 的残差主路径保留了直接传递 $x$ 的通道；Post-Norm 的相加结果还要经过归一化，因此反向传播也会经过它。

讲义讨论了梯度衰减、梯度尖峰与训练稳定性。补充：[Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) 分析了初始化时 Post-LN 靠近输出层的参数梯度较大，以及 Pre-LN 的梯度更易控制。不能把它简化成“Post-Norm 一定梯度消失或爆炸”，也不能推出所有 Pre-Norm 模型都不需要 warmup。

#### Double Norm and Non-residual Post-Norm

讲义还介绍了在子层输出上增加归一化、再做残差相加的变体。一个同时包含输入和输出归一化的例子是：

$$
y=x+N_2(F(N_1(x)))
$$

这里归一化位于子层分支，残差主路径仍直接传递 $x$。理解这些变体时要看计算图中 Norm 的具体位置，仅看“pre / post”名称不够。

#### LayerNorm vs RMSNorm

对一个 token 的特征向量 $x\in\mathbb{R}^{d}$，其中 $d=d_{\text{model}}$：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad
v=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
\operatorname{LayerNorm}(x)_i
=\frac{x_i-\mu}{\sqrt{v+\epsilon}}\gamma_i+\beta_i
$$

LayerNorm 先去均值，再按标准差缩放，通常还包含可训练的缩放 $\gamma$ 和偏移 $\beta$。

$$
\operatorname{RMSNorm}(x)_i
=\frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d}x_j^2+\epsilon}}\gamma_i
$$

**RMS = root mean square（均方根）**。RMSNorm 不做去均值，按均方根缩放，标准形式只有可训练的 $\gamma$，没有 $\beta$。它不保证输出是零均值；RMS 也不是一般意义上的标准差。

张量形状（补充）：输入 $(B,T,d)$，每个位置沿最后一个特征维度计算统计量；保留该维度时，分母形状为 $(B,T,1)$，$\gamma$ 为 $(d,)$，广播后的输出仍为 $(B,T,d)$。不会把 batch 或不同 token 位置混在一起求均值。

参考：[RMSNorm 原论文](https://arxiv.org/abs/1910.07467)。**Pre/Post-Norm 描述位置，LayerNorm/RMSNorm 描述算法**，所以 Pre-RMSNorm 表示在子层之前使用 RMSNorm。

#### FLOPs vs Runtime

讲义强调：**FLOPs 少，不等于运行时间一定按同样比例减少**。RMSNorm 省掉去均值等操作，也减少了部分参数与数据搬运；实际速度仍取决于内存访问、kernel 融合、硬件和张量大小。

因此更准确的说法是“RMSNorm 在许多实验中兼顾效果和效率”，而不是保证它在所有模型上都更快、更稳定。

#### Dropping Bias Terms

不少现代模型去掉 Linear 和归一化中的 bias。讲义从数据搬运与优化稳定性解释这种选择；这不是说 bias 在数学上无效，或去掉 bias 必然提升模型效果。

### Activations and Feedforward Networks

#### ReLU and GELU

传统的无 bias FFN：

$$
\operatorname{FFN}(x)=\operatorname{ReLU}(xW_1)W_2
$$

**$\max(0,xW_1)$ 是逐元素操作**，不是把整个矩阵与 0 比大小。每个元素分别取它与 0 中较大的值，等价于 ReLU，形状不变。

例如一个位置投影后的向量为 $[-2,0,3]$，经过 ReLU 后为 $[0,0,3]$。

GELU 是另一种常见激活，讲义写为 $\operatorname{GELU}(z)=z\Phi(z)$，其中 $\Phi$ 是标准正态分布的累积分布函数；同样逐元素作用。

#### Gated Linear Units

门控 FFN 增加一个输入投影，让两条分支的结果逐元素相乘。ReGLU 的形式是：

$$
\operatorname{FFN}_{\text{ReGLU}}(x)
=\left(\operatorname{ReLU}(xW_1)\odot(xV)\right)W_2
$$

这里用 $\odot$ 表示 **element-wise product**；讲义中的 $\otimes$ 在这个公式里也表示逐元素相乘，不是矩阵乘法或 Kronecker product。

张量形状（补充，沿用讲义的行向量记法）：

| 张量 | 形状 |
|---|---|
| 输入 $x$ | $(B,T,d_{\text{model}})$ |
| 输入投影 $W_1,V$ | $(d_{\text{model}},d_{\text{ff}})$ |
| 两条分支及其逐元素乘积 | $(B,T,d_{\text{ff}})$ |
| 输出投影 $W_2$ | $(d_{\text{ff}},d_{\text{model}})$ |
| FFN 输出 | $(B,T,d_{\text{model}})$ |

注意：这里的矩阵方向服务于 $xW$ 的公式；作业 Linear 参数按 $(d_{\text{out}},d_{\text{in}})$ 存储，核对实现时要区分这两个约定。

门控直觉是让一条分支调节另一条分支的特征。ReGLU/SwiGLU 的门控值不必位于 $[0,1]$，不能都理解为概率开关。

#### GeGLU and SwiGLU

将 ReGLU 中的 ReLU 换成 GELU，就得到 GeGLU；换成 Swish，就得到 SwiGLU。

$$
\operatorname{Swish}(z)=z\operatorname{sigmoid}(z)
=\frac{z}{1+e^{-z}}
$$

这里是参数为 1 的 Swish，也称 SiLU。

$$
\operatorname{FFN}_{\text{SwiGLU}}(x)
=\left(\operatorname{SiLU}(xW_1)\odot(xV)\right)W_2
$$

讲义引用实验说明 SwiGLU/GeGLU 常带来效果收益；这是经验依据，不代表所有任务上都必然优于其他激活。参考：[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)。

#### Serial vs Parallel Layers

**Serial**：一个 block 内先计算 Attention 并做残差相加，再把更新后的表示交给 FFN。

$$
h=x+\operatorname{Attention}(N_1(x)),\qquad
y=h+\operatorname{FFN}(N_2(h))
$$

这里有两次残差相加，FFN 能使用本层 Attention 更新后的表示。

**Parallel**：Attention 和 FFN 分支从同一个 block 输入出发，再将分支输出与残差相加。具体归一化是否共享，要看模型设计。

共享归一化输入的一种形式：

$$
y=x+\operatorname{Attention}(N(x))+\operatorname{FFN}(N(x))
$$

讲义提到 GPT-J、PaLM、GPT-NeoX 等采用过并行层。因为两条分支依赖同一个输入，有机会共享归一化、融合输入投影，减少执行开销。

补充：[PaLM 原论文](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf) 报告其大规模训练配置下约 15% 的提速；8B 消融有小幅质量下降，62B 没有观察到质量下降。这个数字和结果有具体实验条件，不能推广成“并行一定更快、质量一定更差”。

**深度直觉（补充）**：串行层中存在 Attention → FFN 的依赖链，并行层没有这条层内依赖，因此单个 block 的串行子层路径更短。但 block 数并没有减半，两条分支仍包含非线性，多层堆叠也仍有串行依赖；不能直接说整个模型的“有效深度腰斩”。

这里的 Serial/Parallel 指子层的依赖关系，不是多 GPU 并行方式，也不表示只能逐个 token 计算。讲义中的多数模型选择 Serial。

课程没有证明“串行必然有更高质量上限”，也没有把近年的选择统一归因于推理任务或系统优化。观察到一种配置更常见，与解释它为什么更常见，是两件事。

### Position Embeddings

内容向量告诉模型“是什么”，位置机制帮助模型区分“在哪里、相距多远”。讲义比较把位置信息加到输入表示中，以及把它引入 Attention 计算中的不同方式。

#### Sinusoidal, Absolute and Relative Position Embeddings

| 类型 | 位置如何进入模型 | 讲义中的例子 |
|---|---|---|
| Sinusoidal | 把固定正弦/余弦位置向量加到 token embedding | 原始 Transformer |
| Learned absolute | 把可训练的绝对位置向量加到 token embedding | GPT-1/2/3 |
| Relative | 在 Attention 中加入与相对距离有关的项 | T5 等 |
| RoPE | 按位置旋转 Q、K，再做内积 | GPT-J、PaLM、LLaMA 等 |

固定正弦/余弦编码的一个常见形式：

$$
\operatorname{PE}(p)_{2r}=\sin(p\omega_r),\qquad
\operatorname{PE}(p)_{2r+1}=\cos(p\omega_r),\qquad
\omega_r=10000^{-2r/d_{\text{model}}}
$$

其位置向量之间的内积有相对位置结构。但将位置向量加到内容向量后：

$$
(u+p_i)^\top(v+p_j)
=u^\top v+u^\top p_j+p_i^\top v+p_i^\top p_j
$$

中间的内容—位置交叉项一般仍依赖各自的绝对位置。因此讲义讨论的是“相加后的内容内积不只依赖相对位置”，不能简化成“正弦编码完全没有相对位置性质”。

可训练绝对位置表只直接覆盖表中已有的位置；超过表长时需要额外处理。固定正弦编码和 RoPE 可以计算未见位置的编码，但**能算出编码，不等于模型在更长序列上仍能保持效果**。

相对位置偏置的一种形式（补充）：

$$
s_{ij}=\frac{q_i^\top k_j}{\sqrt{d_h}}+b(i-j)
$$

例如 T5 使用按相对距离分桶的可训练偏置。这时完整分数包含内积之外的项，但不能推出它必然很慢或与 FlashAttention 不兼容。[FlashAttention 官方文档](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention) 支持 ALiBi 这类相对位置偏置；其他偏置形式是否支持、开销如何，要看具体 kernel。

#### RoPE: Intuition

**RoPE = Rotary Position Embedding（旋转位置编码）**。把 Q、K 的坐标两两配对，在各自的二维平面中旋转；位置决定旋转角度，不同坐标对使用不同频率。

讲义希望得到的性质：对于给定的内容向量，加入位置后的内积依赖相对距离，而不单独依赖两个绝对位置。

理解三张图时，先假设 `we` 和 `know` 对应的未旋转向量固定：

- 左图：先看没有位置旋转时的两根箭头。
- 中图：`we know that` 中，两个词分别位于 0、1，按各自位置旋转。
- 右图：`of course we know` 中，它们位于 2、3；相对中图，两根箭头又共同增加了同样的旋转角度。

两者都向后移动 2 个位置，相对距离仍为 1。**共同旋转保持长度、夹角和内积不变**，因此加入位置后的点积分数相同。

二维数值例子（补充）：让两根原始箭头都为 $(1,0)$，每个位置旋转 $30^\circ$。位于 0、1 时，角度为 $0^\circ,30^\circ$；位于 2、3 时，为 $60^\circ,90^\circ$。两次夹角都是 $30^\circ$，内积都是 $\cos30^\circ$。

箭头是几何类比，不能把真实高维向量的一个方向直接解释成某个语义。真实模型的 Q、K 会随层和上下文变化；上面的结论限定于固定未旋转内容向量，不保证换一句话后两个词的 Attention 分数完全相同。

#### RoPE: Mathematics

沿用列向量记法。二维旋转矩阵为：

$$
R(\phi)=
\begin{bmatrix}
\cos\phi&-\sin\phi\\
\sin\phi&\cos\phi
\end{bmatrix}
$$

对位置 $p$ 的一对坐标 $(a,b)$，取角度 $p\omega_r$：

$$
\begin{bmatrix}a'\\b'\end{bmatrix}
=R(p\omega_r)\begin{bmatrix}a\\b\end{bmatrix}
=\begin{bmatrix}
a\cos(p\omega_r)-b\sin(p\omega_r)\\
a\sin(p\omega_r)+b\cos(p\omega_r)
\end{bmatrix}
$$

若完整旋转一个偶数维的 head，常见频率约定为：

$$
\omega_r=\Theta^{-2r/d_h},\qquad r=0,\ldots,d_h/2-1
$$

$\Theta$ 是频率基数，原始常见取值为 10000；具体模型可能使用其他基数或缩放方式。把各坐标对的旋转矩阵沿对角线组合，得到 $R_p\in\mathbb{R}^{d_h\times d_h}$。

关键恒等式：$R_i^\top R_j=R_{j-i}$，(旋转矩阵特性,先顺时针再逆时针(或反过来)结果和直接顺(逆)角度差效果一样)所以：

$$
\widetilde q_i=R_iq_i,\qquad \widetilde k_j=R_jk_j
$$

$$
\widetilde q_i^\top\widetilde k_j
=q_i^\top R_i^\top R_jk_j
=q_i^\top R_{j-i}k_j
$$

位置依赖通过 $j-i$ 进入分数，内容依赖仍保留在 $q_i,k_j$ 中。整体平移位置 $i\rightarrow i+c,j\rightarrow j+c$ 不改变相对距离，也不改变固定内容向量的上述内积。

因此不是“Attention 只取决于距离”，而是“在给定内容向量时，RoPE 引入的位置因素只通过相对距离作用”。参考：[RoFormer 原论文](https://arxiv.org/abs/2104.09864)。

#### RoPE in Attention

常见使用方式：先从隐藏状态投影出 Q、K、V，再对 **Q、K** 应用 RoPE，之后计算 Attention；通常不旋转 V，也不是只在最初的 token embedding 上旋转一次。

标准多头 Attention 的形状回顾（补充，设头数 $H$、每头维度 $d_h$）：

| 张量 | 形状 |
|---|---|
| 隐藏状态 | $(B,T,d_{\text{model}})$ |
| Q、K、V | $(B,H,T,d_h)$ |
| RoPE 后的 Q、K | $(B,H,T,d_h)$ |
| Attention 分数/权重 | $(B,H,T,T)$ |
| 各头输出 | $(B,H,T,d_h)$ |
| 拼接并输出投影后 | $(B,T,d_{\text{model}})$ |

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{\widetilde Q\widetilde K^\top}{\sqrt{d_h}}+M\right)V
$$

$M$ 是 causal mask 等掩码，softmax 沿 key 位置维度进行。旋转保持 Q、K 的形状和范数，不增加 Attention 分数矩阵的大小。

正弦/余弦表在完整旋转的情况下可用 $(T,d_h/2)$ 表示，跨 batch 和 head 使用。数学上写成旋转矩阵，是为了理解；计算只需对坐标对做旋转，不必存储一个稠密的 $d_h\times d_h$ 矩阵。

RoPE 提供相对位置结构，同时保留点积形式。长上下文效果仍取决于训练长度、频率设置、位置缩放和数据；“用了 RoPE 就能完美外推”不是课程结论。

### Hyperparameters

讲义区分常见经验配置、实验依据和实际例外。**常见比例是起点，不是必须满足的数学定律。**

#### Feedforward Dimension

MLP Factor 通常指 $d_{\text{ff}}/d_{\text{model}}$。传统两矩阵 FFN 的常见设置是 $d_{\text{ff}}=4d_{\text{model}}$；GLU 多了一份输入投影，因此要结合参数预算比较。

补充推导：设 $d=d_{\text{model}}$，忽略 bias，传统 FFN 的中间维度为 $m$，门控 FFN 的中间维度为 $m'$：

$$
N_{\text{FFN}}=2dm,\qquad N_{\text{GLU}}=3dm'
$$

要求这两种 **FFN 的参数量相同**：

$$
3dm'=2dm\quad\Rightarrow\quad m'=\frac{2}{3}m
$$

若原先 $m=4d$，就得到 $m'=\frac{8}{3}d$。这解释了讲义里的 $4\rightarrow8/3$，但不是所有 SwiGLU 模型必须遵守的固定比例；实际模型还会选用其他宽度或为硬件对齐做取整。

讲义还列出 T5 11B 的极端例子：$d_{\text{ff}}=65536,d_{\text{model}}=1024$，比例为 64。它说明很不一样的配置也能训练，但不能据此判定这个比例最优；后续 T5 v1.1 又采用了更常见的门控 FFN 宽度。

#### Attention Heads and Head Dimension

常见配置是：

$$
H d_h=d_{\text{model}}
$$

例如 GPT-3 的 $96\times128=12288$。但注意力内部总维度可以通过投影另行选择；讲义中 T5 的 $128\times128$ 大于其 $d_{\text{model}}=1024$。

张量形状（补充）：各头输出拼接后为 $(B,T,Hd_h)$，再投影回 $(B,T,d_{\text{model}})$。因此 $Hd_h=d_{\text{model}}$ 是常见设计约定，不是 Attention 本身必须满足的条件。

增加头数和增加每头维度不是同一件事；比较配置时要同时看 $H,d_h,d_{\text{model}}$，以及参数量、计算和 kernel 支持。

#### Depth vs Width

**Depth** 是 block 数 $L$，**Width** 通常指 $d_{\text{model}}$。讲义用 $d_{\text{model}}/L$ 比较模型的宽深比，许多例子在 100–200 附近，也有明显例外；它不是固定最优区间。

补充核算：保持 FFN 比例和注意力内部维度比例近似不变时，主要 block 参数量约为 $O(Ld_{\text{model}}^2)$。所以讨论“更深还是更宽”时，先固定参数或计算预算，避免把结构变化与规模变化混在一起。

讲义强调系统因素：深层之间有依赖，更深的模型通常有更长的执行路径，可能增加延迟、影响并行效率；更宽也会改变矩阵乘法、内存和通信成本。效果与硬件效率需要一起比较。

#### Vocabulary Size

讲义中的常见数量级：较早的单语言模型约 30k–50k，多语言或面向广泛用途的模型常达到 100k–250k，也有超出这个范围的例子。这些是观察，不是词表大小的上下限。

词表大小 $V$ 的权衡（补充）：

- 更大的词表可让常见文本使用更少 token，但改善幅度取决于语言和训练语料。
- 输入 embedding 参数量为 $Vd_{\text{model}}$；输出 LM head 也随 $V$ 增长，是否共享权重影响总参数量。
- 输出 logits 形状为 $(B,T,V)$，更大的词表增加输出投影和概率计算开销。
- 更短的 token 序列可能降低 Attention 等开销，但不能只看压缩率来选择词表。

多语言数据通常需要兼顾更多文字系统与词形。和 Tokenization 部分连接：词表大小、压缩率、模型开销需要一起评估。

#### Dropout and Weight Decay

讲义提出：预训练数据很多、常见训练并不大量重复同一语料，是否还需要正则化？模型对比中，许多较新的配置不使用 dropout，却仍使用 weight decay；不同模型有例外，论文没写 dropout 也不能证明它一定为 0。

**Dropout**：训练时随机屏蔽一部分激活，常按保留概率缩放；推理时关闭。**Weight decay**：在参数更新中让权重向较小尺度收缩。

讲义强调，weight decay 的作用还涉及优化动态以及与学习率调度的相互影响，不能仅解释为“数据太少时防过拟合”。预训练、微调、小数据实验的合适设置也可能不同。

### Training Stability

训练不稳定可能表现为 loss 尖峰、数值溢出或更新异常。讲义重点讨论 softmax 附近的数值和尺度问题；下面的方法分别针对输出归一化、Q/K 尺度或 logits 幅度。

#### Softmax: Scores to Probabilities

**Softmax 把一组实数分数变成非负、总和为 1 的权重。** 用于分类或词表预测时，这些权重构成一个概率分布；用于 Attention 时，它们决定如何加权各个 value。

输入分数常叫 **logits**，可以是正数、负数，也不要求总和为 1。对 $K$ 个候选项，$z\in\mathbb{R}^{K}$：

$$
p_i=\operatorname{softmax}(z)_i
=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$

先对每个分数取指数，再除以全部指数值的总和。指数把有限实数变成正数，除以总和完成归一化；直接除以原始分数之和，则可能遇到负权重或分母为 0。

在精确计算、输入均有限时，$p_i>0$ 且 $\sum_i p_i=1$。实际浮点计算中，很小的概率可能下溢为 0。

数值例子（补充）：

| 候选项 | Logit $z_i$ | $e^{z_i}$（约） | Softmax 权重（约） |
|---|---|---|---|
| A | 1 | 2.718 | 0.090 |
| B | 2 | 7.389 | 0.245 |
| C | 3 | 20.086 | 0.665 |

分母约为 $2.718+7.389+20.086=30.193$。C 得到最大权重，但 A、B 仍保留非零权重。**Softmax 不会直接选出一个类别**；选择最大概率项是 argmax，根据概率随机选择是 sampling。

为什么叫 “soft”：它用平滑权重表达对不同候选项的偏好，而不是只留下最大值对应的一项。较大的 logit 得到较大的权重，但权重还取决于其他候选项：

$$
\frac{p_i}{p_j}=e^{z_i-z_j}
$$

两项的相对权重由分数差决定。所有分数同时增加同一个常数，分布不变；放大分数差，则通常会让分布更集中。

**张量形状：Softmax 不改变形状，只沿指定维度归一化。**

| 场景 | 输入与输出形状 | 哪些项的权重加起来为 1？ |
|---|---|---|
| 一组候选项 | $(K,)$ | 这 $K$ 个候选项 |
| LM 的词表预测 | $(B,T,V)$ | 每个样本、每个位置上的 $V$ 个词表项 |
| 多头 Attention | $(B,H,T_q,T_k)$ | 每个样本、每个头、每个 query 对应的 $T_k$ 个 key 位置 |

LM 的 softmax 沿 **词表维度**，Attention 的 softmax 沿 **key 位置维度**。这些表示中两者恰好都在最后一维，但归一化对象不同；不能把整个 batch 或整条序列一起归一化。

Attention 连接：固定一个 query，其权重为 $a_{ij}$，输出是：

$$
o_i=\sum_j a_{ij}v_j,\qquad \sum_j a_{ij}=1
$$

所以 softmax 把“query 与 key 的匹配分数”转换成“取各个 value 的比例”。Causal mask 通常令禁止位置的分数为 $-\infty$，使其权重为 0；每个 query 至少需要一个允许访问的位置。

**Temperature（补充）**：用正数 $\tau$ 调节分布的集中程度。

$$
p_i=\operatorname{softmax}(z/\tau)_i
$$

- $0<\tau<1$：分数差被放大，分布更集中。
- $\tau>1$：分数差被缩小，分布更平缓。
- $\tau=1$：普通 softmax。

Temperature 不改变 logits 的大小排序；它改变各项的相对概率。存在唯一最大值时，$\tau\rightarrow0^+$ 的分布趋向只选择该最大值；不能直接令 $\tau=0$ 做除法。

训练连接（补充）：若目标类别或目标 token 的 ID 为 $y$，单个位置的交叉熵为：

$$
\mathcal L=-\log p_y
=\log\sum_j e^{z_j}-z_y
$$

提高正确目标相对于其他项的分数，可以降低损失。实际计算通常直接由 logits 求稳定的 log-softmax/交叉熵，避免先得到极小概率再取对数。

Softmax 可微，输出之间也互相影响：

$$
\frac{\partial p_i}{\partial z_j}
=p_i(\delta_{ij}-p_j)
$$

$\delta_{ij}$ 在 $i=j$ 时为 1，否则为 0。因此提高某一项分数，会提高它自己的概率，并压低其他项的概率；分布非常集中时，部分梯度可能很小。概率归一化不等于预测一定正确，也不保证概率已经校准。

以上例子、形状与训练连接是对讲义 softmax 讨论的补充解释；下面继续讨论它的数值稳定性。

#### Softmax Stability

对一个位置的 logits $z\in\mathbb{R}^{V}$：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

指数运算可能溢出，下溢也可能让直接计算的分母出现数值问题。利用对所有 logits 加同一个常数不改变概率的性质：

$$
c=\max_j z_j,\qquad
p_i=\frac{e^{z_i-c}}{\sum_j e^{z_j-c}}
$$

减去最大值后，指数输入不大于 0，且至少有一个指数项为 1（假设原 logits 有限）。这是计算数值稳定性处理；它不改变数学上的 softmax，也不能解决所有训练不稳定问题。

#### Z-Loss

输出 softmax 的归一化常数为：

$$
Z=\sum_{v=1}^{V}e^{z_v}
$$

Z-loss 在交叉熵之外加入一个辅助项，鼓励 $\log Z$ 接近 0：

$$
\mathcal L=\mathcal L_{\text{CE}}+\alpha(\log Z)^2
$$

实际训练通常对有效 token 的上述损失取平均。讲义以 PaLM 为例，其辅助项系数为 $10^{-4}$，具体训练设置不一定相同。

直觉：所有 logits 一起平移时，softmax 概率不变，但 $\log Z$ 会变化；Z-loss 对这部分尺度漂移施加约束。它与“减去最大值”不同：前者改变训练目标，后者只改变等价计算方式。

补充：计算 $\log Z$ 也要使用稳定形式 $c+\log\sum_v e^{z_v-c}$，不能把减去 $c$ 后的 log-sum-exp 直接当成原始 $\log Z$。

#### QK Norm

在 Attention 中对投影得到的 Q、K 再做归一化，以控制点积尺度：

$$
\widehat q_i=N_q(q_i),\qquad
\widehat k_j=N_k(k_j),\qquad
s_{ij}=\frac{\widehat q_i^\top\widehat k_j}{\sqrt{d_h}}
$$

这里是一种示意形式，归一化类型、维度和缩放方式随模型不同。它作用在 Q、K 上，与在整个 Attention 子层输入上做 Pre-Norm 不是同一个操作。

Q、K 幅度过大可能让 softmax 过于集中；QK Norm 从点积输入端控制尺度。讲义列举了使用 LayerNorm/RMSNorm 的模型，具体与 RoPE 的先后次序需看各模型设计，不能把所有实现视为完全相同。

#### Logit Soft-Capping

通过平滑函数限制 logits 幅度，讲义给出的形式为：

$$
\widetilde z=c\tanh(z/c),\qquad c>0
$$

对有限输入，输出位于 $(-c,c)$；小幅度输入附近接近原值，大幅度输入逐渐饱和。可用于 Attention 分数或最终输出 logits，阈值随模型和位置不同。

它会改变 softmax 输入和梯度，与减去最大值的等价变换不同。讲义也提醒计算性能与效果需要实验核对；增加稳定性操作不保证所有配置都获益。

### Attention Variants

讲义从推理资源开销介绍 MQA/GQA，再讨论稀疏、滑动窗口和交错使用不同注意力模式。重点是表达能力、缓存大小和实际运行效率的权衡。



#### KV Cache and Inference Cost

自回归生成时，下一个 token 依赖前面已生成的 token。同一条序列的普通逐 token 解码有顺序依赖；这不代表 prompt 内的计算或不同样本之间无法并行。

**KV cache**：每一层保存已处理位置的 K、V，新位置只计算自己的投影，并用当前 Q 与历史 K、V 做 Attention。标准 causal decoder 中，历史位置不会因为未来 token 到来而重新改变，因此可以复用缓存。

形状（补充，采用 head 在前的表示）：

| 张量 | 单步解码时的形状 |
|---|---|
| 当前 Q | $(B,H_q,1,d_h)$ |
| 历史 K、V cache | 各为 $(B,H_{kv},T,d_h)$ |
| 当前 Attention 分数 | $(B,H_q,1,T)$ |
| 各 query head 的输出 | $(B,H_q,1,d_h)$ |

MQA/GQA 中不同 query head 按对应关系使用共享 KV head。缓存布局也可以是 $(B,T,H_{kv},d_h)$，轴顺序不同但元素数量相同。

设有 $L$ 层、每个元素 $s$ bytes，忽略管理开销且缓存所有历史位置：

$$
\text{KV cache bytes}=2LBT H_{kv}d_hs
$$

系数 2 来自 K 和 V。缓存随 batch、上下文长度、层数和 KV 头数增长；标准缓存不需要保存所有历史 Q。

**Prefill vs Decode**：处理 prompt 时可同时计算多个 query；逐 token 解码时每条序列只有一个新 query，却仍需读取历史 KV，算术强度往往较低。这与第二讲“矩阵—向量计算容易受内存带宽限制”的直觉连接。

补充核算：对标准 MHA，单步投影等计算约为 $O(Bd_{\text{model}}^2)$，当前 query 与全部历史位置的 Attention 约为 $O(BT d_{\text{model}})$。Cache 避免重算历史投影，但不会消除对历史位置的读取和加权。

#### Multi-Query Attention (MQA)

保留多个 query head，但只使用 **一个 KV head**，让所有 query head 共享它：

$$
H_{kv}=1,\qquad H_q\text{ 可以大于 }1
$$

“一个 KV head”不表示 K、V 是一个标量；每个位置仍有 $d_h$ 维的 key 和 value。主要收益是减少 KV 投影和 cache 搬运，对自回归推理尤其有意义；Attention 仍需为各个 query head 计算分数。

讲义讨论了可能的质量代价，是否值得需要在具体模型和任务上比较。

#### Grouped-Query Attention (GQA)

把 query heads 分组，每组共享一个 KV head：

$$
1<H_{kv}<H_q
$$

常见等大小分组要求 $H_q$ 能被 $H_{kv}$ 整除，每组含 $H_q/H_{kv}$ 个 query head。

| 方式 | Query heads | KV heads | 共享关系 |
|---|---|---|---|
| MHA | $H_q$ | $H_q$ | 每个 query head 配自己的 KV head |
| GQA | $H_q$ | $H_{kv}$ | 一组 query heads 共享一个 KV head |
| MQA | $H_q$ | 1 | 全部 query heads 共享同一个 KV head |

例如 $H_q=32,H_{kv}=8$，每 4 个 query heads 共享一个 KV head。在其他条件相同、缓存不复制共享头的前提下，KV cache 是标准 MHA 的 $8/32=1/4$；不代表整个模型的计算量也降到 1/4。

GQA 提供在共享程度和表达能力之间调节的选项。讲义还提到 MLA（Multi-head Latent Attention），它通过潜在表示压缩等方式处理缓存，是另一个方向，不等同于减少 KV 头数。

#### Sparse and Sliding Window Attention

完整 Attention 对所有允许的位置计算分数，序列长度为 $T$ 时，标准 Attention 部分的计算随 $T^2$ 增长。Sparse attention 限制连接模式，只访问部分位置。

Sliding window attention 是一种局部模式：每个位置只关注附近窗口；causal 模型仍不能看未来。窗口宽度为 $w$ 时，Attention 部分的理想计算量可从 $O(BH_qT^2d_h)$ 降到 $O(BH_qTwd_h)$。

这不包括全部投影等开销，实际提速取决于 kernel。局部连接也会改变信息传播：堆叠层数可扩大感受范围，但不等于每层都能直接访问所有历史信息。

#### Interleaving Full and Local Attention

在不同层交错使用全局与局部 Attention：局部层控制开销，全局层提供直接访问远处位置的通道。

讲义以 Cohere Command A 为例，介绍每 4 层包含一个 full-attention 层，并结合全局层的 NoPE 与局部层的 RoPE。NoPE 表示不显式使用位置编码；交错比例与位置机制属于具体模型配置，并非所有 hybrid 模型的共同要求。

其他模型也有交错模式，但组合方式不同。即使部分局部层能够限制 KV cache，full-attention 层仍可能需要较长历史缓存；实际内存要按各层模式分别计算。

### Lecture Recap



一个采用 Pre-RMSNorm、串行子层的 Transformer block 可以写为：

$$
h=x+\operatorname{Attention}(\operatorname{RMSNorm}_1(x))
$$

$$
y=h+\operatorname{SwiGLU}(\operatorname{RMSNorm}_2(h))
$$

$x,h,y$ 均为 $(B,T,d_{\text{model}})$。两个 RMSNorm 分别有自己的可训练参数；Attention 中可对 Q、K 使用 RoPE，GQA 是一种可选的注意力变体。

这是 **一个 block** 的结构。完整语言模型还包括 token embedding、多层 block、最终归一化和输出词表 logits 的 LM head。

完整模型的形状主线（补充）：

| 顺序 | 表示 | 形状 |
|---|---|---|
| 1 | Token IDs | $(B,T)$ |
| 2 | Token embeddings | $(B,T,d_{\text{model}})$ |
| 3 | 多层 Transformer blocks 的输出 | $(B,T,d_{\text{model}})$ |
| 4 | 最终归一化后的隐藏状态 | $(B,T,d_{\text{model}})$ |
| 5 | LM head 输出的 logits | $(B,T,V)$ |

本讲的几个重点：

1. **架构共识**：Pre-Norm、RMSNorm、门控 FFN 和串行 block 较常见，但共同点不等于所有模型的唯一配方。
2. **位置机制**：RoPE 用旋转让点积中的位置因素通过相对距离进入；相对位置性质与长上下文效果需要区分。
3. **超参数**：FFN 比例、头数与宽深比多是经验选择，需要结合预算和系统约束。
4. **训练稳定性**：稳定 softmax、Z-loss、QK Norm、soft-capping 分别作用于不同环节。
5. **推理效率**：KV cache 避免历史重算，MQA/GQA 降低缓存开销，局部/全局交错改变 Attention 的访问模式。

Assignment 1 对照：作业的标准模型采用 Pre-RMSNorm、SwiGLU、RoPE、串行 block 和完整 causal MHA；GQA、额外稳定性方法、局部 Attention 等是本讲介绍的其他设计选择，不应自动加入作业要求。

### Further Reading

**2026 年补充（非讲义内容）**：[Attention Residuals](https://arxiv.org/abs/2603.15031) 研究用对前面层输出的注意力聚合，替代残差中固定权重的累加，并在 Kimi Linear 架构上验证。它关注随深度增长的隐藏状态尺度和各层贡献被稀释的问题。

这是一项具体的残差连接研究，不是归一化位置变体，也不足以推出“Pre-Norm 一定爆炸”或“2026 年模型已经统一采用某种替代方案”。其他架构、优化器与动态计算趋势，后续结合各自的原始论文再补充。

## Attention Alternatives and Mixture of Experts（注意力替代方案与混合专家）

资料：[Spring 2026 第四讲](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。主体按课程内容整理；张量形状、简单例子和额外推导标为补充，具体模型细节结合原始论文核对。

### Lecture Overview

本讲讨论两个不同的效率问题：

- **Attention**：上下文越长，读取和比较历史 token 的开销越大。能否压缩历史，或者只访问部分历史？
- **MLP / FFN**：能否增加模型拥有的参数，同时让每个 token 只使用其中一小部分？

Linear attention、稀疏注意力与 MoE 可以组合使用，但分别改变序列信息的处理方式和参数的使用方式。

### Attention Alternatives

先看单个 head，忽略 batch、投影和多头合并：

$$
Q,K\in\mathbb R^{T\times d_k},\qquad V\in\mathbb R^{T\times d_v}
$$

标准注意力为：

$$
Y=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+\text{mask}\right)V
\in\mathbb R^{T\times d_v}
$$

$QK^\top$ 的形状为 $(T,T)$，每个 query 都要与允许访问的 key 比较。完整或 causal attention 的注意力核心计算量为 $O(T^2(d_k+d_v))$；causal mask 减少约一半有效连接，但不改变平方增长的量级。

课程先回顾两类工具：限制连接范围的 local / sparse attention，以及优化计算和数据搬运的系统方法。补充：FlashAttention 可以避免把完整分数矩阵存到显存，并改善访存效率；完整注意力的算术计算量仍随 $T^2$ 增长。

### Linear Attention

#### Associativity and Tensor Shapes

直觉：把历史 token 的 key-value 信息先汇总成一个固定大小的矩阵，再让 query 从这个矩阵里读取。

课程从一个简化形式开始：把 softmax 等非线性操作替换为恒等映射，省略固定缩放系数，得到：

$$
Y=(QK^\top)V=Q(K^\top V)
$$

矩阵乘法满足结合律，因此可以改变计算顺序。两种顺序的形状为：

| 步骤 | 先算 $QK^\top$ | 先算 $K^\top V$ |
|---|---|---|
| 第一次乘法 | $(T,d_k)(d_k,T)\rightarrow(T,T)$ | $(d_k,T)(T,d_v)\rightarrow(d_k,d_v)$ |
| 第二次乘法 | $(T,T)(T,d_v)\rightarrow(T,d_v)$ | $(T,d_k)(d_k,d_v)\rightarrow(T,d_v)$ |
| 核心计算量 | $O(T^2(d_k+d_v))$ | $O(Td_kd_v)$ |

这里的“线性”是指 **固定特征维度时，计算量随序列长度 $T$ 线性增长**。特征维度、kernel 和序列长度仍会影响实际速度，短序列下不一定更快。

关键区别：

$$
\operatorname{softmax}(QK^\top)V\neq Q(K^\top V)
$$

去掉 softmax 改变了注意力机制，不能把标准 softmax attention 直接通过结合律变成同一个结果。简化形式中的点积也不再自动形成非负、和为 1 的注意力权重。

#### Causal Recurrent Form

对于自回归模型，第 $t$ 个位置只能汇总 $j\le t$ 的历史。令 $q_t,k_t,v_t$ 为列向量：

$$
q_t,k_t\in\mathbb R^{d_k},\qquad v_t\in\mathbb R^{d_v}
$$

维护状态矩阵：

$$
S_0=0,\qquad
S_t=S_{t-1}+k_tv_t^\top
=\sum_{j=1}^{t}k_jv_j^\top
\in\mathbb R^{d_k\times d_v}
$$

读取结果写成行向量：

$$
y_t=q_t^\top S_t
=\sum_{j=1}^{t}(q_t^\top k_j)v_j^\top
\in\mathbb R^{1\times d_v}
$$

这个展开式说明：外积 $k_jv_j^\top$ 写入一条 key-value 关联，当前 query 与各个 key 的匹配程度决定读出的 value 贡献。$S_t$ 可以理解成一个不断更新的关联记忆。

形状（补充）：$k_tv_t^\top$ 是 $(d_k,1)(1,d_v)$ 的**外积**，结果不是标量；$q_t^\top S_t$ 是 $(1,d_k)(d_k,d_v)$，结果包含 $d_v$ 个输出特征。

必须用当前前缀的 $S_t$。如果每个位置都使用包含整条序列的 $K^\top V$，就会读到未来 token。

#### Parallel, Recurrent and Chunkwise Computation

同一个 causal 计算可以从不同角度执行：

- **Parallel form**：显式构造 causal 的 token 两两关系，矩阵乘法适合 GPU，但这种直接形式的计算量仍是平方级。
- **Recurrent form**：依次更新 $S_t$。逐 token 解码时，每一步只更新固定大小的状态。
- **Chunkwise / scan methods**：把序列分块，结合块内并行与块间状态传递，提高训练时的硬件利用率。

课程用“并行训练、循环推理”解释两种视角。补充：线性注意力并不要求训练必须采用平方级算法，具体训练算法可以利用 scan 或分块结构；固定大小的推理状态也不代表反向传播完全不需要保存中间信息。

对简单的多头循环形式，状态形状可以是 $(B,H,d_k,d_v)$，不随已处理的上下文长度增长。代价是历史信息被压缩到状态中，不能像完整注意力那样直接访问每个历史 token 的独立 KV。

#### Kernelized and Normalized Variants

补充：线性注意力不只有“直接删除 softmax”这一种。可以用特征映射 $\phi$ 定义另一种相似度，再加入归一化。例如：

$$
A_t=\sum_{j\le t}\phi(k_j)v_j^\top,\qquad
b_t=\sum_{j\le t}\phi(k_j)
$$

$$
y_t=\frac{\phi(q_t)^\top A_t}{\phi(q_t)^\top b_t}
$$

若 $\phi(x)\in\mathbb R^r$，则 $A_t$ 为 $(r,d_v)$，$b_t$ 为 $(r)$。非负特征及有效分母可以提供归一化权重；相似度、表达能力和数值处理仍取决于具体设计。这不是一般情况下与 softmax 完全等价的变换。

参考：[Transformers are RNNs](https://arxiv.org/abs/2006.16236)。

### Gated Recurrent Models

#### Decay and Input-Dependent Gates

简单累加会持续保留旧信息。加入遗忘系数后：

$$
S_t=\gamma_tS_{t-1}+k_tv_t^\top
$$

$\gamma_t$ 控制旧状态保留多少。课程先介绍衰减形式，再用输入相关的 $\gamma_t=f(x_t)$ 说明选择性记忆：不同 token 可以触发不同的保留程度。

课程对 Mamba-2 的简化视角还包含直接通路：

$$
y_t=q_t^\top S_t+v_t^\top D
$$

$D$ 表示直接通路的参数。这个公式用于连接状态空间模型与线性注意力的直觉；完整 Mamba-2 还包含投影等结构，不能只用一个遗忘门概括其全部设计。

#### Gated DeltaNet

直觉：写入新关联前，先削弱旧记忆中沿当前 key 方向的内容，再写入新的 value。

按课程采用的状态方向，公式为：

$$
S_t=\gamma_t\left(I-\beta_tk_tk_t^\top\right)S_{t-1}
+\beta_tk_tv_t^\top
$$

$$
y_t=q_t^\top S_t
$$

| 量 | 形状或作用 |
|---|---|
| $S_t$ | $(d_k,d_v)$，关联记忆 |
| $I$、$k_tk_t^\top$ | $(d_k,d_k)$，作用在 key 方向上 |
| $\gamma_t$ | 旧状态的整体衰减门 |
| $\beta_t$ | 当前方向的更新强度 |
| $k_tv_t^\top$ | $(d_k,d_v)$，新写入的关联 |

两个门通常由输入计算，分别控制整体遗忘和当前关联的修改。$\beta_t=0$ 时不写入当前关联，但状态仍会乘 $\gamma_t$；只有同时 $\gamma_t=1$ 时才完全不变。

补充推导：

$$
S_t=\gamma_tS_{t-1}
+\beta_tk_t\left(v_t^\top-\gamma_tk_t^\top S_{t-1}\right)
$$

括号里是“目标 value”与“衰减后的旧记忆在当前 key 上读出的 value”之间的差。更新沿 $k_t$ 的方向修正这个差，因此叫 delta rule。

若 $\|k_t\|_2=1$ 且 $\beta_t=1$，$I-k_tk_t^\top$ 会投影掉当前 key 方向。二维例子中，$k_t=(1,0)^\top$、$\gamma_t=\beta_t=1$ 时，状态第一行被新 value 替换，第二行保留。一般门值下是部分削弱；没有单位长度等条件时，不能直接称为精确的正交擦除。

课程把它与 fast weights / test-time training 联系起来：状态可视为在序列内更新的快速记忆。这里更新的是状态，不意味着每个推理 token 都对整个语言模型参数做一次训练。

参考：[Gated Delta Networks](https://arxiv.org/abs/2412.06464)。原论文的状态矩阵方向与这里互为转置，比较公式时要先核对约定。

### Hybrid Architectures

把循环状态层和完整注意力层交错使用：前者控制长上下文成本，后者保留直接检索历史 token 的通道。

课程列举的配置示例：

| 模型例子 | 循环层类型 | 循环层与完整注意力层的示意比例 |
|---|---|---|
| MiniMax-Text-01 / M1 | Linear attention | 7 : 1 |
| Nemotron 3 | Mamba | 约 3 : 1 |
| Qwen3-Next / Qwen3.5 | Gated DeltaNet | 约 3 : 1 |

这些是课件讨论的模型配置，不是所有 hybrid 架构的统一比例。课程也提醒：跨模型比较同时涉及数据、训练预算和其他架构差异，严格控制变量的消融实验仍有限。

即使循环层只保留固定状态，混合模型中的完整注意力层仍需要自己的 KV cache，不能把整个 hybrid 模型描述成完全没有历史缓存。

### Sparse Attention and DSA

#### Top-K Selection

Top-k 表示取分数最高的 $k$ 项。例子（补充）：历史 token A、B、C、D 的分数为 $[0.2,0.9,0.4,0.8]$，Top-2 选择 B、D。

在稀疏注意力中，$k$ 是选中的历史 token 数量；在 MoE 中，$k$ 是选中的专家数量。它们属于不同的选择集合，也与 key 矩阵 $K$ 的转置 $K^\top$ 无关。

#### DeepSeek Sparse Attention

DSA 使用一个轻量的 **lightning indexer** 为历史位置打分，选出相关位置，再对这些位置做主注意力计算。

设第 $t$ 个 query 选择的位置集合为 $\mathcal I_t\subseteq\{1,\ldots,t\}$，则主注意力可以示意为：

$$
y_t=\sum_{j\in\mathcal I_t}a_{tj}v_j^\top,\qquad
a_{tj}=\frac{\exp(s_{tj})}{\sum_{\ell\in\mathcal I_t}\exp(s_{t\ell})}
$$

这里仍对选中的位置使用 softmax；indexer 的打分与主注意力的 $s_{tj}$ 不必相同。DSA 用“选择少量历史位置”减少主注意力开销，不把所有历史压成一个固定状态。

复杂度要分开计算：

- 固定其他维度时，每个 query 最多访问 $k$ 个位置，主注意力约为 $O(Tk)$。
- DeepSeek-V3.2 的轻量 indexer 仍对各个允许的历史位置打分，完整序列的索引计算仍包含平方级工作，只是常数小得多。
- 单步解码时，indexer 对历史的扫描随历史长度增长；Top-k 选择、KV 读取和 kernel 也有开销。

因此不能只凭主注意力的 $O(Tk)$，就把整个 DSA 的总复杂度写成 $O(Tk)$。实际收益来自把昂贵计算集中在少数位置，以及更便宜的索引路径。

课程还介绍 sparse adaptation：在已有 dense 模型上训练索引器并进行后续适配，而不是必须从零预训练稀疏模型。具体流程需要继续训练，不能直接删去连接并假设效果不变。[DeepSeek-V3.2 技术报告](https://arxiv.org/html/2512.02556v1)

#### Comparing the Mechanisms

| 机制 | 如何处理历史 | 推理解码时的历史存储 |
|---|---|---|
| Full attention | 当前 query 直接访问所有允许的历史位置 | KV 随上下文增长 |
| 简单循环线性注意力 | 把历史关联汇总成状态，再从状态读取 | 固定大小状态 |
| DSA | 索引后，只在选中的位置做主注意力 | 仍需保留可检索的历史 KV 等信息 |
| Hybrid | 不同层采用不同机制 | 按各层的状态与缓存分别核算 |

### Mixture of Experts

#### More Parameters with Sparse Activation

直觉：准备多个可学习的 FFN，让每个 token 只调用其中少数几个。模型拥有的参数多，但每次处理一个 token 时只使用部分专家参数。

常见 Transformer MoE 主要把原来的一条 FFN 路径替换成多个 FFN 和一个 router；Attention 子层可以保持原来的设计。“专家”表示不同的参数模块，不要求人工指定哪个处理数学、哪个处理中文。

补充形状：把一批有效 token 展平成 $M$ 个位置，忽略 padding 时 $M=BT$：

| 量 | 形状 |
|---|---|
| Token 表示 $U$ | $(M,d_{\text{model}})$ |
| Router 分数 | $(M,N)$，$N$ 为候选专家数 |
| 每个 token 的 Top-k 专家索引与权重 | 各为 $(M,k)$ |
| 专家 $E_i$ 收到的输入 | $(m_i,d_{\text{model}})$ |
| 专家 $E_i$ 的输出 | $(m_i,d_{\text{model}})$ |
| 按原位置合并的结果 | $(M,d_{\text{model}})$ |

$m_i$ 是分配给专家 $i$ 的 token 数，可能不同；一个 token 选择多个专家时，会在多个专家的输入里出现。专家内部中间维度可以不同于原 dense FFN，但输入、输出宽度需要匹配残差流。

#### Total Parameters, Active Parameters and Cost

设每个 routed expert 有 $P_e$ 个参数，$N$ 个专家中每个 token 激活 $k$ 个；用 $P_{\text{base}}$ 表示 Attention、共享专家等总会使用的参数，$P_r$ 表示 router 参数。示意核算为：

$$
P_{\text{total}}\approx P_{\text{base}}+NP_e+P_r
$$

$$
P_{\text{active per token}}\approx P_{\text{base}}+kP_e+P_r
$$

例如 $N=64,k=2$，routed expert 部分拥有 64 份参数，每个 token 只用其中 2 份。固定专家大小和 $k$，增加 $N$ 可以增加容量而基本维持专家计算量。

“计算量几乎不变”需要限定比较条件：router 仍要打分，token 分发与合并也有开销；改变 $k$ 或专家宽度会改变 FFN 计算量。激活参数量接近，并不保证 FLOPs、吞吐量或延迟完全相同。

所有专家参数仍需要存储并可被访问，可以分布在多张设备上。训练还要存储相关梯度和优化器状态；因此稀疏激活主要节省每个 token 的计算，不能把总参数的存储成本一并消除。

课程用 FLOP-matched 实验介绍 MoE 的效果优势，并强调系统与训练复杂度。许多模型采用 MoE，但不能据此推断所有主流模型都已经改成 MoE。

### Routing and Expert Design

#### Routing Choices

课程区分几种分配方式：

- **Token choice**：每个 token 选择专家；常见方案是 Top-k。
- **Expert choice**：每个专家选择自己接收的 token，可控制专家负载，但需要处理每个 token 被接收的次数。
- **Global assignment**：结合整批 token 的分数与容量约束做匹配。

Router 可以是可学习的打分器，也有随机、哈希或其他分配方法。不同方法需要同时考虑模型效果、分配成本和负载。

#### Top-K Routing and Weighted Outputs

令单个 token 表示 $u_t\in\mathbb R^d$，专家打分向量为：

$$
z_{i,t}=u_t^\top e_i,\qquad e_i\in\mathbb R^d
$$

选择集合 $\mathcal E_t$ 后，MoE 分支输出为：

$$
m_t=\sum_{i\in\mathcal E_t}g_{i,t}E_i(u_t)\in\mathbb R^d
$$

$g_{i,t}$ 是标量，乘的是专家整个输出向量；不同专家输出再逐元素相加。若这条分支位于 Pre-Norm block，可写成 $x_t+m_t$，其中 $u_t=N(x_t)$，残差仍从 $x_t$ 传递。

#### Softmax Before or After Selection

两种常见权重定义：

**先在全部专家上 softmax，再保留 Top-k**：

$$
p_{i,t}=\frac{e^{z_{i,t}}}{\sum_{j=1}^{N}e^{z_{j,t}}},\qquad
g_{i,t}=p_{i,t}\mathbf 1[i\in\mathcal E_t]
$$

如果不再次归一化，被选中权重的总和通常小于 1。

**只在选中的专家上 softmax**：

$$
g_{i,t}=\frac{e^{z_{i,t}}}{\sum_{j\in\mathcal E_t}e^{z_{j,t}}},\qquad i\in\mathcal E_t
$$

此时选中权重的总和为 1。补充：如果第一种方法选中后再除以选中概率的总和，就与第二种方法等价；不能只凭“先后次序”判断两种实现一定不同。Softmax 对同一组 logits 保持排序，也不会改变无额外约束时的 Top-k 集合。

归一化保证的是权重和，不足以单独保证训练更稳定或效果更好。

#### DeepSeek-V3 Routing

具体模型需区分版本。DeepSeek-V3 使用 sigmoid 亲和分数：

$$
s_{i,t}=\operatorname{sigmoid}(u_t^\top e_i)
$$

选中专家后，用原始亲和分数归一化：

$$
g_{i,t}=\frac{s_{i,t}}{\sum_{j\in\mathcal E_t}s_{j,t}},\qquad i\in\mathcal E_t
$$

其负载均衡 bias 和节点限制还会参与决定选择集合。这里的“sigmoid 后归一化”与“对选中 logits 做 softmax”是不同函数；按技术报告核对，不能简单把 V3 归到后者。[DeepSeek-V3 技术报告](https://arxiv.org/html/2412.19437v2)

#### Fine-Grained and Shared Experts

**Fine-grained experts**：用更多、内部宽度更小的专家提供更细的组合。单个专家变小后，可以在类似的激活计算预算内选择更多专家；只增加专家数而不改尺寸或 $k$，是另一种容量变化。

**Shared experts**：一部分专家始终运行，另一部分按 token 路由。示意公式为：

$$
y_t=x_t+\sum_{s=1}^{N_s}E_s^{\text{shared}}(u_t)
+\sum_{i\in\mathcal E_t}g_{i,t}E_i^{\text{routed}}(u_t)
$$

直觉上，共享路径可以承载常用处理，路由路径提供可选择的容量。但这是设计动机，不代表训练后必然形成可解释的知识分工。

课程介绍了 DeepSeek 的相关消融，也对照 OLMoE 的结果：细粒度专家与共享专家的收益需要分开评估，共享专家并非在所有实验中都有增益。

### Training MoEs

#### Discrete Selection and Gradients

Top-k 决定的是离散集合，不能直接把“选择第几个专家”当作平滑函数求导。但被选中专家的运算及连续门控权重仍可参与反向传播；不是整个 MoE 都不能训练。

课程介绍的处理思路包括：强化学习式路由、给 router 分数加入噪声，以及负载均衡辅助目标。强化学习式估计可能有较大方差；噪声有助于探索，但额外随机性和稳定性仍需实验评估。

例如早期 noisy gating 加高斯噪声，Switch 曾采用乘性 jitter，后续工作也有去掉该噪声的配置。它们是具体训练选择，不是所有 MoE 的必备步骤。

#### Heuristic Load Balancing Loss

问题：如果大量 token 都流向同一批专家，热门专家会过载，其他专家缺少训练数据；多设备执行时，还会让某些设备繁忙、其他设备等待。

以 **Switch 的 Top-1 路由**为例，一批有 $M$ 个有效 token、$N$ 个专家，router 概率为 $p_{i,t}$：

$$
f_i=\frac1M\sum_{t=1}^{M}\mathbf 1\left[\arg\max_j p_{j,t}=i\right]
$$

$$
P_i=\frac1M\sum_{t=1}^{M}p_{i,t}
$$

- $f_i$：实际派给专家 $i$ 的 token 比例，来源于离散选择。
- $P_i$：router 给专家 $i$ 的平均概率，仍是连续可微的量。

辅助损失为：

$$
\mathcal L_{\text{balance}}=\alpha N\sum_{i=1}^{N}f_iP_i
$$

再与语言模型主损失相加。$\alpha$ 控制这项约束的强度。

补充例子：若 $f_i=P_i=1/N$，则 $\mathcal L_{\text{balance}}=\alpha$；若两者都集中在同一个专家上，则为 $\alpha N$。这说明它惩罚常见的集中分配，但这是启发式目标，不保证每一批都严格均匀，也不能仅凭这两个例子证明全局最优解。

把当前的离散统计 $f_i$ 视为常量时：

$$
\frac{\partial\mathcal L_{\text{balance}}}{\partial p_{i,t}}
=\frac{\alpha N}{M}f_i
$$

负载大的专家对应较强的概率惩罚，梯度再经 router 传回参数。Top-k 大于 1 时，分配次数和比例的定义可能改变，不能直接套用 Top-1 的归一化约定。[Switch Transformers](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

#### Expert and Device Balancing

专家负载均衡与设备负载均衡有关，但不是同一个统计量：一个设备可能放多个专家，token 还可能跨节点路由。

课程介绍专家级、设备级均衡及限制可访问节点的设计。即使专家接收 token 数接近，通信距离、消息数量和设备计算时间也未必一致；系统约束可以影响路由选择。

#### Adaptive Expert Biases

DeepSeek-V3 给专家设置动态 bias $b_i$，选择时考虑 $s_{i,t}+b_i$：过载专家的 bias 下调，欠载专家的 bias 上调，从而改变它们被选中的机会。

Bias 用于选择集合；输出混合权重仍由原始 $s_{i,t}$ 归一化，不能把调整后的分数直接代入门控权重公式。

这种主要均衡方式称为 auxiliary-loss-free balancing，但 V3 仍保留较小的序列级辅助均衡项。“主要均衡不依赖该辅助损失”不等于“训练中没有任何辅助损失”。

### MoE Systems and Stability

#### Expert Parallelism and All-to-All

把不同专家放到不同设备上，是 expert parallelism。一次 MoE 计算通常需要：按路由结果整理 token → 发给专家所在设备 → 专家计算 → 把结果送回原位置并合并。

跨设备分发与返回常涉及 All-to-All 通信。一个 token 选择多个专家时，可能产生多份分发；共享专家的放置方式也会影响通信。

因此相同激活 FLOPs 的 MoE 与 dense 模型，实际速度可能不同。通信带宽、负载不均、专家收到的小 batch 和 kernel 效率都会影响吞吐量。

课程还介绍压缩通信中的表示宽度等方向。目的在于减少传输，而非仅减少模型的总参数量。

#### Capacity, Padding and Token Dropping

固定容量实现会给每个专家预留接收位置。对 Top-1，一种典型容量定义为：

$$
C\approx\left\lceil\text{capacity factor}\cdot\frac MN\right\rceil
$$

补充：Top-k 下平均分配次数约为 $Mk/N$，容量公式的具体约定要看实现。容量过大可能浪费 padding 计算，过小则容易溢出。

**Token dropping** 通常表示某次专家分支不处理超出容量的 token；残差路径仍可保留该 token 的表示，不是把这个 token 从输入文本永久删除。Top-k 时也可能只丢弃某个专家分配。

容量基于整批分配时，同一个 token 是否溢出可能受同批其他 token 影响，产生 batch 相关的行为。这个问题依赖容量、丢弃策略与实现，不能推断所有 MoE 都必然丢 token。

课程介绍 MegaBlocks 一类块稀疏矩阵乘法：适应不同专家的 token 数，减少固定容量带来的 padding / dropping 权衡。DeepSeek-V3 的报告也采用不丢 token 的训练与推理策略，说明 dropping 不是 MoE 的必要条件。

#### Router Precision and Z-Loss

Router 的小幅数值误差可能改变 Top-k 集合。课程介绍对路由相关计算选择性使用 FP32，以改善稳定性；不意味着所有专家运算也必须用 FP32。

另一种稳定性工具是 **router z-loss**。令 $z_{i,t}$ 为 softmax router 的原始 logits，示意为：

$$
\mathcal L_z=\frac{\lambda}{M}\sum_{t=1}^{M}
\left(\log\sum_{i=1}^{N}e^{z_{i,t}}\right)^2
$$

它约束 router logits 的尺度漂移；load balancing loss 关注分配负载，两者作用不同。计算 log-sum-exp 时仍要做数值稳定处理。

#### Fine-Tuning and Upcycling

课程讨论小规模微调数据下 MoE 的过拟合风险，以及只微调部分非 MoE 参数等应对方法。适合的方法取决于数据与任务，不能据此规定所有 MoE 微调都要冻结专家。

**Upcycling**：用已训练的 dense 模型初始化 MoE，再继续训练，让专家与 router 学会分工。可以复制或重新组织 FFN 参数，也可以配合扰动、细粒度专家等设计。

补充直觉：若多个专家一开始完全相同，且选中门控权重之和为 1，它们的加权输出与单个相同专家一致。仅复制参数不会自动产生不同能力，后续训练和路由仍然重要。

### DeepSeek as a Case Study

#### Expert Configurations

课程用不同版本说明专家粒度和路由设计的演化：

| 版本 | Routed experts | 每个 token 激活的 routed experts | Shared experts |
|---|---|---|---|
| DeepSeekMoE（V1） | 64 | 6 | 2 |
| DeepSeek-V2 | 160 | 6 | 2 |
| DeepSeek-V3 | 256 | 8 | 1 |

Shared experts 不包含在 routed Top-k 的数字中。V3 技术报告给出的模型规模约为总参数 671B、每个 token 激活 37B；不能把 routed experts 的激活比例直接当作整个模型的参数比例。

#### Multi-Head Latent Attention (MLA)

MLA 处理 KV cache 大小，MoE 处理 FFN 参数的稀疏激活，两者可以同时存在。

直觉：多个 head 的内容 K、V 由一个较低维的 latent 表示生成。对隐藏状态 $h_t\in\mathbb R^d$：

$$
c_t^{KV}=W^{DKV}h_t\in\mathbb R^{d_c}
$$

$$
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

补充形状：$W^{DKV}$ 为 $(d_c,d)$；若内容 key、value 各有 $H$ 个宽度为 $d_h$ 的 head，上投影矩阵可示意为 $(Hd_h,d_c)$，结果再按 head 拆分。

课程用矩阵结合律说明内容 key 的上投影可以吸收到 query 一侧。对一个内容 key head：

$$
q^\top(W^{UK}c)=\left((W^{UK})^\top q\right)^\top c
$$

这样可以在低维 latent 空间计算内容匹配。Value 的线性上投影也可与加权汇总结合，避免为每个历史位置保存完整展开的 KV。

但 RoPE 带来的旋转依赖位置，不能把所有旋转都吸收到一个固定投影里。MLA 因此分出位置相关的 key 分量 $k_t^R$，与内容分量组合使用。

实际需要缓存 **$c_t^{KV}$ 和位置 key $k_t^R$**。若位置分量宽度为 $d_R$，每个历史位置约保存 $d_c+d_R$ 个元素；缓存仍随 $T$ 增长，只是每个位置更小。Query 压缩则主要帮助训练时的激活内存，不是把历史 Q 加入 cache。

“DeepSeek Attention”不是这里某一种精确算法的名字：**MLA 压缩 KV 表示，DSA 选择主注意力访问的位置**；NSA 是另一项稀疏注意力设计，也不能与 DSA 直接当作同一个机制。

#### Multi-Token Prediction (MTP)

通常在位置 $t$ 预测 $x_{t+1}$。MTP 增加辅助预测模块，让训练表示还受到后续 token 预测目标的监督。

课程以 DeepSeek-V3 为例：主路径预测 $x_{t+1}$，一个额外模块预测 $x_{t+2}$。该辅助模块把位置 $t$ 的主模型表示与训练时真实 $x_{t+1}$ 的 embedding 结合，经过投影及额外 Transformer block，再预测 $x_{t+2}$。

补充形状：两个宽度为 $d$ 的表示拼接后为 $2d$，投影回 $d$；输出 head 仍给出词表大小 $V$ 的 logits。辅助路径训练时使用它所在预测步骤之前的真实 token，不意味着主路径可以看到未来答案。

MTP 增加训练监督，也增加训练计算。推理时可以移除辅助模块；若用于 speculative decoding，则由辅助模块提出候选、主模型验证，不能把多个未来位置当作互相独立并直接接受。

### Lecture Recap

1. **Linear attention**：更换注意力机制后利用结合律，以固定状态汇总历史；它与标准 softmax attention 不完全等价。
2. **Gated recurrent models**：门控决定保留和更新什么，delta rule 沿当前 key 方向修正记忆；混合架构保留完整注意力层的直接检索能力。
3. **Sparse attention**：减少主注意力访问的位置，但索引、缓存和实际 kernel 开销仍需计算。
4. **MoE**：增加总参数容量，让每个 token 只激活少数专家；router、负载、存储和通信决定实际效率。
5. **Model combinations**：MoE、MLA、DSA、MTP 分别涉及专家参数、KV 表示、历史位置选择与训练预测目标，不是同一项优化的不同名字。



