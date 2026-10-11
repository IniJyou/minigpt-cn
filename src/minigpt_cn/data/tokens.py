"""读取已有 pilot 的原始 uint16 token；不下载、不清洗、不重新编码。"""

from pathlib import Path

import numpy as np
import torch


def read_tokens(path, vocab_size):
    path = Path(path)
    if path.stat().st_size < 4 or path.stat().st_size % 2:
        raise ValueError("数据必须包含至少两个原始 uint16 token，且字节数为偶数")
    with path.open("rb") as file:
        if file.read(6) == b"\x93NUMPY":
            raise ValueError("需要原始 uint16 二进制，不能使用带文件头的 .npy")
    # 固定小端 uint16；memmap 只映射文件，不把整份训练语料复制进内存。
    tokens = np.memmap(path, dtype="<u2", mode="r")
    if int(tokens.max()) >= vocab_size:
        raise ValueError("数据 token ID 超出模型词表")
    return tokens


def get_batch(dataset, batch_size, max_seq_length, device, *, generator):
    # 原作业：x/y 都为 [batch_size,max_seq_length]，y 比 x 向后移动一个 token。
    n = len(dataset)
    if n <= max_seq_length:
        raise ValueError("训练集长度必须大于 context_length")
    if batch_size <= 0 or max_seq_length <= 0:
        raise ValueError("batch_size 和 max_seq_length 必须为正数")
    # 留出 max_seq_length+1 个位置，最后一个合法起点也要可以采样到。
    # 项目适配：使用独立 CPU Generator，避免验证/模型初始化改变后续 batch。
    starts = torch.randint(0, n - max_seq_length, (batch_size,), generator=generator)
    x = torch.stack(
        [torch.from_numpy(dataset[i : i + max_seq_length].astype(np.int64)) for i in starts]
    )
    y = torch.stack(
        [torch.from_numpy(dataset[i + 1 : i + max_seq_length + 1].astype(np.int64)) for i in starts]
    )
    return x.to(device), y.to(device)
