"""从固定M3原始记录生成M4图与样例页；不导入模型或改写历史产物。

绘图依赖已有matplotlib；模型运行环境不需要它。默认拒绝已有输出，
复建时用--output-root指定新的目录，不覆盖已验收图与文档。
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCES = {
    "logs/m3_training_report/m3_pretrain_100m_return_v1/metrics.jsonl": "2b1831437903a1204ac5b0be0f9b2ccbad25e4111d17fc7e1e8ca24a093d51bd",
    "logs/m3_training_report/training_report.json": "0d24ba7ff92ed11889262d6244c1f15ba467e04dd567f785c8c814e8363cf761",
    "logs/m3_final_local_evaluation_v1/evaluation_report.json": "9c6c731a8b80afa64531bca2bbeed6c098850a7b810f536e1f7d5af10eef031f",
    "logs/m3_final_local_evaluation_v1/generations.jsonl": "0c7d150f3e2b4863cdbbc1f78bed28a8cefbc3aa6826890a331821a75fd4aecf",
    "logs/m3_contiune_2000_report/metrics.jsonl": "d0c714b1c0b11325744266a13b77b653da13a483e7f86e46f604d4f9eadfe323",
    "logs/m3_continuation_comparison_v1/comparison.json": "53530af73af441ce05a0e0379b8f7e39f07931068709417610eeadf6778e9eb4",
    "logs/m3_continuation_comparison_v1/semantic_review.json": "3bd39a0886778344940156b2d2e9b3d975e71caae802afddb22302733f95b85c",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def build_chart(path, metrics, continued):
    # matplotlib只用于离线图形，不属于模型/训练依赖，不改变项目虚拟环境。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 11, "svg.fonttype": "path",
                         "path.simplify": False, "svg.hashsalt": "minigpt-cn-m4-fixed"})
    train = [row for row in metrics if row["event"] == "train"]
    valid = [row for row in metrics if row["event"] == "validation"]
    extension = [row for row in continued if row["event"] == "validation"]
    assert len(train) == 611 and len(valid) == 26 and len(extension) == 6
    assert all(math.isfinite(row["loss"]) for row in train + valid + extension)
    assert valid[0]["step"] == 0 and valid[-1]["step"] == 12208
    assert extension[0]["step"] == 12208 and extension[-1]["step"] == 14208
    assert valid[-1]["loss"] == extension[0]["loss"]
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), gridspec_kw={"width_ratios": [1.24, 1]})
    fig.patch.set_facecolor("#f7f9fc")
    fig.suptitle("MiniGPT-CN 33M · 固定训练与有界续训", fontsize=20, fontweight="bold", color="#152d45", x=0.06, ha="left")
    for ax in axes:
        ax.set_facecolor("#ffffff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bbc7d6")
        ax.tick_params(colors="#536579")
        ax.grid(axis="y", color="#e6ecf3", linewidth=0.8)
        ax.set_xlabel("已完成更新步数")
        ax.set_ylabel("Next-token NLL")
    axes[0].plot([r["step"] for r in train], [r["loss"] for r in train], color="#2383d1", alpha=0.35, linewidth=0.9,
                 label="Train：611条原始minibatch记录")
    axes[0].plot([r["step"] for r in valid], [r["loss"] for r in valid], color="#c56818", marker="o", markersize=3.5,
                 linewidth=1.8, label="Valid：26次完整验证")
    axes[0].set_title("原训练 · 100,007,936处理tokens", loc="left", pad=16, fontsize=13)
    axes[0].set_xlim(0, 12600)
    axes[0].set_ylim(3.3, 10.1)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    axes[0].annotate("9.7741 → 3.9629", xy=(12208, valid[-1]["loss"]), xytext=(7200, 5.4),
                     color="#9a4e0e", arrowprops={"arrowstyle": "->", "color": "#9a4e0e"}, fontsize=11)
    late = [r for r in valid if r["step"] >= 6000]
    axes[1].plot([r["step"] for r in late], [r["loss"] for r in late], color="#c56818", marker="o", markersize=4,
                 linewidth=1.8, label="原训练完整valid")
    axes[1].plot([r["step"] for r in extension], [r["loss"] for r in extension], color="#7755b7", marker="D", markersize=4,
                 linewidth=1.8, label="追加2000步完整valid")
    axes[1].axvline(12208, color="#9baabb", linewidth=1, linestyle="--")
    axes[1].set_title("后段valid放大 · 终点预先固定", loc="left", pad=16, fontsize=13)
    axes[1].set_xlim(5800, 14500)
    axes[1].set_ylim(3.91, 4.33)
    axes[1].legend(frameon=False, loc="upper right", fontsize=9)
    axes[1].annotate("12208步\n3.962903", xy=(12208, valid[-1]["loss"]), xytext=(9500, 4.03),
                     color="#9a4e0e", arrowprops={"arrowstyle": "->", "color": "#9a4e0e"}, fontsize=10)
    axes[1].annotate("14208步\n3.945888", xy=(14208, extension[-1]["loss"]), xytext=(12400, 4.11),
                     color="#69469f", arrowprops={"arrowstyle": "->", "color": "#69469f"}, fontsize=10)
    fig.text(0.06, 0.055, "原始记录点直接连线，未平滑；Train为随机窗口minibatch，Valid每次覆盖297389个目标。", color="#536579", fontsize=10)
    fig.text(0.06, 0.018, "图中是云端训练过程BF16验证；本地独立评估单独列表。降低loss未带来通过门槛的连贯性改善。", color="#536579", fontsize=10)
    fig.subplots_adjust(left=0.065, right=0.975, top=0.78, bottom=0.19, wspace=0.26)
    fig.savefig(path / "training-loss.png", dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(path / "training-loss.svg", facecolor=fig.get_facecolor(), metadata={"Date": None, "Description": "M3 frozen training records; source SHA-256 values are bound in scripts/build_showcase.py"})
    svg_path = path / "training-loss.svg"
    # Matplotlib的路径属性常带行尾空格；仅规范文本空白，保留数值/路径与图形。
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip(" \t") for line in svg_text.splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    return {"train_points": len(train), "valid_points": len(valid), "continuation_valid_points": len(extension),
            "training_valid_first": valid[0]["loss"], "training_valid_last": valid[-1]["loss"],
            "continuation_valid_last": extension[-1]["loss"], "smoothing": False}


def build_page(evaluation, comparison, semantic, samples):
    assert len(samples) == 12
    labels = ["geography", "history", "astronomy", "technology", "biology", "traditional_chinese"]
    assert [(r["prompt_id"], r["decoding"]) for r in samples] == [(name, mode) for name in labels for mode in ("greedy", "sampling")]
    valid, test = evaluation["evaluations"]["valid"], evaluation["evaluations"]["test"]
    text = [
        "# MiniGPT-CN 33M：从实现到真实训练", "",
        "从零实现中文Decoder-only Transformer，完成BPE、固定训练、保存/恢复、独立评估与受控优化。工程链路已完成；生成仍有循环、离题与句意混乱，本页原样展示结果。", "",
        "[模型卡](../MODEL_CARD.md) · [本地复现](local-reproduction.md) · [M4验收入口](m4-acceptance.md) · [M3完整实验报告](m3-experiment-report.md)", "",
        "## 项目成果", "",
        "| 范围 | 已完成 |", "| --- | --- |",
        "| 模型核心 | 8层/8头，33104384参数，RMSNorm/RoPE/SwiGLU，权重共享、dropout |",
        "| Tokenizer | byte-level BPE增量训练、保存加载；16000词表含EOS256，15743 merges |",
        "| 训练与评估 | AdamW、混合精度、完整valid尾块覆盖、周期保存、优化器/采样器/RNG恢复 |",
        "| 推理 | greedy、temperature/top-p、EOS、可选KV Cache与重复惩罚，滑窗重建 |",
        "| 实际运行 | 原12208步/100007936处理tokens，另做固定2000步续训对照 |",
        "| 当前边界 | 中文文本续写实验；未做指令微调，连贯性与事实性未验收 |", "",
        "最终train为73520篇/48187994个token位置，来源固定中文Wikipedia版本；约2.075倍算术曝光，不是1亿唯一语料token。6个源分片已按有界规则处理，不代表逐文档事实、语义或许可批准。", "",
        "## 模型结构", "", "```mermaid", "flowchart LR",
        '    I["token IDs · B×S"] --> E["共享词嵌入 · V×512"]',
        '    E --> D["8个Decoder Block\nRMSNorm → RoPE Attention → SwiGLU\n残差与训练dropout"]',
        '    D --> N["RMSNorm"]', '    N --> H["共享LM head"]', '    H --> O["logits · B×S×16000"]', "```", "",
        "S最多512；输入与目标只右移一次。结构与计算入口见[运行参考](runtime-reference.md)。", "",
        "## 真实训练曲线", "", "![原12208步训练和固定2000步续训的真实loss曲线](assets/training-loss.svg)", "",
        "[PNG图片](assets/training-loss.png) · [SVG矢量图](assets/training-loss.svg)", "",
        "左图为611条原始train minibatch loss与26次完整valid；右图放大6000步以后的完整valid，并展示追加2000步的6次valid。每次valid覆盖297389目标，未平滑或挑选点。Train/valid上下文与统计范围不同，曲线间距不直接当作泛化差距。", "",
        "图中云端原final valid=3.962902672139143，续训final=3.9458879316974587。下面是同一PC独立评估，精度/环境差异如实区分。", "",
        "## 原final独立评估", "", "| 集合 | 完整目标 | loss | perplexity |", "| --- | ---: | ---: | ---: |",
        f"| valid | {valid['targets']} | {valid['loss']} | {valid['ppl']} |",
        f"| test | {test['targets']} | {test['loss']} | {test['ppl']} |", "",
        "Windows/Python3.12.4/PyTorch2.14.0+cu126，RTX4060 Laptop，BF16；不是本轮重跑的分数。test用于原模型一次报告，没有用于选权重、延长或优化对照。PPL只在本数据/词表/协议内解释，不与不同评测直接排名。", "",
        "## 固定续写：全部12条", "",
        "按原协议顺序列出6个提示词×greedy/sampling全部结果。每条保留完整续写、原长度和停止原因；没有挑最好结果。sampling为seed42、temperature0.8/top-p0.9；两模式均BF16、无缓存、最多128新token，重复惩罚保持默认1.0。", "",
    ]
    for i, row in enumerate(samples, 1):
        assert "```" not in row["continuation"] and "</details>" not in row["continuation"]
        text += ["<details>", f"<summary>{i:02d} · {row['prompt_id']} / {row['decoding']} · {row['generated_tokens']} token · {row['stop_reason']}</summary>", "",
                 f"提示词：`{row['prompt']}`", "",
                 f"实际新增{row['generated_tokens']} token，停止={row['stop_reason']}，非法UTF-8={row['continuation_utf8_invalid']}。", "",
                 "```text", row["continuation"], "```", "", "</details>", ""]
    text += ["这12条续写存在明显重复/离题/句意混乱，生成质量未通过。折叠仅为方便阅读，不隐藏失败样例；原始ID与本地完整记录仍保留在Git忽略的logs/m3_final_local_evaluation_v1。", "",
             "## 有界优化：如实保留未改善结论", "",
             "先做原始预测、缓存与因果前缀检查，再对重复惩罚和固定采样作受控比较。降低表面重复没有得到连贯性通过结论。最后固定原数据/优化器/RNG、最小LR3e-5，只追加2000步，预先约定同时满足valid下降≥0.01与18对连贯性至少10改善/至多3变差。", "",
             "| 同一PC环境对照 | 原12208步final | 续训14208步final |", "| --- | ---: | ---: |",
             f"| 完整valid loss | {comparison['validations']['baseline']['loss']} | {comparison['validations']['candidate']['loss']} |",
             "| 平均完整输出重复token 4-gram | 7.780309% | 7.655319% |",
             "| 剔除EOS后共同长度重复均值 | 6.523893% | 7.704540% |", "",
             f"valid loss下降{comparison['same_environment_valid_loss_decrease']:.6f}，数值门槛通过；全部18对定性审阅为{semantic['counts']['improved']}改善、{semantic['counts']['same']}无明确整体变化、{semantic['counts']['worse']}变差，连贯性门槛未通过。不是盲评、独立人工或事实准确率，也不宣称未见数据泛化改善。", "",
             "因此保留原12208步模型为展示/复现基线，续训权重仅保留对照，不继续加步数或搜索解码参数。详细方法与证据见[续训对照](bounded-continuation-execution.md#本地独立对照结果2026-10-15)。", "",
             "## 在本地体验", "",
             "已有准确版本权重与tokenizer的用户，按[本地复现](local-reproduction.md)运行CPU最小例子或可选CUDA采样。只有代码的新克隆不含权重、tokenizer或原始日志，不会自动下载/训练替代品。模型身份和适用边界见[模型卡](../MODEL_CARD.md)。", "",
             "页面与图片可在GitHub直接阅读，也可打开PNG/SVG查看；无需部署服务。代码MIT许可与上游语料/权重的适用条件分别记录，本轮不公开权重或数据。", "",
             "## 展示材料复建", "",
             "脚本[scripts/build_showcase.py](../scripts/build_showcase.py)只读取上述固定原始JSON/JSONL，先核对SHA-256，使用已有matplotlib生成两种格式曲线及本页；不导入模型或改写历史记录。默认拒绝已有输出，复建时指定全新输出目录。", "",
             "```powershell", "python scripts/build_showcase.py --output-root logs/m4_showcase_rebuild_v1", "```", "",
             "绘图库只用于离线图形，以上python须指向已有完整Matplotlib绘图环境；模型推理的项目.venv不需要它。本次使用已有本地Matplotlib3.10.8及依赖，没有安装或修改项目环境。复建需要原始日志；没有日志的代码新克隆仍可阅读已提交图片与本页。", ""]
    return "\n".join(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=PROJECT)
    args = parser.parse_args(argv)
    root = args.output_root.resolve()
    assets, page = root / "docs/assets", root / "docs/showcase.md"
    targets = [assets / "training-loss.svg", assets / "training-loss.png", page]
    if any(path.exists() for path in targets):
        raise FileExistsError("展示产物已存在；先检查，复建用新的--output-root，不覆盖")
    actual = {name: sha(PROJECT / name) for name in SOURCES}
    if actual != SOURCES:
        raise ValueError("来源摘要不符，不绘制或生成成功页面")
    metrics = read_rows(PROJECT / "logs/m3_training_report/m3_pretrain_100m_return_v1/metrics.jsonl")
    continued = read_rows(PROJECT / "logs/m3_contiune_2000_report/metrics.jsonl")
    samples = read_rows(PROJECT / "logs/m3_final_local_evaluation_v1/generations.jsonl")
    evaluation = read_json(PROJECT / "logs/m3_final_local_evaluation_v1/evaluation_report.json")
    comparison = read_json(PROJECT / "logs/m3_continuation_comparison_v1/comparison.json")
    semantic = read_json(PROJECT / "logs/m3_continuation_comparison_v1/semantic_review.json")
    assets.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT / "logs/m4_showcase_v1/matplotlib_cache"))
    chart = build_chart(assets, metrics, continued)
    with page.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(build_page(evaluation, comparison, semantic, samples))
    if {name: sha(PROJECT / name) for name in SOURCES} != SOURCES:
        raise ValueError("原始输入发生变化，停止")
    print(json.dumps({"status": "PASS", "output_root": str(root), "chart": chart, "fixed_samples": 12,
                      "sources_unchanged": True, "model_run": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
