"""验证安装后的顶层包和规划中的子包都能正常导入。"""

import minigpt_cn
import minigpt_cn.data
import minigpt_cn.evaluation
import minigpt_cn.inference
import minigpt_cn.model
import minigpt_cn.tokenizer
import minigpt_cn.training


def test_version() -> None:
    """包版本是发布和 checkpoint 追踪的基础标识。"""

    assert minigpt_cn.__version__ == "0.1.0"


def test_planned_subpackages_import() -> None:
    """提前固定源码布局，后续实现组件时无需移动公共包路径。"""

    packages = (
        minigpt_cn.data,
        minigpt_cn.evaluation,
        minigpt_cn.inference,
        minigpt_cn.model,
        minigpt_cn.tokenizer,
        minigpt_cn.training,
    )

    assert all(package.__name__.startswith("minigpt_cn.") for package in packages)
