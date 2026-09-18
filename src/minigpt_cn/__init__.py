"""MiniGPT-CN 的顶层包。

这里只暴露稳定且常用的公共接口，具体实现分别放在对应子包中。
"""

from minigpt_cn.config import ModelConfig, load_model_config

__all__ = ["ModelConfig", "load_model_config", "__version__"]

__version__ = "0.1.0"
