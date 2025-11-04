from enum import Enum
from pathlib import Path
from qfluentwidgets import getIconColor, Theme, FluentIconBase


class MyFluentIcon(FluentIconBase, Enum):
    """Custom icons"""

    CONFIG = "Config"  # 配置界面小图标
    VISUALIZATION = "Visualization"  # 可视化界面小图标

    def path(self, theme=Theme.AUTO):
        # 获取当前文件所在目录的父目录(UI目录),然后拼接icons路径
        ui_dir = Path(__file__).parent.parent  # 从 icons 目录回到 UI 目录
        icon_path = ui_dir / "icons" / f"{self.value}.svg"
        return str(icon_path)
