from qfluentwidgets import NavigationItemPosition, FluentWindow, SubtitleLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from PyQt6.QtWidgets import QFrame, QHBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

# 添加相对导入路径
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from UI.icons.MyFluentIcon import MyFluentIcon
from UI.config.config_interface import ConfigInterface
from UI.visualization.visualization_interface import VisualizationInterface


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        self.label = SubtitleLabel(text, self)
        self.hBoxLayout = QHBoxLayout(self)

        setFont(self.label, 24)
        self.label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )  # 注意对齐常量从以前的 Qt.AlignCenter 变为 Qt.AlignmentFlag.AlignCenter
        self.hBoxLayout.addWidget(self.label, 1, Qt.AlignmentFlag.AlignCenter)

        # 必须给子界面设置全局唯一的对象名
        self.setObjectName(text.replace(" ", "-"))


class Window(FluentWindow):
    """主界面"""

    def __init__(self):
        super().__init__()
        # 创建子界面,实际使用时将 Widget 换成自己的子界面
        self.homeInterface = Widget("Home Interface", self)
        self.settingInterface = Widget("Setting Interface", self)
        self.albumInterface = Widget("Album Interface", self)
        self.albumInterface1 = Widget("Album Interface 1", self)
        self.configInterface = ConfigInterface(self)
        self.visualizationInterface = VisualizationInterface(self)

        self.initNavigation()
        self.initWindow()

    def initNavigation(self):
        self.addSubInterface(self.homeInterface, FIF.HOME, "Home")

        self.addSubInterface(
            self.configInterface, MyFluentIcon.CONFIG, "Config Interface"
        )
        self.addSubInterface(
            self.visualizationInterface,
            MyFluentIcon.VISUALIZATION,
            "Visualization Interface",
        )
        self.navigationInterface.addSeparator()  # 添加导航栏分割线

        self.addSubInterface(
            self.albumInterface, FIF.ALBUM, "Albums", NavigationItemPosition.SCROLL
        )
        self.addSubInterface(
            self.albumInterface1, FIF.ALBUM, "Album 1", parent=self.albumInterface
        )

        self.addSubInterface(
            self.settingInterface,
            FIF.SETTING,
            "Settings",
            NavigationItemPosition.BOTTOM,
        )

    def initWindow(self):
        self.resize(900, 700)
        self.setWindowIcon(QIcon(":/qfluentwidgets/images/logo.png"))
        self.setWindowTitle("Drone Detection System Dashboard")


from PyQt6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    app.exec()
