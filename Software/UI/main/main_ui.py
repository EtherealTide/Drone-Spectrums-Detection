from qfluentwidgets import NavigationItemPosition, FluentWindow, SubtitleLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from PyQt6.QtWidgets import QFrame, QHBoxLayout
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from qfluentwidgets import setTheme, Theme
from PyQt6.QtCore import QEventLoop, QTimer

# 添加相对导入路径
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
from UI.icons.MyFluentIcon import MyFluentIcon
from UI.config.config_interface import ConfigInterface
from UI.visualization.visualization_interface import VisualizationInterface
from UI.home.home import HomeInterface
from qfluentwidgets import SplashScreen
from qframelesswindow import FramelessWindow, StandardTitleBar
from PyQt6.QtCore import QSize


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

    def __init__(self, dataprocessor=None):
        super().__init__()
        self.data_processor = dataprocessor
        logo_path = Path(__file__).parent.parent / "main" / "logo.png"
        # 显示启动界面
        self.startInterface(logo_path)
        # 创建子页面
        self.homeInterface = HomeInterface(self, data_processor=self.data_processor)
        self.configInterface = ConfigInterface(self)
        self.visualizationInterface = VisualizationInterface(self)
        self.settingInterface = Widget("Setting Interface", self)
        self.albumInterface = Widget("Album Interface", self)
        self.albumInterface1 = Widget("Album Interface 1", self)

        self.initNavigation()
        self.initWindow(logo_path)

    def startInterface(self, logo_path):
        self.resize(700, 600)
        self.setWindowTitle("Drone Detection System Dashboard")
        self.setWindowIcon(QIcon(str(logo_path)))
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(200, 200))

        # 2. 在创建其他子页面前先显示主界面
        self.show()
        self.createSubInterface()
        # 4. 隐藏启动页面
        self.splashScreen.finish()

    def createSubInterface(self):
        """使用事件循环模拟耗时操作"""
        loop = QEventLoop(self)
        QTimer.singleShot(1000, loop.quit)  # 1秒后退出循环
        loop.exec()  # 在这期间，事件循环继续运行，UI 可以正常渲染

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

    def initWindow(self, my_logo_path):
        self.resize(1280, 720)
        # 将logo改为自定义图标

        self.setWindowIcon(QIcon(str(my_logo_path)))
        # self.setWindowIcon(QIcon(":/qfluentwidgets/images/logo.png"))
        self.setWindowTitle("Drone Detection System Dashboard")
        # 强制设置全局背景为白色
        self.setStyleSheet(
            """
            QWidget {
                background-color: white;
            }
            FluentWindow {
                background-color: white;
            }
        """
        )


from PyQt6.QtWidgets import QApplication
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)

    w = Window()
    w.show()
    app.exec()
