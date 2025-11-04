# coding:utf-8
from qfluentwidgets import SplashScreen
from qframelesswindow import FramelessWindow, StandardTitleBar
from PyQt6.QtCore import QSize, QEventLoop, QTimer
from PyQt6.QtGui import QIcon


class Demo(FramelessWindow):

    def __init__(self):
        super().__init__()
        self.resize(700, 600)
        self.setWindowTitle("PyQt-Fluent-Widgets")
        self.setWindowIcon(QIcon(":/qfluentwidgets/images/logo.png"))

        # 1. 创建启动页面
        self.splashScreen = SplashScreen(self.windowIcon(), self)
        self.splashScreen.setIconSize(QSize(102, 102))

        # 2. 在创建其他子页面前先显示主界面
        self.show()

        # 3. 创建子界面
        self.createSubInterface()
        import time

        # time.sleep(3)  # 模拟耗时操作
        # 4. 隐藏启动页面
        self.splashScreen.finish()

    def createSubInterface(self):  #
        loop = QEventLoop(self)
        QTimer.singleShot(200, loop.quit)  # 模拟耗时操作
        loop.exec()


if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = Demo()
    w.show()
    app.exec()
