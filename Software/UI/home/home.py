# home界面的整体布局
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from qfluentwidgets import CardWidget, BodyLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from .config_interface import ConfigInterface
from .visualization import HomeVisualizationCard
from ..utils.component import Component


class HomeInterface(QWidget):
    def __init__(self, parent=None, data_processor=None, state=None):
        super().__init__(parent)
        self.data_processor = data_processor
        self.state = state
        self.setObjectName("HomeInterface")
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧可视化卡片
        self.visualization_card = HomeVisualizationCard(
            self, data_processor=self.data_processor
        )
        splitter.addWidget(self.visualization_card)

        # 右侧配置界面卡片
        self.config_interface = ConfigInterface(self, state=self.state)
        splitter.addWidget(self.config_interface)

        # 设置初始比例 (左:右 = 4:2)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self.setStyleSheet(
            """
            #HomeInterface { 
                background: white; 
            }
            QSplitter::handle {
                background-color: #E0E0E0;
                width: 2px;
            }
            """
        )


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = HomeInterface()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
