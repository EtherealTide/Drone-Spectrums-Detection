# home界面的整体布局
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from qfluentwidgets import CardWidget, BodyLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from .config_interface import ConfigInterface
from .visualization import HomeVisualizationCard
from ..utils.component import Component


class HomeInterface(QWidget):
    def __init__(self, parent=None, data_processor=None, state=None, detector=None):
        super().__init__(parent)
        self.data_processor = data_processor
        self.state = state
        self.detector = detector
        self.setObjectName("HomeInterface")
        self.setup_ui()

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：可视化卡片（传入state）
        self.visualization_card = HomeVisualizationCard(
            parent=self,
            data_processor=self.data_processor,
            detector=self.detector,
            state=self.state,  # 传入state
        )
        splitter.addWidget(self.visualization_card)

        # 右侧：配置界面（传入state）
        self.config_interface = ConfigInterface(
            parent=self,
            state=self.state,  # 传入state
        )
        splitter.addWidget(self.config_interface)

        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

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
