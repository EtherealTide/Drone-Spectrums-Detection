# home界面的整体布局
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from .config_interface import ConfigInterface
from .visualization import HomeVisualizationCard
from ..settings.theme_manager import get_theme_manager


class HomeInterface(QWidget):
    def __init__(self, parent=None, data_processor=None, state=None, detector=None):
        super().__init__(parent)
        self.data_processor = data_processor
        self.state = state
        self.detector = detector
        self.setObjectName("HomeInterface")
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：可视化卡片
        self.visualization_card = HomeVisualizationCard(
            parent=self,
            data_processor=self.data_processor,
            detector=self.detector,
            state=self.state,
        )
        self.splitter.addWidget(self.visualization_card)

        # 右侧：配置界面
        self.config_interface = ConfigInterface(
            parent=self,
            state=self.state,
        )
        self.splitter.addWidget(self.config_interface)
        # 设置左右面板的拉伸比例为5:3
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.splitter)

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #HomeInterface {{
                background-color: {palette['window_bg']};
            }}
            QSplitter::handle {{
                background-color: {palette['card_border']};
                width: 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {palette['nav_hover']};
            }}
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
