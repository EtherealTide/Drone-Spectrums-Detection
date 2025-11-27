from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter

from .visualization import WaterfallVisualizationCard
from .config_interface import WaterfallConfigInterface
from ..settings.theme_manager import get_theme_manager


class WaterfallInterface(QWidget):
    def __init__(self, parent=None, data_processor=None, detector=None, state=None):
        super().__init__(parent)
        self.data_processor = data_processor
        self.detector = detector
        self.state = state
        self.setObjectName("WaterfallInterface")
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        self.visualization_card = WaterfallVisualizationCard(
            parent=self,
            data_processor=self.data_processor,
            detector=self.detector,
            state=self.state,
        )
        self.splitter.addWidget(self.visualization_card)

        self.config_interface = WaterfallConfigInterface(
            parent=self,
            state=self.state,
        )
        self.splitter.addWidget(self.config_interface)

        # Favor a wider left pane and set initial sizes
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([1000, 500])

        main_layout.addWidget(self.splitter)

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #WaterfallInterface {{
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
    window = WaterfallInterface()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
