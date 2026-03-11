from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSplitter
from .config_interface import SpectrumConfigInterface
from .visualization import SpectrumVisualizationCard
from ..settings.theme_manager import get_theme_manager


class SpectrumInterface(QWidget):
    def __init__(self, parent=None, shm_spectrum=None, state=None, detector=None):
        super().__init__(parent)
        self.shm_spectrum = shm_spectrum
        self.state = state
        self.setObjectName("SpectrumInterface")
        self.theme_manager = get_theme_manager()
        self.setup_ui()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: spectrum visualization
        self.visualization_card = SpectrumVisualizationCard(
            parent=self,
            shm_spectrum=self.shm_spectrum,
            state=self.state,
        )
        self.splitter.addWidget(self.visualization_card)

        # Right: config panel
        self.config_interface = SpectrumConfigInterface(
            parent=self,
            state=self.state,
        )
        self.splitter.addWidget(self.config_interface)

        # Favor a wider left pane and set initial sizes
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setSizes([900, 500])

        main_layout.addWidget(self.splitter)

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #SpectrumInterface {{
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
    window = SpectrumInterface()
    window.resize(1280, 720)
    window.show()
    sys.exit(app.exec())
