from __future__ import annotations

from PyQt6.QtCore import QEventLoop, QSize, Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplashScreen,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from UI.home.home import HomeInterface
from UI.settings.settings_interface import SettingsInterface
from UI.settings.theme_manager import get_theme_manager
from UI.visualization.visualization_interface import VisualizationInterface


class Widget(QFrame):
    """Simple placeholder widget used for unfinished sections."""

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        label = QLabel(text, self)
        label.setFont(font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setObjectName(text.replace(" ", "-"))


class Window(QMainWindow):
    """Main application window fully based on PyQt6 widgets."""

    def __init__(self, dataprocessor=None, state=None, detector=None):
        super().__init__()
        self.data_processor = dataprocessor
        self.state = state
        self.detector = detector
        self.logo_path = Path(__file__).parent / "logo.png"
        self._splash = None
        self.theme_manager = get_theme_manager()

        self.startInterface()
        self._build_interfaces()
        self.initWindow()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def startInterface(self):
        """Display a lightweight splash screen while loading heavy resources."""
        self.resize(700, 560)
        self.setWindowTitle("Drone Detection System Dashboard")
        if self.logo_path.exists():
            logo_icon = QIcon(str(self.logo_path))
            self.setWindowIcon(logo_icon)
            pixmap = QPixmap(str(self.logo_path)).scaled(
                220,
                220,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._splash = QSplashScreen(pixmap)
            self._splash.show()
            QApplication.processEvents()
        self.show()
        self.createSubInterface()
        if self._splash:
            self._splash.close()

    def _build_interfaces(self):
        central = QWidget(self)
        central.setObjectName("MainWindowBackground")
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.navigation_list = QListWidget()
        self.navigation_list.setObjectName("NavigationList")
        self.navigation_list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.navigation_list.setIconSize(QSize(28, 28))
        self.navigation_list.setSpacing(4)
        self.navigation_list.setFixedWidth(220)

        self.stack = QStackedWidget()
        self.stack.setObjectName("CentralStack")

        layout.addWidget(self.navigation_list)
        layout.addWidget(self.stack, 1)

        # Build sub interfaces
        self.homeInterface = HomeInterface(
            self,
            data_processor=self.data_processor,
            state=self.state,
            detector=self.detector,
        )
        self.visualizationInterface = VisualizationInterface(self)
        self.settingInterface = SettingsInterface(self, self.theme_manager)
        self.albumInterface = Widget("Album Interface", self)
        self.albumInterface1 = Widget("Album Interface 1", self)

        self._add_page("Home", self.homeInterface, "🏠")
        self._add_page(
            "Visualization Interface",
            self.visualizationInterface,
            "📊",
        )
        self._add_page("Albums", self.albumInterface, "🖥️")
        self._add_page("Album 1", self.albumInterface1, "📂")
        self._add_page("Settings", self.settingInterface, "⚙️")

        self.navigation_list.currentRowChanged.connect(self.stack.setCurrentIndex)
        if self.navigation_list.count():
            self.navigation_list.setCurrentRow(0)

    def _add_page(self, title: str, widget: QWidget, icon_text: str = ""):
        """添加页面，使用文本图标"""
        self.stack.addWidget(widget)

        # 创建自定义item
        item = QListWidgetItem(f"{icon_text}  {title}")
        item.setSizeHint(QSize(200, 46))
        self.navigation_list.addItem(item)

    def createSubInterface(self):
        """Simulate loading delay to keep splash screen visible."""
        loop = QEventLoop(self)
        QTimer.singleShot(800, loop.quit)
        loop.exec()

    def initWindow(self):
        self.resize(1200, 900)
        if self.logo_path.exists():
            self.setWindowIcon(QIcon(str(self.logo_path)))
        self.setMinimumSize(1080, 600)

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {palette['window_bg']};
                color: {palette['text_primary']};
            }}
            QWidget#MainWindowBackground {{
                background-color: {palette['window_bg']};
            }}
            QListWidget#NavigationList {{
                background-color: {palette['panel_bg']};
                border: none;
                border-right: 1px solid {palette['card_border']};
                padding: 12px 0;
                color: {palette['nav_text']};
            }}
            QListWidget#NavigationList::item {{
                margin: 4px 12px;
                padding: 10px 12px;
                border-radius: 10px;
            }}
            QListWidget#NavigationList::item:selected {{
                background-color: {palette['nav_selected']};
                color: {palette['nav_text']};
            }}
            QListWidget#NavigationList::item:hover {{
                background-color: {palette['nav_hover']};
            }}
            QStackedWidget#CentralStack {{
                background-color: {palette['stack_bg']};
            }}
            QLabel#BodyLabel {{
                color: {palette['text_primary']};
            }}
            """
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
