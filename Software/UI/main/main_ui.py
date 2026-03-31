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

from UI.spectrum.home import SpectrumInterface
from UI.settings.settings_interface import SettingsInterface
from UI.settings.theme_manager import get_theme_manager
from UI.waterfall.waterfall_ui import WaterfallInterface
from UI.performance.performance_ui import PerformanceUI


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

    def __init__(
        self,shm_spectrum=None, shm_detection=None, state=None, perf_manager=None
    ):
        super().__init__()

        # 设置窗口大小 - 修改为更合理的尺寸
        from PyQt6.QtWidgets import QApplication

        screen = QApplication.primaryScreen().availableGeometry()

        # 使用屏幕可用区域的 90%，避免超出屏幕
        width = int(screen.width() * 0.9)
        height = int(screen.height() * 0.9)

        self.resize(width, height)

        # 居中显示
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.move(x, y)

        self.shm_spectrum = shm_spectrum
        self.shm_detection = shm_detection
        self.state = state
        self.perf_manager = perf_manager
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
        self.spectrumInterface = SpectrumInterface(
            self,
            shm_spectrum=self.shm_spectrum,
            state=self.state,
        )
        self.waterfallInterface = WaterfallInterface(
            self,
            shm_detection=self.shm_detection,
            state=self.state,
        )
        self.settingInterface = SettingsInterface(self, self.theme_manager)
        self.performanceInterface = PerformanceUI(self.perf_manager, self)
        
        self.albumInterface = Widget("Album Interface", self)
        self.albumInterface1 = Widget("Album Interface 1", self)

        self._add_page("Spectrum", self.spectrumInterface, "📡")
        self._add_page(
            "Waterfall",
            self.waterfallInterface,
            "🌊",
        )
        self._add_page("Performance", self.performanceInterface, "📈")
        # self._add_page("Albums", self.albumInterface, "🖥️")
        # self._add_page("Album 1", self.albumInterface1, "📂")
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
