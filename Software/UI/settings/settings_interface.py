from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..utils.component import BodyLabel, Component
from .theme_manager import Theme, get_theme_manager


class SettingsInterface(QWidget):
    """Settings page providing theme toggles."""

    def __init__(self, parent=None, theme_manager=None):
        super().__init__(parent)
        self.setObjectName("SettingsInterface")
        self.component = Component()
        self.theme_manager = theme_manager or get_theme_manager()

        self.light_button = None
        self.dark_button = None
        self._build_ui()

        self.theme_manager.themeChanged.connect(self._on_theme_changed)
        self.theme_manager.paletteChanged.connect(self._apply_palette)
        # Initialize button state & palette
        self._on_theme_changed(self.theme_manager.theme.value)
        self._apply_palette(self.theme_manager.palette)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        theme_layout, theme_card = self.component.create_card(
            self, height=200, layout_type="QVBoxLayout"
        )
        title = BodyLabel("Theme", theme_card)
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        theme_layout.addWidget(title)

        subtitle = BodyLabel(
            "Choose between light and dark appearance to match your workspace.",
            theme_card,
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        theme_layout.addWidget(subtitle)

        button_row = QHBoxLayout()
        button_row.setSpacing(12)
        self.light_button = QPushButton("Light Theme", theme_card)
        self.dark_button = QPushButton("Dark Theme", theme_card)
        for btn in (self.light_button, self.dark_button):
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setMinimumHeight(36)
            button_row.addWidget(btn)

        self.light_button.clicked.connect(
            lambda: self.theme_manager.set_theme(Theme.LIGHT)
        )
        self.dark_button.clicked.connect(lambda: self.theme_manager.set_theme(Theme.DARK))

        button_row.addStretch()
        theme_layout.addLayout(button_row)
        theme_layout.addStretch()
        layout.addWidget(theme_card)

    def _on_theme_changed(self, theme_value: str):
        is_light = theme_value == Theme.LIGHT.value
        if self.light_button:
            self.light_button.blockSignals(True)
            self.light_button.setChecked(is_light)
            self.light_button.blockSignals(False)
        if self.dark_button:
            self.dark_button.blockSignals(True)
            self.dark_button.setChecked(not is_light)
            self.dark_button.blockSignals(False)

    def _apply_palette(self, palette: dict):
        button_style = """
            QPushButton {{
                border: 1px solid {border};
                border-radius: 10px;
                padding: 6px 12px;
                background-color: {bg};
                color: {text};
                font: 11pt 'Segoe UI';
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:checked {{
                background-color: {checked_bg};
                color: {checked_text};
                border-color: transparent;
            }}
        """.format(
            border=palette["button_border"],
            bg=palette["button_bg"],
            text=palette["button_text"],
            hover=palette["button_hover"],
            checked_bg=palette["button_checked_bg"],
            checked_text=palette["button_checked_text"],
        )
        for btn in (self.light_button, self.dark_button):
            if btn:
                btn.setStyleSheet(button_style)

