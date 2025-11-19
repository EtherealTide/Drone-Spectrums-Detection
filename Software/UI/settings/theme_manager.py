from __future__ import annotations

from enum import Enum
from typing import Dict

from PyQt6.QtCore import QObject, pyqtSignal


class Theme(str, Enum):
    LIGHT = "light"
    DARK = "dark"


THEME_PALETTES: Dict[Theme, dict] = {
    Theme.LIGHT: {
        "window_bg": "#f5f6fa",
        "panel_bg": "#f1f3f5",
        "stack_bg": "#ffffff",
        "text_primary": "#1f1f1f",
        "text_secondary": "#3f3f3f",
        "card_bg": "#ffffff",
        "card_border": "rgba(0, 0, 0, 0.18)",
        "shadow": "rgba(0, 0, 0, 0.28)",
        "nav_selected": "#dbe4ff",
        "nav_hover": "#edf2ff",
        "nav_text": "#1f1f1f",
        "accent": "#4F8BFF",
        "button_bg": "#ffffff",
        "button_text": "#1f1f1f",
        "button_border": "#ced4da",
        "button_hover": "#e9ecef",
        "button_checked_bg": "#4F8BFF",
        "button_checked_text": "#ffffff",
        "input_bg": "#ffffff",
        "input_border": "#ced4da",
        "input_focus": "#4F8BFF",
        "switch_track_on": "#4F8BFF",
        "switch_track_off": "#C6CAD1",
        "switch_text": "#1f1f1f",
        "switch_knob": "#ffffff",
    },
    Theme.DARK: {
        "window_bg": "#1F2024",
        "panel_bg": "#25262b",
        "stack_bg": "#2c2d32",
        "text_primary": "#f3f5f7",
        "text_secondary": "#cfd3da",
        "card_bg": "#2f3036",
        "card_border": "#3d3e45",
        "shadow": "rgba(0, 0, 0, 0.65)",
        "nav_selected": "#3a475e",
        "nav_hover": "#333b4d",
        "nav_text": "#f5f5f5",
        "accent": "#7AA2FF",
        "button_bg": "#2f3036",
        "button_text": "#f5f5f5",
        "button_border": "#4b4d55",
        "button_hover": "#3a3c45",
        "button_checked_bg": "#7AA2FF",
        "button_checked_text": "#0f1117",
        "input_bg": "#2f3036",
        "input_border": "#4b4d55",
        "input_focus": "#7AA2FF",
        "switch_track_on": "#7AA2FF",
        "switch_track_off": "#4b4d55",
        "switch_text": "#f5f5f5",
        "switch_knob": "#fdfdfd",
    },
}


class ThemeManager(QObject):
    """Central theme registry used across the UI."""

    paletteChanged = pyqtSignal(dict)
    themeChanged = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._theme = Theme.LIGHT

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def palette(self) -> dict:
        return dict(THEME_PALETTES[self._theme])

    def set_theme(self, theme: Theme | str):
        if isinstance(theme, str):
            theme = Theme(theme)

        if theme == self._theme:
            return

        self._theme = theme
        palette_snapshot = self.palette
        self.themeChanged.emit(self._theme.value)
        self.paletteChanged.emit(palette_snapshot)


_THEME_MANAGER: ThemeManager | None = None


def get_theme_manager() -> ThemeManager:
    global _THEME_MANAGER
    if _THEME_MANAGER is None:
        _THEME_MANAGER = ThemeManager()
    return _THEME_MANAGER

