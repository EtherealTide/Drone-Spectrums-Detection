from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QColor, QFont, QPainter, QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..settings.theme_manager import get_theme_manager


class CardWidget(QFrame):
    """Card styled container that mimics the Fluent look."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme_manager = get_theme_manager()
        self.setObjectName("CardWidget")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(0, 0, 0, 90))
        self.setGraphicsEffect(shadow)
        self._shadow = shadow
        self._theme_manager.paletteChanged.connect(self._apply_palette)
        self._apply_palette(self._theme_manager.palette)

    def _apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            QFrame#CardWidget {{
                background-color: {palette['card_bg']};
                border-radius: 14px;
                border: 1px solid {palette['card_border']};
            }}
            """
        )
        if self._shadow:
            color = QColor(palette.get("shadow", "#000000"))
            self._shadow.setColor(color)


class BodyLabel(QLabel):
    """Typography helper to replace qfluentwidgets.BodyLabel."""

    def __init__(self, text: str | QWidget = "", parent: QWidget | None = None):
        actual_parent = parent
        actual_text = text
        if isinstance(text, QWidget) and parent is None:
            actual_parent = text
            actual_text = ""
        super().__init__(actual_text, actual_parent)
        self.setObjectName("BodyLabel")
        self._custom_color = False
        self._theme_manager = get_theme_manager()
        font = QFont("Segoe UI", 11)
        font.setWeight(QFont.Weight.Medium)
        self.setFont(font)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._theme_manager.paletteChanged.connect(self._apply_palette)
        self._apply_palette(self._theme_manager.palette)

    def setTextColor(self, light: QColor | str, dark: QColor | str | None = None):
        self._custom_color = True
        light_color = QColor(light) if not isinstance(light, QColor) else light
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.WindowText, light_color)
        self.setPalette(palette)
        if dark:
            self._dark_color = QColor(dark)

    def _apply_palette(self, palette: dict):
        if self._custom_color:
            return
        color = QColor(palette["text_primary"])
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, color)
        self.setPalette(pal)


class SwitchButton(QWidget):
    """Simple Fluent-like toggle switch."""

    checkedChanged = pyqtSignal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._theme_manager = get_theme_manager()
        self._checked = False
        self._text_on = "On"
        self._text_off = "Off"
        self._offset = 0.0
        self._track_on = QColor("#4F8BFF")
        self._track_off = QColor("#C6CAD1")
        self._text_color = QColor("#1f1f1f")
        self._knob_color = QColor("#ffffff")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(160)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

        self.setMinimumWidth(100)
        self.setMinimumHeight(32)
        self._theme_manager.paletteChanged.connect(self._update_palette)
        self._update_palette(self._theme_manager.palette)

    def sizeHint(self):
        return QSize(110, 32)

    def setOnText(self, text: str):
        self._text_on = text
        self.update()

    def setOffText(self, text: str):
        self._text_off = text
        self.update()

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked == checked:
            return
        self._checked = checked

        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

        self.checkedChanged.emit(checked)
        self.update()

    def toggle(self):
        self.setChecked(not self._checked)

    def mouseReleaseEvent(self, event):
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self.rect().contains(event.position().toPoint())
            and self.isEnabled()
        ):
            self.toggle()
        super().mouseReleaseEvent(event)

    def offset(self):
        return self._offset

    def setOffset(self, value: float):
        self._offset = value
        self.update()

    offset = pyqtProperty(float, fget=offset, fset=setOffset)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        track_height = 22
        track_width = 46
        track_rect = QRectF(
            0, (rect.height() - track_height) / 2, track_width, track_height
        )
        radius = track_height / 2

        background = self._track_on if self._checked else self._track_off
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(background)
        painter.drawRoundedRect(track_rect, radius, radius)

        knob_diameter = track_height - 4
        knob_x = 2 + (track_width - knob_diameter - 2) * self._offset
        knob_rect = QRectF(knob_x, track_rect.top() + 2, knob_diameter, knob_diameter)
        painter.setBrush(self._knob_color)
        painter.drawEllipse(knob_rect)

        text_rect = QRectF(
            track_rect.right() + 10,
            0,
            rect.width() - track_rect.width() - 10,
            rect.height(),
        )
        painter.setPen(self._text_color)
        painter.setFont(QFont("Segoe UI", 10))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._text_on if self._checked else self._text_off,
        )

    def _update_palette(self, palette: dict):
        self._track_on = QColor(palette["switch_track_on"])
        self._track_off = QColor(palette["switch_track_off"])
        self._text_color = QColor(palette["switch_text"])
        self._knob_color = QColor(palette["switch_knob"])
        self.update()


class Component:
    """Factory helpers that create styled widgets."""

    def __init__(self):
        pass

    def create_combobox(self, parent: QWidget, items: list[str]):
        combobox = QComboBox(parent)
        combobox.addItems(items)
        manager = get_theme_manager()

        def apply_palette(palette: dict):
            combobox.setStyleSheet(
                f"""
                QComboBox {{
                    padding: 4px 8px;
                    border: 1px solid {palette['input_border']};
                    border-radius: 8px;
                    font: 11pt 'Segoe UI';
                    background-color: {palette['input_bg']};
                    color: {palette['text_primary']};
                }}
                QComboBox::drop-down {{
                    width: 24px;
                    border: none;
                }}
                QListView {{
                    background-color: {palette['stack_bg']};
                    color: {palette['text_primary']};
                }}
                """
            )

        manager.paletteChanged.connect(apply_palette)
        apply_palette(manager.palette)
        return combobox

    def create_card(
        self, parent: QWidget, height=100, width=None, layout_type="QHBoxLayout"
    ):
        card = CardWidget(parent)
        if height is not None:
            card.setFixedHeight(height)
        else:  # 设置为自适应高度
            card.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
            )
        if width is not None:
            card.setFixedWidth(width)

        if layout_type == "QHBoxLayout":
            layout = QHBoxLayout(card)
        elif layout_type == "QVBoxLayout":
            layout = QVBoxLayout(card)
        elif layout_type == "QGridLayout":
            layout = QGridLayout(card)
        else:
            raise ValueError(f"Unknown layout type: {layout_type}")

        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        return layout, card

    def create_switch_button(self, parent: QWidget, text_on="On", text_off="Off"):
        switch_button = SwitchButton(parent)
        switch_button.setOnText(text_on)
        switch_button.setOffText(text_off)
        switch_button.setChecked(False)
        return switch_button

    def create_label(
        self,
        parent: QWidget,
        text,
        color_light=None,
        color_dark=None,
        alignment=None,
    ):
        label = BodyLabel(text, parent)
        if color_light is not None:
            label.setTextColor(color_light, color_dark)
        if alignment:
            label.setAlignment(alignment)
        return label

    def create_line_edit(self, parent: QWidget, placeholder=None, width=400):
        line_edit = QLineEdit(parent)
        if placeholder:
            line_edit.setPlaceholderText(placeholder)
        line_edit.setFixedWidth(width)
        line_edit.setClearButtonEnabled(True)
        manager = get_theme_manager()

        def apply_palette(palette: dict):
            line_edit.setStyleSheet(
                f"""
                QLineEdit {{
                    border: 1px solid {palette['input_border']};
                    border-radius: 8px;
                    padding: 6px 8px;
                    font: 11pt 'Segoe UI';
                    background-color: {palette['input_bg']};
                    color: {palette['text_primary']};
                }}
                QLineEdit:focus {{
                    border-color: {palette['input_focus']};
                }}
                """
            )

        manager.paletteChanged.connect(apply_palette)
        apply_palette(manager.palette)
        return line_edit
