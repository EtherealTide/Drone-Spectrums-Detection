# 各种组件的添加
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout
from qfluentwidgets import (
    CardWidget,
    PushButton,
    LineEdit,
    setCustomStyleSheet,
    SwitchButton,
    BodyLabel,
    QColor,
)


class Component:
    def __init__(self):
        pass

    # 创建卡片并设置高度和水平
    def create_card(self, parent: QWidget, height=100, layout_type="QHBoxLayout"):
        card = CardWidget(parent)
        card.setFixedHeight(height)
        if layout_type == "QHBoxLayout":
            layout = QHBoxLayout(card)
        elif layout_type == "QVBoxLayout":
            layout = QVBoxLayout(card)
        elif layout_type == "QGridLayout":
            layout = QGridLayout(card)
        return layout, card

    # 创建开关按钮，并设置开关对应显示的文本
    def create_switch_button(self, parent: QWidget, text_on="On", text_off="Off"):
        switch_button = SwitchButton(parent)
        switch_button.setOnText(text_on)
        switch_button.setOffText(text_off)
        switch_button.setChecked(False)
        return switch_button

    # 设置标签，包括文本内容和颜色，对齐方式，以及宽度
    def create_label(
        self,
        parent: QWidget,
        text,
        color_light,
        color_dark,
        alignment=None,
        width=None,
    ):
        label = BodyLabel(text, parent)
        label.setTextColor(QColor(color_light), QColor(color_dark))
        if alignment:
            label.setAlignment(alignment)
        if width:
            label.setFixedWidth(width)

        return label

    # 设置输入框
    def create_line_edit(self, parent: QWidget, placeholder=None, width=400):
        line_edit = LineEdit(parent)
        if placeholder:
            line_edit.setPlaceholderText(placeholder)
        line_edit.setFixedWidth(width)
        line_edit.setClearButtonEnabled(True)
        return line_edit
