from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QComboBox,
    QLineEdit,
    QPushButton,
)
import logging

from ..utils.component import Component
from ..utils.custom_style import CONFIRM_BUTTON_STYLE
from ..settings.theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class WaterfallConfigInterface(QWidget):
    """Config panel for the waterfall view."""

    connection_request = pyqtSignal(bool)
    parameter_change_request = pyqtSignal(str, str, object)

    def __init__(self, parent=None, state=None):
        super().__init__(parent)
        self.setObjectName("WaterfallConfigInterface")
        self.component = Component()
        self.state = state
        self.connection_switch = None
        self.theme_manager = get_theme_manager()
        self._value_label_widgets = []
        self._value_labels = {}
        self.setup_ui()

        if self.state:
            self.state.connection_changed.connect(self.on_connection_state_changed)
            self.state.parameters_changed.connect(self.on_parameters_updated)

        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        config_layout, config_card = self.component.create_card(self, height=600)

        self.config_tree = QTreeWidget(config_card)
        self.config_tree.setHeaderHidden(True)
        self.config_tree.setColumnCount(2)
        self.config_tree.setColumnWidth(0, 250)
        self.config_tree.setColumnWidth(1, 250)

        self.build_tree()
        self.config_tree.expandAll()

        config_layout.addWidget(self.config_tree)
        layout.addWidget(config_card)

    def build_tree(self):
        self.config_tree.clear()

        system_item = QTreeWidgetItem(["System Status"])
        self.config_tree.addTopLevelItem(system_item)

        connection_item = QTreeWidgetItem(["Connection"])
        system_item.addChild(connection_item)

        switch_widget = QWidget()
        switch_layout = QHBoxLayout(switch_widget)
        switch_layout.setContentsMargins(5, 5, 5, 5)
        self.connection_switch = self.component.create_switch_button(
            switch_widget, "Connected", "Disconnected"
        )
        self.connection_switch.checkedChanged.connect(self.on_switch_toggled)
        switch_layout.addWidget(self.connection_switch)
        switch_layout.addStretch()
        connection_item.setSizeHint(1, QSize(0, switch_widget.sizeHint().height() + 10))
        self.config_tree.setItemWidget(connection_item, 1, switch_widget)

        receiver_item = QTreeWidgetItem(["Receiver"])
        self.config_tree.addTopLevelItem(receiver_item)
        receiver_params = [
            (
                "FFT_Length",
                self.state.fft_length,
                ["128", "256", "512", "1024", "2048", "4096", "8192"],
            ),
            (
                "Decimation_factor",
                self.state.decimation_factor,
                ["4", "8", "16", "32", "64", "128", "256", "512", "1024"],
            ),
            ("Centre_frequency(MHz)", self.state.center_frequency, None),
            ("SPAN(MHz)", self.state.span, None),
        ]
        for name, value, options in receiver_params:
            self.add_parameter(receiver_item, "Receiver", name, value, options)

        waterfall_item = QTreeWidgetItem(["UI_Waterfall"])
        self.config_tree.addTopLevelItem(waterfall_item)
        waterfall_params = [
            ("waterfall_height", self.state.waterfall_height, None),
        ]
        for name, value, options in waterfall_params:
            self.add_parameter(waterfall_item, "UI_Waterfall", name, value, options)

        detection_item = QTreeWidgetItem(["Detection"])
        self.config_tree.addTopLevelItem(detection_item)
        detection_params = [
            ("conf_threshold", self.state.conf_threshold, None),
            ("iou_threshold", self.state.iou_threshold, None),
            (
                "image_size",
                self.state.image_size,
                ["default", "512", "1024", "1280", "1600", "1920", "2048", "2560"],
            ),
        ]
        for name, value, options in detection_params:
            self.add_parameter(detection_item, "Detection", name, value, options)
        scanner_params = [
            ("control_lost_threshold", self.state.control_lost_threshold, None),
            ("scan_bandwidth_mhz", self.state.scan_bandwidth_mhz, None),
        ]
        for name, value, options in scanner_params:
            self.add_parameter(detection_item, "Scanner", name, value, options)

    def add_parameter(
        self, parent_item, param_group, param_name, current_value, options
    ):
        param_item = QTreeWidgetItem([param_name])
        parent_item.addChild(param_item)

        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(5)

        value_label = self.component.create_label(
            param_widget,
            str(current_value),
            None,
            None,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        value_label.setFixedWidth(80)
        param_layout.addWidget(value_label)
        self._value_label_widgets.append(value_label)

        if options:
            input_widget = QComboBox(param_widget)
            input_widget.addItems(options)
            input_widget.setCurrentText(str(current_value))
            input_widget.setFixedWidth(120)
        else:
            input_widget = QLineEdit(param_widget)
            input_widget.setText(str(current_value))
            input_widget.setFixedWidth(120)

        param_layout.addWidget(input_widget)

        set_button = QPushButton("Set", param_widget)
        set_button.setFixedWidth(50)
        set_button.setStyleSheet(CONFIRM_BUTTON_STYLE)

        def update_value():
            new_value = input_widget.currentText() if options else input_widget.text()
            try:
                numeric_value = new_value
                if new_value != "default":
                    numeric_value = float(new_value)
                    if numeric_value.is_integer():
                        numeric_value = int(numeric_value)
                self.parameter_change_request.emit(
                    param_group, param_name, numeric_value
                )
                logger.info(
                    f"Request parameter update: {param_group}.{param_name} = {numeric_value}"
                )
            except ValueError:
                logger.error(f"Invalid parameter value {param_name} = {new_value}")

        set_button.clicked.connect(update_value)
        param_layout.addWidget(set_button)
        param_layout.addStretch()

        param_item.setSizeHint(1, QSize(0, param_widget.sizeHint().height() + 10))
        self.config_tree.setItemWidget(param_item, 1, param_widget)
        self._value_labels[f"{param_group}.{param_name}"] = value_label

    def apply_palette(self, palette: dict):  # 用于应用主题调色板
        self.setStyleSheet(
            f"""
            #WaterfallConfigInterface {{
                background-color: {palette['stack_bg']};
            }}
            """
        )

        tree_style = f"""
            QTreeWidget {{
                background-color: {palette['card_bg']};
                border: none;
                color: {palette['text_primary']};
                outline: 0;
            }}
            QTreeWidget::item {{
                margin: 2px 0;
                border-radius: 6px;
            }}
            QTreeWidget::item:selected {{
                background-color: {palette['nav_selected']};
                color: {palette['nav_text']};
            }}
            QTreeWidget::item:hover {{
                background-color: {palette['nav_hover']};
            }}
        """
        self.config_tree.setStyleSheet(tree_style)

        chip_style = f"""
            QLabel {{
                background-color: {palette['panel_bg']};
                border-radius: 6px;
                padding: 4px 6px;
                color: {palette['text_primary']};
                font-weight: 600;
            }}
        """
        for label in self._value_label_widgets:
            label.setStyleSheet(chip_style)

    def on_parameters_updated(self, change_info):
        key = f"{change_info['group']}.{change_info['name']}"
        if key in self._value_labels:
            self._value_labels[key].setText(str(change_info["value"]))
            logger.info(f"UI display refreshed: {key} = {change_info['value']}")

    def on_connection_state_changed(
        self, is_connected
    ):  # 更新连接状态，在state变化时调用
        if self.connection_switch:
            self.connection_switch.blockSignals(True)
            self.connection_switch.setChecked(is_connected)  # 更新开关状态
            self.connection_switch.blockSignals(False)
            self.connection_switch.setEnabled(True)

    def on_switch_toggled(self, checked):  # 切换连接状态，在用户操作时发出连接请求信号
        logger.info(f"Connection toggle changed: {checked}")
        self.connection_switch.setEnabled(False)
        self.connection_request.emit(checked)

        from PyQt6.QtCore import QTimer

        QTimer.singleShot(1000, lambda: self.connection_switch.setEnabled(True))


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = WaterfallConfigInterface()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())
