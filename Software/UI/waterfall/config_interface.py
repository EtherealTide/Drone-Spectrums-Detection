from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
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

        receiver_item = QTreeWidgetItem(["Slave Computer"])
        self.config_tree.addTopLevelItem(receiver_item)
        receiver_params = [
            ("Centre_frequency(MHz)", self.state.center_frequency, None),
            ("SPAN(MHz)", self.state.span, ["1000", "200", "100", "50", "25", "12.5"]),
        ]
        for name, value, options in receiver_params:
            self.add_parameter(receiver_item, "Slave Computer", name, value, options)

        dataprocess_item = QTreeWidgetItem(["Data Process"])
        self.config_tree.addTopLevelItem(dataprocess_item)
        dataprocess_params = [
            ("waterfall_height", self.state.waterfall_height, None),
            ("enable_noise_filter", self.state.enable_noise_filter, ["Enabled", "Disabled"]),
            ("noise_filter_mode", self.state.noise_filter_mode, ["subtraction", "threshold"]),
            ("noise_alpha", self.state.noise_alpha, None),
        ]   
        for name, value, options in dataprocess_params:
            self.add_parameter(dataprocess_item, "Data_Process", name, value, options)
        detection_item = QTreeWidgetItem(["Detection"])
        self.config_tree.addTopLevelItem(detection_item)

        detection_params = [
            ("conf_threshold", self.state.conf_threshold, None),
            ("iou_threshold", self.state.iou_threshold, None),
        ]
        for name, value, options in detection_params:
            self.add_parameter(detection_item, "Detection", name, value, options)
        
        ui_waterfall_item = QTreeWidgetItem(["UI Waterfall"])
        self.config_tree.addTopLevelItem(ui_waterfall_item)
        ui_waterfall_params = [
            ("spectrum_left_freq(MHz)", self.state.waterfall_left_freq, None),
            ("spectrum_right_freq(MHz)", self.state.waterfall_right_freq, None),
        ]
        for name, value, options in ui_waterfall_params:
            self.add_parameter(ui_waterfall_item, "UI_Waterfall", name, value, options)

    def add_parameter(
        self, parent_item, param_group, param_name, current_value, options
    ):
        '''
        Args:
            parent_item: 父级树节点(QTreeWidgetItem)
            param_group: 参数组名称
            param_name: 参数名称
            current_value: 当前参数值
            options: 选项列表，如果提供则使用下拉框，否则使用文本框
        '''
        # 核心思路：每一行是一个子树节点和一个包含标签、输入控件和按钮的Widget。
        # 子树节点显示参数名称，Widget显示当前值和输入框。输入框的类型根据是否提供选项决定。
        # 点击按钮时读取输入值并发出参数更新信号。
        param_item = QTreeWidgetItem([param_name]) # 创建显示参数名称的树节点，放在第一列
        parent_item.addChild(param_item)

        param_widget = QWidget() 
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(5)

        value_label = self.component.create_label( # 创建显示当前参数值的标签
            param_widget,
            str(current_value),
            None,
            None,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        value_label.setFixedWidth(80)
        param_layout.addWidget(value_label)
        self._value_label_widgets.append(value_label)
        # 创建输入控件，根据是否提供选项决定使用下拉框还是文本框，并设置初始值和样式
        if options:
            input_widget = self.component.create_combobox(param_widget, options)
            input_widget.setCurrentText(str(current_value))
            input_widget.setFixedWidth(120)
        else:
            input_widget = self.component.create_line_edit(param_widget, width=120)
            input_widget.setText(str(current_value))

        param_layout.addWidget(input_widget)

        set_button = QPushButton("Set", param_widget)
        set_button.setFixedWidth(50)
        set_button.setStyleSheet(CONFIRM_BUTTON_STYLE)

        def update_value():
            new_value = input_widget.currentText() if options else input_widget.text()
            try:
                numeric_value = new_value 
                
                # 1. Handle Boolean (Enabled/Disabled)
                if new_value == "Enabled":
                    numeric_value = True
                elif new_value == "Disabled":
                    numeric_value = False
                
                # 2. Handle Numbers (if original value was number)
                # We interpret as number if it looks like one AND the original wasn't a string (unless it was a string that looked like a number, edge case)
                # But safer logic: Try convert to float/int.
                # If the user passed a STRING option "subtraction", keep it as string.
                # If current_value is a number, we enforce number.
                elif isinstance(current_value, (int, float)) and not isinstance(current_value, bool):
                     numeric_value = float(new_value)
                     if numeric_value.is_integer():
                         numeric_value = int(numeric_value)
                
                # 3. Handle Strings (explicitly or fallback)
                # If current_value was string, we just pass the new string (already done by default)
                
                # Log and Emit
                self.parameter_change_request.emit(
                    param_group, param_name, numeric_value
                )
                logger.info(
                    f"Request parameter update: {param_group}.{param_name} = {numeric_value} (type: {type(numeric_value).__name__})"
                )
            except ValueError:
                logger.error(f"Invalid parameter value {param_name} = {new_value}")

        set_button.clicked.connect(update_value) # set按钮链接到更新函数，点击时读取输入值并发出参数更新信号
        param_layout.addWidget(set_button)
        param_layout.addStretch()

        param_item.setSizeHint(1, QSize(0, param_widget.sizeHint().height() + 10))
        self.config_tree.setItemWidget(param_item, 1, param_widget) # 将包含输入控件的Widget放到树节点的第二列
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
        # 无论连接成功与否，一秒后重新启用开关，一秒内用户无法再次切换，避免重复请求和状态混乱
        QTimer.singleShot(1000, lambda: self.connection_switch.setEnabled(True))


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = WaterfallConfigInterface()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())
