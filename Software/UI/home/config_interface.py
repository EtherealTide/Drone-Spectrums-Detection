# home界面的右侧卡片内容，主要是各个模块的参数调整，使用树状控件
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTreeWidgetItem,
    QLabel,
)
from qfluentwidgets import (
    CardWidget,
    BodyLabel,
    TreeWidget,
    ComboBox,
    SpinBox,
    LineEdit,
    PushButton,
    SwitchButton,
    setCustomStyleSheet,
)
from qfluentwidgets import FluentIcon as FIF
from ..utils.component import Component
from ..utils.custom_style import CONFIRM_BUTTON_STYLE
import json
import os
from PyQt6.QtCore import QSize
import logging

logger = logging.getLogger(__name__)


class ConfigInterface(QWidget):
    # 定义信号
    connection_request = pyqtSignal(bool)  # True=连接, False=断开
    parameter_change_request = pyqtSignal(str, str, object)  # (group, name, value)

    def __init__(self, parent=None, state=None):
        super().__init__(parent)
        self.setObjectName("ConfigInterface")
        self.component = Component()
        self.state = state
        self.connection_switch = None
        self.setup_ui()

        # 监听状态变化
        if self.state:
            self.state.connection_changed.connect(self.on_connection_state_changed)
            # UI层监听参数变化只是为了更新显示值
            self.state.parameters_changed.connect(self.on_parameters_updated)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 创建参数配置卡片
        config_layout, config_card = self.component.create_card(self, height=600)

        # 创建树状控件
        self.config_tree = TreeWidget(config_card)
        self.config_tree.setHeaderHidden(True)
        self.config_tree.setColumnCount(2)
        self.config_tree.setColumnWidth(0, 250)
        self.config_tree.setColumnWidth(1, 250)

        # 构建树状结构（从state读取）
        self.build_tree()
        self.config_tree.expandAll()

        config_layout.addWidget(self.config_tree)
        layout.addWidget(config_card)

    def build_tree(self):
        """构建参数树状结构（从state读取初始值）"""
        # 1. System Status 节点
        system_item = QTreeWidgetItem(["System Status"])
        self.config_tree.addTopLevelItem(system_item)

        # 连接状态子项
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

        # 2. Receiver 节点
        receiver_item = QTreeWidgetItem(["Receiver"])
        self.config_tree.addTopLevelItem(receiver_item)

        # 从state读取初始值
        self.add_parameter(
            receiver_item,
            "Receiver",
            "Length",
            self.state.fft_length,
            ["128", "256", "512", "1024", "2048", "4096", "8192"],
        )

        self.add_parameter(
            receiver_item,
            "Receiver",
            "Decimation_factor",
            self.state.decimation_factor,
            ["4", "8", "16", "32", "64", "128", "256", "512", "1024"],
        )

        self.add_parameter(
            receiver_item,
            "Receiver",
            "Centre_frequency(MHz)",
            self.state.center_frequency,
            None,
        )

        self.add_parameter(
            receiver_item, "Receiver", "bandwidth(MHz)", self.state.bandwidth, None
        )

        # 3. UI 节点
        ui_item = QTreeWidgetItem(["UI"])
        self.config_tree.addTopLevelItem(ui_item)

        self.add_parameter(
            ui_item, "UI", "spectum_left_freq(MHz)", self.state.spectrum_left_freq, None
        )

        self.add_parameter(
            ui_item,
            "UI",
            "spectum_right_freq(MHz)",
            self.state.spectrum_right_freq,
            None,
        )
        # yolo检测参数节点
        detection_item = QTreeWidgetItem(["Detection"])
        self.config_tree.addTopLevelItem(detection_item)
        self.add_parameter(
            detection_item,
            "Detection",
            "conf_threshold",
            self.state.conf_threshold,
            None,
        )
        self.add_parameter(
            detection_item, "Detection", "iou_threshold", self.state.iou_threshold, None
        )

    def add_parameter(
        self, parent_item, param_group, param_name, current_value, options
    ):
        """添加参数行"""
        param_item = QTreeWidgetItem([param_name])
        parent_item.addChild(param_item)

        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(5, 5, 5, 5)
        param_layout.setSpacing(5)

        # 当前值标签
        value_label = self.component.create_label(
            param_widget,
            str(current_value),
            "#000000",
            "#E8F4F8",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        value_label.setFixedWidth(60)
        value_label.setStyleSheet(
            "background-color: #E8F4F8; border-radius: 3px; padding: 3px;"
        )
        param_layout.addWidget(value_label)

        # 输入控件
        if options:
            input_widget = ComboBox(param_widget)
            input_widget.addItems(options)
            input_widget.setCurrentText(str(current_value))
            input_widget.setFixedWidth(100)
        else:
            input_widget = LineEdit(param_widget)
            input_widget.setText(str(current_value))
            input_widget.setFixedWidth(100)

        param_layout.addWidget(input_widget)

        # Set按钮
        set_button = PushButton("Set", param_widget)
        set_button.setFixedWidth(50)
        setCustomStyleSheet(set_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE)

        def update_value():
            """发送参数更新请求"""
            new_value = input_widget.currentText() if options else input_widget.text()
            try:
                # 转换为数值
                numeric_value = float(new_value)
                if numeric_value.is_integer():
                    numeric_value = int(numeric_value)

                # 发射信号给main处理
                self.parameter_change_request.emit(
                    param_group, param_name, numeric_value
                )

                logger.info(
                    f"请求更新参数: {param_group}.{param_name} = {numeric_value}"
                )

            except ValueError:
                logger.error(f"无效的参数值: {param_name} = {new_value}")

        set_button.clicked.connect(update_value)
        param_layout.addWidget(set_button)
        param_layout.addStretch()

        param_item.setSizeHint(1, QSize(0, param_widget.sizeHint().height() + 10))
        self.config_tree.setItemWidget(param_item, 1, param_widget)

        # 保存引用（用于后续更新显示）
        if not hasattr(self, "_value_labels"):
            self._value_labels = {}
        self._value_labels[f"{param_group}.{param_name}"] = value_label

    def on_parameters_updated(self, change_info):
        """参数更新后，刷新UI显示值"""
        group = change_info["group"]
        name = change_info["name"]
        value = change_info["value"]

        key = f"{group}.{name}"
        if key in self._value_labels:
            self._value_labels[key].setText(str(value))
            logger.info(f"UI显示已更新: {key} = {value}")

    def on_connection_state_changed(self, is_connected):
        """连接状态变化时更新开关"""
        if self.connection_switch:
            self.connection_switch.blockSignals(True)
            self.connection_switch.setChecked(is_connected)
            self.connection_switch.blockSignals(False)
            self.connection_switch.setEnabled(True)

    def on_switch_toggled(self, checked):
        """开关被切换时触发"""
        logger.info(f"连接开关被切换: {checked}")
        self.connection_switch.setEnabled(False)
        self.connection_request.emit(checked)

        from PyQt6.QtCore import QTimer

        QTimer.singleShot(1000, lambda: self.connection_switch.setEnabled(True))


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = ConfigInterface()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())
