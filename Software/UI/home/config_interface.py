# home界面的右侧卡片内容，主要是各个模块的参数调整，使用树状控件
from PyQt6.QtCore import Qt
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


class ConfigInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ConfigInterface")
        self.component = Component()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 读取参数配置文件
        config_path = os.path.join(
            os.path.dirname(__file__), "../config/parameters.json"
        )
        with open(config_path, "r", encoding="utf-8") as f:
            self.parameters = json.load(f)

        # 创建参数配置卡片
        config_layout, config_card = self.component.create_card(self, height=600)

        # 创建树状控件
        self.config_tree = TreeWidget(config_card)
        self.config_tree.setHeaderHidden(False)  # 显示表头
        self.config_tree.setColumnCount(2)  # 设置两列：参数名和值
        self.config_tree.setHeaderLabels(["Parameter", "Value/Control"])

        # 设置列宽
        self.config_tree.setColumnWidth(0, 200)  # 第一列宽度
        self.config_tree.setColumnWidth(1, 250)  # 第二列宽度

        # 构建树状结构
        self.build_tree()

        # 展开所有节点
        self.config_tree.expandAll()

        config_layout.addWidget(self.config_tree)
        layout.addWidget(config_card)

        self.setStyleSheet("#ConfigInterface { background: white; }")

    def build_tree(self):
        """构建参数树状结构"""

        # 1. System Status 节点
        system_item = QTreeWidgetItem(["System Status"])
        self.config_tree.addTopLevelItem(system_item)

        # 连接状态子项 - 使用开关
        connection_item = QTreeWidgetItem(["Connection"])
        system_item.addChild(connection_item)

        # 创建开关控件
        switch_widget = QWidget()
        switch_layout = QHBoxLayout(switch_widget)
        switch_layout.setContentsMargins(0, 0, 0, 0)

        switch = self.component.create_switch_button(
            switch_widget, "Connected", "Disconnected"
        )
        switch.checkedChanged.connect(
            lambda checked: print(
                f"System connection: {'Connected' if checked else 'Disconnected'}"
            )
        )
        switch_layout.addWidget(switch)
        switch_layout.addStretch()

        self.config_tree.setItemWidget(connection_item, 1, switch_widget)

        # 2. Receiver 节点
        receiver_item = QTreeWidgetItem(["Receiver"])
        self.config_tree.addTopLevelItem(receiver_item)

        # FFT 参数子节点
        fft_params = self.parameters.get("FFT", {})
        fft_item = QTreeWidgetItem(["FFT"])
        receiver_item.addChild(fft_item)

        # FFT Length - 使用卡片形式
        self.add_fft_parameter(
            fft_item,
            "Length",
            fft_params.get("Length", 256),
            ["128", "256", "512", "1024", "2048", "4096", "8192"],
        )

        # FFT Decimation Factor
        self.add_fft_parameter(
            fft_item,
            "Decimation Factor",
            fft_params.get("Decimation_factor", 256),
            ["64", "128", "256", "512", "1024"],
        )

        # FFT Centre Frequency
        self.add_fft_parameter(
            fft_item,
            "Centre Frequency (MHz)",
            fft_params.get("Centre_frequency(MHz)", 400),
            None,  # None 表示使用输入框
        )

    def add_fft_parameter(self, parent_item, param_name, current_value, options):
        """添加FFT参数行 - 包含当前值、选择框和Set按钮"""
        param_item = QTreeWidgetItem([param_name])
        parent_item.addChild(param_item)

        # 创建参数控制卡片
        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(2, 2, 2, 2)
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
            input_widget.setPlaceholderText(f"Enter {param_name}")
            input_widget.setFixedWidth(100)

        param_layout.addWidget(input_widget)

        # Set按钮
        set_button = PushButton("Set", param_widget)
        set_button.setFixedWidth(50)
        setCustomStyleSheet(set_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE)

        def update_value():
            new_value = input_widget.currentText() if options else input_widget.text()
            try:
                numeric_value = float(new_value)
                if numeric_value.is_integer():
                    numeric_value = int(numeric_value)

                # 更新显示
                value_label.setText(str(numeric_value))

                # 更新参数文件
                if param_name == "Length":
                    self.parameters["FFT"]["Length"] = numeric_value
                elif param_name == "Decimation Factor":
                    self.parameters["FFT"]["Decimation_factor"] = numeric_value
                elif param_name == "Centre Frequency (MHz)":
                    self.parameters["FFT"]["Centre_frequency(MHz)"] = numeric_value

                self.save_parameters()
                print(f"FFT parameter '{param_name}' updated to {numeric_value}")
            except ValueError:
                print(f"Invalid value for parameter '{param_name}': {new_value}")

        set_button.clicked.connect(update_value)
        param_layout.addWidget(set_button)
        param_layout.addStretch()

        self.config_tree.setItemWidget(param_item, 1, param_widget)

    def add_dac_adc_parameter(
        self, parent_item, param_name, current_value, range_tuple
    ):
        """添加DAC/ADC参数行"""
        param_item = QTreeWidgetItem([param_name])
        parent_item.addChild(param_item)

        # 创建参数控制卡片
        param_widget = QWidget()
        param_layout = QHBoxLayout(param_widget)
        param_layout.setContentsMargins(2, 2, 2, 2)
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

        # 输入框
        input_widget = LineEdit(param_widget)
        input_widget.setText(str(current_value))
        input_widget.setPlaceholderText(f"Enter {param_name}")
        input_widget.setFixedWidth(100)
        param_layout.addWidget(input_widget)

        # Set按钮
        set_button = PushButton("Set", param_widget)
        set_button.setFixedWidth(50)
        setCustomStyleSheet(set_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE)

        def update_value():
            new_value = input_widget.text()
            try:
                numeric_value = float(new_value)

                # 检查范围
                min_val, max_val, step = range_tuple
                if not (min_val <= numeric_value <= max_val):
                    print(f"Value out of range [{min_val}, {max_val}]")
                    return

                # 更新显示
                value_label.setText(str(numeric_value))
                print(f"Parameter '{param_name}' updated to {numeric_value}")

                # TODO: 这里可以添加保存到参数文件的逻辑
            except ValueError:
                print(f"Invalid value for parameter '{param_name}': {new_value}")

        set_button.clicked.connect(update_value)
        param_layout.addWidget(set_button)
        param_layout.addStretch()

        self.config_tree.setItemWidget(param_item, 1, param_widget)

    def save_parameters(self):
        """保存参数到JSON文件"""
        config_path = os.path.join(
            os.path.dirname(__file__), "../config/parameters.json"
        )
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.parameters, f, indent=2, ensure_ascii=False)
        print("Parameters saved successfully!")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = ConfigInterface()
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())
