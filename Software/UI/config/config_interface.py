from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel
from qfluentwidgets import PushButton, setCustomStyleSheet, SingleDirectionScrollArea
from PyQt6.QtCore import Qt
from .parameter_table import ParameterTable
from ..utils.custom_style import ADD_BUTTON_STYLE, CONFIRM_BUTTON_STYLE
from ..utils.component import Component


class ConfigInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ConfigInterface")
        self.parameter_table = ParameterTable("parameters.json")
        self.setup_ui()

    def setup_ui(self):
        component = Component()
        # 主布局 - 包含滚动区域
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create a scroll area to hold all configuration cards
        scroll_area = SingleDirectionScrollArea(
            orient=Qt.Orientation.Vertical, parent=self
        )
        scroll_widget = QWidget()
        scroll_widget.setObjectName("ConfigScrollWidget")
        layout = QVBoxLayout(scroll_widget)
        # 系统状态卡片
        system_state_card_layout, system_state_card = component.create_card(
            self, height=80
        )
        communication_connect_switch = component.create_switch_button(
            self, "connect", "disconnect"
        )
        communication_connect_switch.checkedChanged.connect(
            lambda checked: print("Communication Connect:", checked)
        )
        system_state_card_layout.addWidget(communication_connect_switch)
        ip_port_label = component.create_label(
            self, "IP: 192.168.1.1, Port: 8080", "#000000", "#FFFFFF"
        )
        system_state_card_layout.addWidget(ip_port_label)
        layout.addWidget(system_state_card)

        # 创建DAC参数配置卡片
        dac_card = self.create_parameter_group_card("DAC", component, height=500)
        layout.addWidget(dac_card)
        # 创建ADC参数配置卡片
        adc_card = self.create_parameter_group_card("ADC", component, height=500)
        layout.addWidget(adc_card)
        # 创建FFT参数配置卡片
        fft_card = self.create_parameter_group_card("FFT", component, height=150)
        layout.addWidget(fft_card)
        layout.addStretch()  # 添加弹性空间
        scroll_area.setWidget(scroll_widget)
        scroll_area.setWidgetResizable(True)  # 使内容自适应大小
        main_layout.addWidget(scroll_area)
        self.setStyleSheet("#ConfigInterface { background: white; }")
        self.resize(2560, 1440)

    def create_parameter_group_card(
        self, group_name: str, component: Component, height: int = None
    ) -> QWidget:
        """
        创建参数组的调节界面卡片

        Args:
            group_name: 参数组名称（如 "DAC", "FFT"）
            component: Component实例，用于创建UI组件

        Returns:
            包含参数组配置界面的卡片Widget
        """
        # 获取该组的所有参数
        group_data = self.parameter_table.get_parameter(group_name, default={})

        # 创建卡片 - 不指定高度，或者指定一个足够大的高度
        card_layout, card_widget = component.create_card(
            self, height=height, layout_type="QVBoxLayout"
        )

        # 添加组名标题（居中显示）
        title_label = component.create_label(
            self,
            group_name,
            "#000000",
            "#F0F0F0",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        title_label.setStyleSheet(
            "font-size: 18px; font-weight: bold; padding: 10px; "
            "background-color: #F0F0F0; border-radius: 5px;"
        )
        card_layout.addWidget(title_label)

        # 创建网格布局用于参数行
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        row = 0
        # 遍历该组下的所有子项
        for sub_name, sub_params in group_data.items():
            if not isinstance(sub_params, dict):
                # 如果是简单键值对（如FFT的Length），直接处理
                self._add_simple_parameter_row(
                    grid_layout, row, group_name, sub_name, sub_params, component
                )
                row += 1
            else:
                # 如果是包含多个字段的子项（如DAC00）
                self._add_complex_parameter_row(
                    grid_layout, row, group_name, sub_name, sub_params, component
                )
                row += 1

        card_layout.addLayout(grid_layout)
        card_layout.addStretch()  # 添加弹性空间

        return card_widget

    def _add_simple_parameter_row(
        self,
        layout: QGridLayout,
        row: int,
        group: str,
        param_name: str,
        current_value,
        component: Component,
    ):
        """添加简单参数行（一个参数一个值）"""
        col = 0

        # 参数名标签
        name_label = component.create_label(
            self,
            f"{param_name}:",
            "#000000",
            "#FFFFFF",
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        )
        name_label.setFixedWidth(150)
        layout.addWidget(name_label, row, col)
        col += 1

        # 当前值显示
        value_label = component.create_label(
            self,
            str(current_value),
            "#000000",
            "#E8F4F8",
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter,
        )
        value_label.setFixedWidth(100)
        value_label.setObjectName(f"{group}_{param_name}_value")
        layout.addWidget(value_label, row, col)
        col += 1

        # 输入框
        line_edit = component.create_line_edit(
            self, placeholder=f"New {param_name}", width=150
        )
        layout.addWidget(line_edit, row, col)
        col += 1

        # 确认按钮
        confirm_button = PushButton("Set", self)
        confirm_button.setFixedWidth(80)
        confirm_button.clicked.connect(
            lambda: self._update_simple_parameter(
                group, param_name, line_edit.text(), value_label
            )
        )
        setCustomStyleSheet(confirm_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE)
        layout.addWidget(confirm_button, row, col)

    def _add_complex_parameter_row(
        self,
        layout: QGridLayout,
        row: int,
        group: str,
        sub_name: str,
        sub_params: dict,
        component: Component,
    ):
        """添加复杂参数行（一个子项包含多个字段）"""
        col = 0

        # 子项名标签（如DAC00）
        name_label = component.create_label(
            self,
            f"{sub_name}:",
            "#000000",
            "#FFFFFF",
            alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        )
        name_label.setFixedWidth(150)
        layout.addWidget(name_label, row, col)
        col += 1

        # 为每个字段创建控件
        for field_name, current_value in sub_params.items():
            # 当前值显示
            value_label = component.create_label(
                self,
                f"{field_name}: {current_value}",
                "#000000",
                "#E8F4F8",
                alignment=Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter,
            )
            value_label.setFixedWidth(120)
            value_label.setObjectName(f"{group}_{sub_name}_{field_name}_value")
            layout.addWidget(value_label, row, col)
            col += 1

            # 输入框
            line_edit = component.create_line_edit(
                self, placeholder=f"New {field_name}", width=120
            )
            layout.addWidget(line_edit, row, col)
            col += 1

            # 确认按钮
            confirm_button = PushButton("Set", self)
            confirm_button.setFixedWidth(80)
            confirm_button.clicked.connect(
                lambda checked=False, g=group, sn=sub_name, fn=field_name, le=line_edit, lbl=value_label: self._update_complex_parameter(
                    g, sn, fn, le.text(), lbl
                )
            )
            setCustomStyleSheet(
                confirm_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE
            )
            layout.addWidget(confirm_button, row, col)
            col += 1

    def _update_simple_parameter(
        self, group: str, param: str, value: str, label: QLabel
    ):
        """更新简单参数"""
        try:
            # 尝试转换为数字
            try:
                numeric_value = float(value)
                if numeric_value.is_integer():
                    numeric_value = int(numeric_value)
            except ValueError:
                numeric_value = value  # 保持字符串

            # 更新参数表（需要修改ParameterTable以支持简单参数）
            self.parameter_table.parameters[group][param] = numeric_value
            self.parameter_table.save_parameters()

            # 更新显示
            label.setText(str(numeric_value))
            print(f"Parameter '{group}.{param}' updated to {numeric_value}")
        except Exception as e:
            print(f"Error updating parameter: {e}")

    def _update_complex_parameter(
        self, group: str, sub_name: str, field: str, value: str, label: QLabel
    ):
        """更新复杂参数"""
        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                numeric_value = int(numeric_value)

            # 更新参数表
            self.parameter_table.set_parameter(group, sub_name, field, numeric_value)
            self.parameter_table.save_parameters()

            # 更新显示
            label.setText(f"{field}: {numeric_value}")
            print(f"Parameter '{group}.{sub_name}.{field}' updated to {numeric_value}")
        except ValueError:
            print(f"Invalid value for parameter: {value}")
        except Exception as e:
            print(f"Error updating parameter: {e}")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = ConfigInterface()
    window.show()
    sys.exit(app.exec())
