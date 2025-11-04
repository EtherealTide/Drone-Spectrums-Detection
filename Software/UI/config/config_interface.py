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
from PyQt6.QtCore import Qt
from .parameter_table import ParameterTable
from ..utils.custom_style import ADD_BUTTON_STYLE, CONFIRM_BUTTON_STYLE
from ..utils.component import Component


class ConfigInterface(QWidget):  # 继承自 QWidget
    def __init__(self, parent=None):
        super().__init__(parent)  # 调用父类构造函数

        self.setObjectName("ConfigInterface")  # 设置对象名称
        # 初始化参数表
        self.parameter_table = ParameterTable("parameters.txt")
        self.setup_ui()

    def setup_ui(self):
        # 加载组件创建器
        component = Component()
        # 设置主布局
        layout = QVBoxLayout(self)  # 整个界面采用垂直布局
        system_state_card_layout, system_state_card = component.create_card(
            self, height=80
        )  # 创建系统状态卡片,注意传入self作为父组件
        # 创建开关按钮，用于定义系统连接
        communication_connect_switch = component.create_switch_button(
            self, "connect", "disconnect"
        )
        # 连接信号槽，每次状态改变时，发送指令
        communication_connect_switch.checkedChanged.connect(
            lambda checked: print("Communication Connect:", checked)
        )
        system_state_card_layout.addWidget(communication_connect_switch)
        ip_port_label = component.create_label(
            self, "IP: 192.168.1.1, Port: 8080", "#000000", "#FFFFFF"
        )

        system_state_card_layout.addWidget(ip_port_label)
        layout.addWidget(system_state_card)
        # 创建卡片组件
        slave_adc_config_card_layout, slave_adc_config_card = component.create_card(
            self, height=500, layout_type="QGridLayout"
        )
        row, col = 0, 0
        for param in self.parameter_table.parameters.keys():
            # 在输入框左侧显示当前值
            current_fre, current_phase = self.parameter_table.get_parameter(
                param
            ).split("/")
            param_name_label = component.create_label(
                self,
                f"{param}+ Frequency: {current_fre}, Phase: {current_phase}",
                "#000000",
                "#FFFFFF",
                alignment=Qt.AlignmentFlag.AlignVCenter
                | Qt.AlignmentFlag.AlignRight,  # 垂直居中靠右对齐
            )
            # 创建输入框
            fre_line_edit = component.create_line_edit(
                self, placeholder=f"Set {param} Frequency", width=200
            )
            confirm_button = PushButton("Set", self)
            confirm_button.setFixedWidth(80)
            # 点击确认按钮时，更新参数表中的值
            confirm_button.clicked.connect(
                lambda _, p=param, le=fre_line_edit, lbl=param_name_label: self.update_parameter(
                    p, le.text(), lbl
                )
            )
            phase_line_edit = component.create_line_edit(
                self, placeholder=f"Set {param} Phase", width=200
            )
            # 创建确认按钮
            confirm_button = PushButton("Set", self)
            # 设置按钮宽度
            confirm_button.setFixedWidth(80)
            # 点击确认按钮时，更新参数表中的值
            confirm_button.clicked.connect(
                lambda _, p=param, le=fre_line_edit, lbl=param_name_label: self.update_parameter(
                    p, le.text(), lbl
                )
            )
            # 函数解析：lambda是匿名函数，_表示忽略的参数（这里是点击事件），p、le、lbl分别绑定当前的参数名、输入框和标签，：后面的函数体调用update_parameter方法更新参数
            # 设置为自定义的样式表
            setCustomStyleSheet(
                confirm_button, CONFIRM_BUTTON_STYLE, CONFIRM_BUTTON_STYLE
            )
            config_card_layout.addWidget(label, row, col)
            config_card_layout.addWidget(line_edit, row, col + 1)
            config_card_layout.addWidget(confirm_button, row, col + 2)
            # 弹性布局调整
            # config_card_layout.setRowStretch(row, 1)
            col += 3  # Move to the next column
            if col >= 6:  # If we have filled 3 columns, move to the next row
                col = 0
                row += 1
        # 将卡片组件添加到主布局中
        layout.addWidget(config_card)
        # 设置样式表
        self.setStyleSheet("#CommandInterface { background: white; }")
        self.resize(1280, 720)  # 设置初始大小

    def update_parameter(self, param, value, label):
        try:
            # 更新参数表中的值
            self.parameter_table.set_parameter(param, float(value))
            # 更新界面上的显示
            label.setText(f"{param}: {value}")
            print(f"Parameter '{param}' updated to {value}.")
        except ValueError:
            print(f"Invalid value for parameter '{param}': {value}")


if __name__ == "__main__":

    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)  # 创建应用程序实例，传入命令行参数
    window = ConfigInterface()  # 创建主窗口实例
    window.show()  # 显示主窗口
    sys.exit(app.exec())  # 运行应用程序主循环
