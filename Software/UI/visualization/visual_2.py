# 使用基于echarts的可视化界面
from pyecharts import options as opts
from pyecharts.charts import Line, Bar, Pie
from PyQt6.QtWidgets import QWidget, QVBoxLayout
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
from pyecharts.globals import CurrentConfig

# from ..utils.component import Component

# from ..icons.MyFluentIcon import MyFluentIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl
import os

CurrentConfig.ONLINE_HOST = "file:///" + os.path.join(os.getcwd(), "assets", "")


class VisualizationInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VisualizationInterface")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        # chart_layout, chart_card = Component().create_card(self, height=600)
        # 创建一个简单的折线图

        chart = QWebEngineView()
        # 设置 QWebEngineView 背景透明
        chart.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        chart.setStyleSheet("background: transparent;")
        line = (
            Line()
            .add_xaxis(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            .add_yaxis("Example Data", [820, 932, 901, 934, 1290, 1330, 1320])
            .set_global_opts(title_opts=opts.TitleOpts(title="Line Chart Example"))
        )
        line.render("temp_line_chart.html")
        chart.setUrl(QUrl.fromLocalFile(os.path.abspath("temp_line_chart.html")))
        # 将图表渲染为 HTML 文件

        # chart = self.create_line_chart(line)
        # chart_layout.addWidget(chart)
        # layout.addWidget(chart_card)
        layout.addWidget(chart)
        self.setStyleSheet("#VisualizationInterface { background: white; }")
        self.resize(1280, 720)  # 设置初始大小


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    assets_dir = os.path.join(os.getcwd(), "assets")
    if not os.path.exists(assets_dir):
        os.makedirs(assets_dir)
    app = QApplication(sys.argv)
    window = VisualizationInterface()
    window.show()
    sys.exit(app.exec())
