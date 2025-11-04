from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis, QBarCategoryAxis

from ..utils.component import Component
import os


class VisualizationInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("VisualizationInterface")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        chart_layout, chart_card = Component().create_card(self, height=600)

        # 使用 Qt Charts 原生折线图
        x_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        y_values = [820, 932, 901, 934, 1290, 1330, 1320]

        series = QLineSeries()
        for i, y in enumerate(y_values):
            series.append(i, y)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Line Chart Example")
        chart.legend().hide()

        # 类别型 X 轴
        axisX = QBarCategoryAxis()
        axisX.append(x_labels)
        chart.addAxis(axisX, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axisX)

        # 数值型 Y 轴
        axisY = QValueAxis()
        ymin, ymax = min(y_values), max(y_values)
        padding = max(1, int((ymax - ymin) * 0.1))
        axisY.setRange(ymin - padding, ymax + padding)
        chart.addAxis(axisY, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axisY)

        chart_view = QChartView(chart)
        chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)

        chart_layout.addWidget(chart_view)
        layout.addWidget(chart_card)

        self.setStyleSheet("#VisualizationInterface { background: white; }")
        self.resize(1280, 720)


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = VisualizationInterface()
    window.show()
    sys.exit(app.exec())
