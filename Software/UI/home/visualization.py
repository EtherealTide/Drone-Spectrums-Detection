# home界面的左侧卡片内容，包括上下两张图
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QPieSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor
from qfluentwidgets import CardWidget, BodyLabel, setFont
from qfluentwidgets import FluentIcon as FIF
from ..utils.component import Component


class HomeVisualizationCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HomeVisualizationCard")
        self.component = Component()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 上方图表卡片 - 折线图
        top_chart_layout, top_chart_card = self.component.create_card(self, height=280)
        top_chart = self.create_line_chart()
        top_chart_view = QChartView(top_chart)
        top_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        top_chart_layout.addWidget(top_chart_view)
        layout.addWidget(top_chart_card)

        # 下方图表卡片 - 饼图
        bottom_chart_layout, bottom_chart_card = self.component.create_card(
            self, height=280
        )
        bottom_chart = self.create_pie_chart()
        bottom_chart_view = QChartView(bottom_chart)
        bottom_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        bottom_chart_layout.addWidget(bottom_chart_view)
        layout.addWidget(bottom_chart_card)

        self.setStyleSheet("#HomeVisualizationCard { background: white; }")

    def create_line_chart(self):
        """创建折线图 - 功率谱"""
        series = QLineSeries()
        series.setName("Power Spectrum")

        # 模拟功率谱数据
        import random

        for i in range(100):
            freq = 90 + i * 0.1
            power = -80 + random.uniform(-10, 10)
            # 添加几个峰值
            if abs(freq - 92) < 0.5 or abs(freq - 95) < 0.5 or abs(freq - 98) < 0.5:
                power += 40
            series.append(freq, power)

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("Power Spectrum (dBm)")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

        # X轴
        axis_x = QValueAxis()
        axis_x.setTitleText("Frequency (MHz)")
        axis_x.setLabelFormat("%.1f")
        axis_x.setRange(90, 100)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        # Y轴
        axis_y = QValueAxis()
        axis_y.setTitleText("Power (dBm)")
        axis_y.setRange(-100, 0)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        return chart

    def create_pie_chart(self):
        """创建饼图 - 系统资源分布"""
        series = QPieSeries()
        series.append("DAC Active", 8)
        series.append("ADC Active", 8)
        series.append("FFT Processing", 1)
        series.append("Idle", 3)

        # 设置切片样式
        slice_colors = [
            QColor("#0078D4"),  # 蓝色
            QColor("#107C10"),  # 绿色
            QColor("#D83B01"),  # 橙色
            QColor("#767676"),  # 灰色
        ]

        for i, slice in enumerate(series.slices()):
            slice.setLabelVisible(True)
            slice.setLabel(f"{slice.label()}\n{slice.percentage():.1%}")
            slice.setBrush(slice_colors[i % len(slice_colors)])

        chart = QChart()
        chart.addSeries(series)
        chart.setTitle("System Resources Distribution")
        chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        chart.legend().setAlignment(Qt.AlignmentFlag.AlignRight)

        return chart


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    window = HomeVisualizationCard()
    window.show()
    sys.exit(app.exec())
