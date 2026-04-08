from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGridLayout
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen

from UI.settings.theme_manager import get_theme_manager

class PerformanceUI(QWidget):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        # Number of ticks inside the window
        self.max_points = 100
        
            
        self.window_duration_seconds = (self.max_points * self.manager.timer_time) / 1000.0
        
        self.x_counter = 0
        
        self.theme_manager = get_theme_manager()
        
        self._setup_ui()
        if self.manager is not None:
            self.manager.stats_updated.connect(self.update_charts)
            
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        self.title = QLabel("System Performance Dashboard")
        title_font = self.title.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title)

        grid = QGridLayout()
        layout.addLayout(grid)

        # Create charts
        self.recv_chart = self._create_chart("Data Receive FPS")
        self.det_chart = self._create_chart("YOLO Detection FPS")
        self.cpu_chart = self._create_chart("CPU Utilization (%)")
        self.gpu_chart = self._create_chart("GPU Utilization (%)")
        
        # Series
        self.recv_series = QLineSeries()
        self.recv_series.setName("FPS")
        self.det_series = QLineSeries()
        self.det_series.setName("FPS")
        self.cpu_series = QLineSeries()
        self.cpu_series.setName("%")
        self.gpu_series = QLineSeries()
        self.gpu_series.setName("%")
        
        self.recv_chart.addSeries(self.recv_series)
        self.det_chart.addSeries(self.det_series)
        self.cpu_chart.addSeries(self.cpu_series)
        self.gpu_chart.addSeries(self.gpu_series)
        
        self.axis_x_recv, self.axis_y_recv = self._setup_axes(self.recv_chart, "Time (s)", "FPS", 40000, 100000)
        self.axis_x_det, self.axis_y_det = self._setup_axes(self.det_chart, "Time (s)", "FPS", 0, 100)
        self.axis_x_cpu, self.axis_y_cpu = self._setup_axes(self.cpu_chart, "Time (s)", "%", 0, 100)
        self.axis_x_gpu, self.axis_y_gpu = self._setup_axes(self.gpu_chart, "Time (s)", "%", 0, 100)
        
        recv_view = QChartView(self.recv_chart)
        recv_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        recv_view.setStyleSheet("background: transparent;")
        
        det_view = QChartView(self.det_chart)
        det_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        det_view.setStyleSheet("background: transparent;")

        cpu_view = QChartView(self.cpu_chart)
        cpu_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        cpu_view.setStyleSheet("background: transparent;")
        
        gpu_view = QChartView(self.gpu_chart)
        gpu_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        gpu_view.setStyleSheet("background: transparent;")
        
        grid.addWidget(recv_view, 0, 0)
        grid.addWidget(det_view, 0, 1)
        grid.addWidget(cpu_view, 1, 0)
        grid.addWidget(gpu_view, 1, 1)
        
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Statistics to Excel (CSV)")
        self.save_btn.setMinimumHeight(40)
        if self.manager is not None:
            self.save_btn.clicked.connect(self.manager.save_to_excel)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)

    def apply_palette(self, palette: dict):
        bg_color = QColor(palette.get('card_bg', palette.get('panel_bg', '#ffffff')))
        text_color = QColor(palette.get('text_primary', '#000000'))
        grid_color = QColor(palette.get('card_border', '#e0e0e0'))
        
        self.setStyleSheet(f"background-color: {palette.get('window_bg', '#f5f5f5')};")
        self.title.setStyleSheet(f"color: {text_color.name()}; background-color: transparent;")
        
        for chart in [self.recv_chart, self.det_chart, self.cpu_chart, self.gpu_chart]:
            chart.setBackgroundBrush(bg_color)
            chart.setTitleBrush(text_color)
            if chart.theme() != QChart.ChartTheme.ChartThemeLight:
                pass
            
            for ax in chart.axes():
                ax.setLabelsBrush(text_color)
                ax.setTitleBrush(text_color)
                grid_pen = QPen(grid_color)
                grid_pen.setWidth(1)
                ax.setGridLinePen(grid_pen)
                
                # Make axis outline visible against background
                line_pen = QPen(text_color)
                line_pen.setWidth(1)
                ax.setLinePen(line_pen)
        
    def _create_chart(self, title):
        chart = QChart()
        chart.setTitle(title)
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)
        chart.setBackgroundRoundness(0)
        chart.margins().setBottom(0)
        chart.legend().hide()
        return chart
        
    def _setup_axes(self, chart, title_x, title_y, min_y, max_y):
        axis_x = QValueAxis()
        axis_x.setTitleText(title_x)
        axis_x.setRange(0, self.window_duration_seconds)
        axis_x.setLabelFormat("%.1f")
        
        axis_y = QValueAxis()
        axis_y.setTitleText(title_y)
        axis_y.setRange(min_y, max_y)
        axis_y.setLabelFormat("%d")
        
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)
            
        return axis_x, axis_y
            
    def update_charts(self, stats: dict):
        self.x_counter += 1
        
        # Convert counter to actual time in seconds
        current_time_sec = (self.x_counter * self.manager.timer_time) / 1000.0
        
        x = current_time_sec
        
        self.recv_series.append(x, stats["recv_fps"])
        self.det_series.append(x, stats["det_fps"])
        self.cpu_series.append(x, stats["cpu_util"])
        self.gpu_series.append(x, stats["gpu_util"])
        
        if self.recv_series.count() > self.max_points:
            self.recv_series.removePoints(0, 1)
            self.det_series.removePoints(0, 1)
            self.cpu_series.removePoints(0, 1)
            self.gpu_series.removePoints(0, 1)
            
        if self.x_counter > self.max_points:
            window_start = x - self.window_duration_seconds
            self.axis_x_recv.setRange(window_start, x)
            self.axis_x_det.setRange(window_start, x)
            self.axis_x_cpu.setRange(window_start, x)
            self.axis_x_gpu.setRange(window_start, x)
        else:
            self.axis_x_recv.setRange(0, self.window_duration_seconds)
            self.axis_x_det.setRange(0, self.window_duration_seconds)
            self.axis_x_cpu.setRange(0, self.window_duration_seconds)
            self.axis_x_gpu.setRange(0, self.window_duration_seconds)
        
        # if stats["recv_fps"] > self.axis_y_recv.max() * 0.9:
        #     self.axis_y_recv.setRange(0, (stats["recv_fps"] // 10 + 2) * 10)
        # if stats["det_fps"] > self.axis_y_det.max() * 0.9:
        #     self.axis_y_det.setRange(0, (stats["det_fps"] // 10 + 2) * 10)
