from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap
from qfluentwidgets import CardWidget, BodyLabel
from qfluentwidgets import FluentIcon as FIF
from ..utils.component import Component
import numpy as np
from collections import deque


class HomeVisualizationCard(QWidget):
    def __init__(self, parent=None, data_processor=None):
        super().__init__(parent)
        self.setObjectName("HomeVisualizationCard")
        self.component = Component()

        # 数据处理器引用
        self.data_processor = data_processor

        # 瀑布图参数 - 减少高度以提高性能
        self.waterfall_height = 100  # 从200降到100
        self.waterfall_width = 512  # 固定宽度，用于降采样
        self.waterfall_data = deque(maxlen=self.waterfall_height)

        # 统计信息
        self.frame_displayed = 0
        self.last_frame_id = -1

        # 性能统计
        self.update_count = 0
        self.skip_count = 0

        # 频率轴参数
        self.center_freq = 2400  # MHz
        self.sample_rate = 20  # MHz

        # 预计算的colormap（避免重复计算）
        self.colormap_cache = self._generate_colormap()

        self.setup_ui()

        # 降低更新频率：从50ms改为100ms (10fps)
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)

    def _generate_colormap(self):
        """预生成colormap查找表"""
        colormap = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            val = i / 255.0
            if val < 0.25:
                r = 0
                g = int(val * 4 * 255)
                b = 255
            elif val < 0.5:
                r = 0
                g = 255
                b = int((0.5 - val) * 4 * 255)
            elif val < 0.75:
                r = int((val - 0.5) * 4 * 255)
                g = 255
                b = 0
            else:
                r = 255
                g = int((1.0 - val) * 4 * 255)
                b = 0
            colormap[i] = [r, g, b]
        return colormap

    def start_update(self):
        """启动可视化更新"""
        self.update_timer.start(100)  # 100ms = 10fps

    def stop_update(self):
        """停止可视化更新"""
        self.update_timer.stop()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # 上方图表卡片 - 功率谱
        top_chart_layout, top_chart_card = self.component.create_card(self, height=300)
        self.spectrum_chart = self.create_spectrum_chart()
        self.spectrum_chart_view = QChartView(self.spectrum_chart)
        self.spectrum_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        top_chart_layout.addWidget(self.spectrum_chart_view)
        layout.addWidget(top_chart_card)

        # 下方图表卡片 - 瀑布图
        bottom_chart_layout, bottom_chart_card = self.component.create_card(
            self, height=260
        )

        # 瀑布图显示标签
        self.waterfall_label = QLabel(bottom_chart_card)
        self.waterfall_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.waterfall_label.setStyleSheet("background-color: black;")
        self.waterfall_label.setMinimumHeight(180)
        bottom_chart_layout.addWidget(self.waterfall_label)

        # 统计信息标签
        self.stats_label = BodyLabel(bottom_chart_card)
        self.stats_label.setText("等待数据...")
        bottom_chart_layout.addWidget(self.stats_label)

        layout.addWidget(bottom_chart_card)

        self.setStyleSheet("#HomeVisualizationCard { background: white; }")

    def create_spectrum_chart(self):
        """创建功率谱图表"""
        self.spectrum_series = QLineSeries()
        self.spectrum_series.setName("Power Spectrum")

        chart = QChart()
        chart.addSeries(self.spectrum_series)
        chart.setTitle("Real-time Power Spectrum")
        chart.setAnimationOptions(QChart.AnimationOption.NoAnimation)

        # X轴 - 频率
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("Frequency (MHz)")
        self.axis_x.setLabelFormat("%.1f")
        self.axis_x.setRange(
            self.center_freq - self.sample_rate / 2,
            self.center_freq + self.sample_rate / 2,
        )
        chart.addAxis(self.axis_x, Qt.AlignmentFlag.AlignBottom)
        self.spectrum_series.attachAxis(self.axis_x)

        # Y轴 - 功率
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("Power (dB)")
        self.axis_y.setRange(-100, 20)
        chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.spectrum_series.attachAxis(self.axis_y)

        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        return chart

    def update_visualization(self):
        """更新可视化（频谱图和瀑布图）"""
        if self.data_processor is None:
            return

        # 获取最新数据
        latest_data = self.data_processor.get_latest_data()

        if latest_data is None:
            return

        # 检查是否是新帧
        frame_id = latest_data["frame_id"]
        if frame_id == self.last_frame_id:
            self.skip_count += 1
            return  # 同一帧，不重复处理

        self.last_frame_id = frame_id
        self.frame_displayed += 1
        self.update_count += 1

        # 提取频谱数据
        spectrum = latest_data["spectrum"]

        # 只在每2帧更新一次瀑布图（进一步降低频率）
        if self.update_count % 2 == 0:
            self.update_waterfall(spectrum)

        # 每次都更新频谱图
        self.update_spectrum(spectrum)

        # 每5帧更新一次统计信息
        if self.update_count % 5 == 0:
            self.update_stats(latest_data)

    def update_spectrum(self, spectrum_data):
        """更新频谱图 - 修复版（使用QPointF）"""
        # 降采样到固定点数（如500个点）
        target_points = 500
        fft_length = len(spectrum_data)
        downsample_factor = max(1, fft_length // target_points)

        # 构建QPointF列表
        points = []
        freq_resolution = self.sample_rate / fft_length
        freq_start = self.center_freq - self.sample_rate / 2

        for i in range(0, fft_length, downsample_factor):
            freq = freq_start + i * freq_resolution
            power = spectrum_data[i]
            points.append(QPointF(freq, power))

        # 使用replace替代clear+append（更高效）
        self.spectrum_series.replace(points)

        # 动态调整Y轴范围（但不是每次都调整）
        if self.update_count % 10 == 0:
            max_power = np.max(spectrum_data)
            min_power = np.min(spectrum_data)
            margin = max(5, (max_power - min_power) * 0.1)  # 至少5dB的margin
            self.axis_y.setRange(min_power - margin, max_power + margin)

    def update_waterfall(self, spectrum_data):
        """更新瀑布图 - 优化版"""
        # 降采样到固定宽度
        fft_length = len(spectrum_data)
        downsample_factor = max(1, fft_length // self.waterfall_width)
        downsampled = spectrum_data[::downsample_factor][: self.waterfall_width]

        # 添加新的频谱线
        self.waterfall_data.append(downsampled)

        # 如果数据不足，返回
        if len(self.waterfall_data) < 5:
            return

        # 转换为numpy数组
        waterfall_array = np.array(list(self.waterfall_data), dtype=np.float32)
        height, width = waterfall_array.shape

        # 归一化到0-255（使用固定范围避免频繁计算）
        vmin = -80  # 固定最小值
        vmax = -20  # 固定最大值
        normalized = np.clip(
            (waterfall_array - vmin) / (vmax - vmin) * 255, 0, 255
        ).astype(np.uint8)

        # 使用预计算的colormap
        colored_image = self.colormap_cache[normalized]

        # 转换为QImage
        qimage = QImage(
            colored_image.data, width, height, width * 3, QImage.Format.Format_RGB888
        )

        # 缩放到合适大小（使用更快的缩放算法）
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.waterfall_label.width(),
            self.waterfall_label.height(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.FastTransformation,  # 使用快速变换
        )

        self.waterfall_label.setPixmap(pixmap)

    def update_stats(self, data):
        """更新统计信息"""
        drop_rate = (
            (self.skip_count / (self.update_count + self.skip_count) * 100)
            if self.update_count > 0
            else 0
        )

        stats_text = f"""
        <p><b>显示帧数:</b> {self.frame_displayed}</p>
        <p><b>处理帧数:</b> {data['frame_id']}</p>
        <p><b>跳过帧数:</b> {self.skip_count} ({drop_rate:.1f}%)</p>
        <p><b>最大值:</b> {data['max_value']:.2f} dB</p>
        <p><b>最小值:</b> {data['min_value']:.2f} dB</p>
        """
        self.stats_label.setText(stats_text)

    def update_config(self, center_freq, sample_rate):
        """更新频率配置"""
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.axis_x.setRange(
            center_freq - sample_rate / 2, center_freq + sample_rate / 2
        )
