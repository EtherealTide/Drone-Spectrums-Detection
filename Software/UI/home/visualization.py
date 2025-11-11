from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis
from PyQt6.QtGui import QPainter, QColor, QImage, QPixmap
from qfluentwidgets import CardWidget, BodyLabel
from qfluentwidgets import FluentIcon as FIF
from ..utils.component import Component
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import logging


class HomeVisualizationCard(QWidget):
    def __init__(self, parent=None, data_processor=None, detector=None):
        super().__init__(parent)
        self.setObjectName("HomeVisualizationCard")
        self.component = Component()

        # 数据处理器引用（用于获取频谱数据）
        self.data_processor = data_processor

        # 算法检测器引用（用于获取检测结果图像）
        self.detector = detector

        # 统计信息
        self.frame_displayed = 0
        self.last_frame_id = -1
        self.update_count = 0
        self.skip_count = 0

        # 检测统计
        self.last_detection_count = 0

        # 频率轴参数
        self.center_freq = 2400  # MHz
        self.sample_rate = 20  # MHz

        self.setup_ui()

        # 定时器用于更新可视化
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)

    def start_update(self):
        """启动可视化更新"""
        self.update_timer.start(25)  # 25ms = 40fps

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

        # 下方卡片 - 检测结果图像
        bottom_chart_layout, bottom_chart_card = self.component.create_card(
            self, height=300
        )

        # 检测结果图像标签
        self.detection_label = QLabel(bottom_chart_card)
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("background-color: black;")
        self.detection_label.setMinimumHeight(200)
        bottom_chart_layout.addWidget(self.detection_label)

        # 检测统计信息标签
        self.detection_stats_label = BodyLabel(bottom_chart_card)
        self.detection_stats_label.setText("等待检测数据...")
        bottom_chart_layout.addWidget(self.detection_stats_label)

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
        self.axis_y.setTitleText("Power (Normalized)")
        self.axis_y.setRange(0, 1)  # 归一化数据范围
        chart.addAxis(self.axis_y, Qt.AlignmentFlag.AlignLeft)
        self.spectrum_series.attachAxis(self.axis_y)

        chart.legend().setAlignment(Qt.AlignmentFlag.AlignBottom)

        return chart

    def update_visualization(self):
        """更新可视化（频谱图和检测结果）"""
        if self.data_processor is None:
            return

        # 获取数据处理统计信息
        stats = self.data_processor.get_stats()
        if stats is None:
            return

        # 检查是否是新帧
        frame_id = stats["frame_id"]
        if frame_id == self.last_frame_id:
            self.skip_count += 1
            return  # 同一帧，不重复处理

        self.last_frame_id = frame_id
        self.frame_displayed += 1
        self.update_count += 1

        # 获取最新频谱数据
        spectrum = self.data_processor.get_latest_spectrum()
        if spectrum is not None:
            # 每次都更新频谱图
            self.update_spectrum(spectrum)

        # 每帧都更新检测结果图像（如果检测器可用）
        if self.detector is not None:
            self.update_detection_image()

    def update_spectrum(self, spectrum_data):
        """更新频谱图"""
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

        # 动态调整Y轴范围（每10帧调整一次）
        if self.update_count % 10 == 0:
            max_power = np.max(spectrum_data)
            min_power = np.min(spectrum_data)
            margin = max(0.05, (max_power - min_power) * 0.1)  # 至少5%的margin
            self.axis_y.setRange(max(0, min_power - margin), min(1, max_power + margin))

    def update_detection_image(self):
        """更新检测结果图像"""
        try:
            # 从检测器获取带框的图像
            detection_image = self.detector.get_detection_image()

            if detection_image is None:
                # 无检测图像，显示提示
                self.detection_label.setText("等待检测结果...")
                return

            # 转换numpy数组为QImage并显示
            h, w = detection_image.shape[:2]

            # 确保是RGB格式
            if len(detection_image.shape) == 2:  # 灰度图
                detection_image = np.stack([detection_image] * 3, axis=-1)

            # 创建QImage
            qimage = QImage(
                detection_image.data,
                w,
                h,
                w * 3,
                QImage.Format.Format_RGB888,
            )

            # 缩放到标签大小并显示
            pixmap = QPixmap.fromImage(qimage).scaled(
                self.detection_label.width(),
                self.detection_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,  # 保持宽高比
                Qt.TransformationMode.SmoothTransformation,  # 平滑缩放
            )
            self.detection_label.setPixmap(pixmap)

            # 每5帧更新一次检测统计信息
            if self.update_count % 5 == 0:
                self.update_detection_stats()

        except Exception as e:
            logging.error(f"更新检测图像失败: {e}", exc_info=True)
            self.detection_label.setText(f"显示错误: {str(e)}")

    def update_detection_stats(self):
        """更新检测统计信息"""
        if self.detector is None:
            return

        try:
            # 获取检测统计
            detection_stats = self.detector.get_detection_stats()

            # 获取当前检测结果
            detection_results = self.detector.get_detection_results()

            # 计算检测FPS
            current_count = detection_stats.get("detection_count", 0)
            detection_delta = current_count - self.last_detection_count
            self.last_detection_count = current_count

            # 构建统计信息文本
            stats_text = f"""
            <p><b>检测次数:</b> {detection_stats.get('total_detections', 0)}</p>
            <p><b>当前目标:</b> {detection_stats.get('current_objects', 0)}</p>
            <p><b>总目标数:</b> {detection_stats.get('total_objects', 0)}</p>
            """

            # 显示当前检测到的目标详情
            if detection_results:
                stats_text += "<p><b>检测详情:</b></p>"
                for i, result in enumerate(detection_results[:3]):  # 最多显示3个
                    stats_text += (
                        f"<p>  {i+1}. {result['class_name']}: "
                        f"{result['confidence']:.2f}</p>"
                    )
                if len(detection_results) > 3:
                    stats_text += f"<p>  ... 还有 {len(detection_results)-3} 个</p>"
            else:
                stats_text += "<p><b>当前无目标</b></p>"

            self.detection_stats_label.setText(stats_text)

        except Exception as e:
            logging.error(f"更新检测统计失败: {e}", exc_info=True)
            self.detection_stats_label.setText(f"统计错误: {str(e)}")

    def update_config(self, center_freq, sample_rate):
        """更新频率配置"""
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.axis_x.setRange(
            center_freq - sample_rate / 2, center_freq + sample_rate / 2
        )
