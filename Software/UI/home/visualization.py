from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSplitter, QSizePolicy
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
    def __init__(self, parent=None, data_processor=None, detector=None, state=None):
        super().__init__(parent)
        self.setObjectName("HomeVisualizationCard")
        self.component = Component()

        # 引用
        self.data_processor = data_processor
        self.detector = detector
        self.state = state

        # 统计信息
        self.frame_displayed = 0
        self.last_frame_id = -1
        self.update_count = 0

        self.setup_ui()

        # 定时器用于更新可视化
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)

    def start_update(self):
        """启动可视化更新"""
        self.update_timer.start(40)  # 40ms = 25fps

    def stop_update(self):
        """停止可视化更新"""
        self.update_timer.stop()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)  # 减小间距
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 顶部对齐

        # 上方图表卡片 - 功率谱
        top_chart_layout, top_chart_card = self.component.create_card(self, height=300)
        top_chart_card.setMinimumHeight(300)  # 设置最小高度
        top_chart_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        self.spectrum_chart = self.create_spectrum_chart()
        self.spectrum_chart_view = QChartView(self.spectrum_chart)
        self.spectrum_chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        top_chart_layout.addWidget(self.spectrum_chart_view)
        layout.addWidget(top_chart_card)

        # 下方卡片 - 使用Splitter分割检测结果图像和统计信息
        bottom_chart_layout, bottom_chart_card = self.component.create_card(
            self, height=400
        )
        bottom_chart_card.setMinimumHeight(350)  # 设置最小高度
        bottom_chart_card.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        # 创建水平分割器
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：检测结果图像
        detection_widget = QWidget()
        detection_layout = QVBoxLayout(detection_widget)
        detection_layout.setContentsMargins(0, 0, 0, 0)

        self.detection_label = QLabel(detection_widget)
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setStyleSheet("background-color: black;")
        self.detection_label.setMinimumSize(300, 300)  # 设置最小尺寸
        self.detection_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        detection_layout.addWidget(self.detection_label)

        bottom_splitter.addWidget(detection_widget)

        # 右侧：检测统计信息
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        stats_layout.setAlignment(Qt.AlignmentFlag.AlignTop)  # 统计信息也顶部对齐

        self.detection_stats_label = BodyLabel(stats_widget)
        self.detection_stats_label.setText("等待检测数据...")
        self.detection_stats_label.setWordWrap(True)
        self.detection_stats_label.setMinimumWidth(150)  # 设置最小宽度
        self.detection_stats_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )  # 文字顶部左对齐
        self.detection_stats_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        stats_layout.addWidget(self.detection_stats_label)
        stats_layout.addStretch()  # 底部添加弹性空间

        bottom_splitter.addWidget(stats_widget)

        # 设置初始比例 7:3
        bottom_splitter.setStretchFactor(0, 7)
        bottom_splitter.setStretchFactor(1, 3)

        bottom_chart_layout.addWidget(bottom_splitter)
        layout.addWidget(bottom_chart_card)

        # 设置上下比例 - 让它们根据内容自适应
        layout.setStretch(0, 1)  # 功率谱
        layout.setStretch(1, 1)  # 检测结果

        self.setStyleSheet(
            """
            #HomeVisualizationCard { 
                background: white; 
            }
            QSplitter::handle {
                background-color: #D0D0D0;
                width: 2px;
            }
            QSplitter::handle:hover {
                background-color: #A0A0A0;
            }
        """
        )

    def create_spectrum_chart(self):
        """创建功率谱图表"""
        chart = QChart()
        chart.setAnimationOptions(
            QChart.AnimationOption.NoAnimation
        )  # 关闭动画以提高性能

        self.spectrum_series = QLineSeries()
        self.spectrum_series.setName("功率谱")
        chart.addSeries(self.spectrum_series)
        # 从state读取频率参数
        left_freq = self.state.spectrum_left_freq
        right_freq = self.state.spectrum_right_freq
        center_freq = self.state.center_frequency
        # 创建X轴（频率轴）
        axis_x = QValueAxis()
        axis_x.setTitleText(f"频率 (MHz)")
        axis_x.setRange(left_freq, right_freq)
        axis_x.setLabelFormat("%.1f")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.spectrum_series.attachAxis(axis_x)
        # 创建Y轴（功率轴）
        axis_y = QValueAxis()
        axis_y.setTitleText("归一化功率 ")
        axis_y.setRange(0, 1)
        axis_y.setLabelFormat("%.0f")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        self.spectrum_series.attachAxis(axis_y)

        return chart

    def update_spectrum(self, spectrum_data):
        """更新频谱图"""
        if spectrum_data is None:
            return
        # 计算频率点
        max_freq = self.state.sample_rate
        self.freq_points = np.linspace(0, max_freq / 1e6, self.state.fft_length)

        # 更新数据点
        points = [
            QPointF(freq, power) for freq, power in zip(self.freq_points, spectrum_data)
        ]
        self.spectrum_series.replace(points)

    def update_visualization(self):
        """更新可视化"""
        self.update_count += 1

        try:
            # 更新频谱图
            if self.data_processor:
                spectrum_data = self.data_processor.get_latest_spectrum()
                if spectrum_data is not None:
                    self.update_spectrum(spectrum_data)
            # 更新检测图像
            if self.detector:
                self.update_detection_image()
                self.frame_displayed += 1
            # 每5帧更新一次统计信息
            if self.update_count % 5 == 0:
                self.update_detection_stats()

        except Exception as e:
            logging.error(f"可视化更新异常: {e}", exc_info=True)

    def update_detection_image(self):
        """更新检测结果图像"""
        try:
            detection_image = self.detector.get_detection_image()

            if detection_image is None:
                self.detection_label.setText("等待检测结果...")
                return

            # 确保是RGB格式
            if len(detection_image.shape) == 2:
                detection_image = np.stack([detection_image] * 3, axis=-1)

            # 转置图像
            detection_image = np.transpose(detection_image, (1, 0, 2))

            # 获取Label尺寸
            label_width = self.detection_label.width()
            label_height = self.detection_label.height()

            h, w = detection_image.shape[:2]
            label_ratio = label_width / label_height
            image_ratio = w / h

            # 裁剪图像以匹配Label比例
            if image_ratio > label_ratio:
                # 图像更宽，裁剪左右
                new_width = int(h * label_ratio)
                start_x = (w - new_width) // 2
                detection_image = detection_image[:, start_x : start_x + new_width, :]
            else:
                # 图像更高，裁剪上下
                new_height = int(w / label_ratio)
                start_y = (h - new_height) // 2
                detection_image = detection_image[start_y : start_y + new_height, :, :]

            # 确保数组是连续的
            detection_image = np.ascontiguousarray(detection_image)

            h, w, c = detection_image.shape
            bytes_per_line = w * c

            # 确保数据类型是uint8
            if detection_image.dtype != np.uint8:
                detection_image = detection_image.astype(np.uint8)

            # 使用tobytes()而不是.data来创建QImage
            qimage = QImage(
                detection_image.tobytes(),
                w,
                h,
                bytes_per_line,
                QImage.Format.Format_RGB888,
            )

            # 缩放到Label大小
            pixmap = QPixmap.fromImage(qimage).scaled(
                label_width,
                label_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.detection_label.setPixmap(pixmap)

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
            # 获取发送数据帧数
            sent_frames = self.state.sent_frames
            # 获取接收数据帧数
            received_frames = self.state.received_frames
            # 获取处理帧数
            processed_frames = self.data_processor.get_stats().get("frame_id", 1)
            # 获取显示帧数
            displayed_frames = self.frame_displayed
            # 获取fps
            detection_fps = detection_stats.get("fps", 0.0)

            # 构建统计信息文本
            stats_text = f"""
            <p><b>发送帧数:</b> {sent_frames}</p>
            <p><b>接收帧数:</b> {received_frames}</p>
            <p><b>数据处理帧数:</b> {processed_frames}</p>
            <p><b>显示帧数:</b> {displayed_frames}</p>
            <p><b>检测FPS:</b> {detection_fps:.2f}</p>
            <p><b>检测次数:</b> {detection_stats.get('total_detections', 0)}</p>
            """

            # 显示当前检测到的目标详情
            if detection_results:
                stats_text += "<p><b>检测详情:</b></p>"
                for i, result in enumerate(detection_results[:3]):
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

    def update_config(self):
        """更新频率配置"""
        # 从state读取最新的频率参数
        left_freq = self.state.spectrum_left_freq
        right_freq = self.state.spectrum_right_freq
        self.axis_x = self.spectrum_chart.axes(Qt.Orientation.Horizontal)[0]
        self.axis_x.setRange(left_freq, right_freq)
