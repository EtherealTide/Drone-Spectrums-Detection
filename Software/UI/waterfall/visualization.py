from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import logging

from ..utils.component import Component, BodyLabel
from ..settings.theme_manager import get_theme_manager

logger = logging.getLogger(__name__)


class WaterfallVisualizationCard(QWidget):
    """Shows detector result image; falls back to raw waterfall when detector unavailable."""

    def __init__(self, parent=None, data_processor=None, detector=None, state=None):
        super().__init__(parent)
        self.setObjectName("WaterfallVisualizationCard")
        self.component = Component()
        self.theme_manager = get_theme_manager()
        self.data_processor = data_processor
        self.detector = detector
        self.state = state
        self.frame_displayed = 0

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)

        self.setup_ui()
        self.theme_manager.paletteChanged.connect(self.apply_palette)
        self.apply_palette(self.theme_manager.palette)

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(8)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        viz_layout, viz_card = self.component.create_card(self, height=520)
        self.result_label = QLabel(viz_card)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setMinimumHeight(420)
        self.result_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.result_label.setStyleSheet("background-color: black;")
        viz_layout.addWidget(self.result_label)
        layout.addWidget(viz_card)

        # 统计卡片
        stats_layout, stats_card = self.component.create_card(self, height=260)

        self.stats_label = BodyLabel(stats_card)
        self.stats_label.setWordWrap(True)
        self.stats_label.setAlignment(
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft
        )
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_card)

        layout.setStretch(0, 3)
        layout.setStretch(1, 1)

    def start_update(self):
        self.update_timer.start(20)

    def stop_update(self):
        self.update_timer.stop()

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #WaterfallVisualizationCard {{
                background-color: {palette['window_bg']};
            }}
            """
        )

        # 为统计标签设置样式
        stats_label_style = f"""
            QLabel {{
                color: {palette['text_primary']};
                background-color: {palette['card_bg']};
                padding: 5px;
            }}
        """
        self.stats_label.setStyleSheet(stats_label_style)

    def update_visualization(self):
        try:
            detection_image = None
            detection_stats = {}
            if self.detector:
                detection_image = self.detector.get_detection_image()
                detection_stats = self.detector.get_detection_stats()
                scanning_controller_stats = (
                    self.detector.scanning_controller.get_status()
                )
            if detection_image is None and self.data_processor:
                detection_image = self.data_processor.get_waterfall_image()

            if detection_image is None or detection_image.size == 0:
                self.result_label.setText("等待检测/瀑布图数据...")
            else:
                # Transpose for UI display (data_process kept transpose for detector)
                if detection_image.ndim == 3:
                    detection_image = np.transpose(detection_image, (1, 0, 2))
                self._update_label(self.result_label, detection_image)
                self.frame_displayed += 1

            processor_stats = (
                self.data_processor.get_stats() if self.data_processor else {}
            )
            self._update_stats(
                processor_stats, detection_stats, scanning_controller_stats
            )

        except Exception as e:
            logger.error(f"获取瀑布图或检测数据失败: {e}", exc_info=True)
            self.result_label.setText(f"显示错误: {e}")

    def _update_label(self, label: QLabel, image_array):
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        image_array = np.ascontiguousarray(image_array)

        height, width, channels = image_array.shape
        bytes_per_line = width * channels

        qimage = QImage(
            image_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(qimage).scaled(
            label.width(),
            label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _update_stats(
        self, stats: dict, detection_stats: dict, scanning_controller_stats: dict
    ):
        sent_frames = getattr(self.state, "sent_frames", 0) if self.state else 0
        received_frames = getattr(self.state, "received_frames", 0) if self.state else 0
        processed_frames = stats.get("frame_id", 0)
        detection_frames = (
            detection_stats.get("detection_count", 0) if detection_stats else 0
        )
        detection_fps = detection_stats.get("fps", 0.0) if detection_stats else 0.0
        freq_range_str = scanning_controller_stats.get(
            "frequency_range_str", ("N/A", "N/A")
        )
        scan_mode = scanning_controller_stats.get("scan_mode", "N/A")
        # 基本统计信息 - 3列布局
        stats_table = f"""
        <table cellpadding='4' cellspacing='0' width='100%'>
            <tr>
                <td width='33%'>发送帧数: {sent_frames}</td>
                <td width='33%'>接收帧数: {received_frames}</td>
                <td width='34%'>数据处理帧数: {processed_frames}</td>
            </tr>
            <tr>
                <td>检测帧数: {detection_frames}</td>
                <td>检测FPS: {detection_fps:.2f}</td>
                <td>显示帧数: {self.frame_displayed}</td>
            </tr>
            <tr>
                <td colspan='2'>频率范围: {freq_range_str}</td>
                <td>当前扫描状态: {scan_mode}</td>
            </tr>
        </table>
        """

        html_parts = [stats_table]

        # 检测结果 - 多列显示
        if detection_stats and self.detector:
            try:
                detection_results = self.detector.get_detection_results()
            except Exception:
                detection_results = []

            if detection_results:
                html_parts.append("<p style='margin-top: 10px;'><b>检测结果:</b></p>")

                # 将检测结果分为多列显示（每列最多显示3个）
                cols = 3  # 显示列数
                rows = []
                for i in range(0, len(detection_results), cols):
                    row_items = detection_results[i : i + cols]
                    row_html = "<tr>"
                    for result in row_items:
                        class_name = result.get("class_name", "未知")
                        confidence = result.get("confidence", 0)
                        row_html += f"<td width='{100//cols}%'>{class_name} {confidence:.2f}</td>"
                    # 填充空单元格
                    for _ in range(cols - len(row_items)):
                        row_html += f"<td width='{100//cols}%'></td>"
                    row_html += "</tr>"
                    rows.append(row_html)

                html_parts.append(
                    "<table cellpadding='2' cellspacing='0' width='100%'>"
                )
                html_parts.extend(rows)
                html_parts.append("</table>")
            else:
                html_parts.append("<p style='margin-top: 10px;'><b>当前无目标</b></p>")

        self.stats_label.setText("".join(html_parts))

    def update_config(self):
        pass
