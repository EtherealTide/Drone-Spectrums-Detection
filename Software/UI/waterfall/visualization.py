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

        stats_layout, stats_card = self.component.create_card(self, height=260)
        self.stats_label = BodyLabel(stats_card)
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumHeight(180)
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_card)

        layout.setStretch(0, 3)
        layout.setStretch(1, 1)

    def start_update(self):
        self.update_timer.start(40)

    def stop_update(self):
        self.update_timer.stop()

    def apply_palette(self, palette: dict):
        self.setStyleSheet(
            f"""
            #WaterfallVisualizationCard {{
                background-color: {palette['window_bg']};
            }}
            QLabel {{
                color: {palette['text_primary']};
            }}
            """
        )
        self.stats_label.setStyleSheet(f"color: {palette['text_primary']};")

    def update_visualization(self):
        try:
            detection_image = None
            detection_stats = {}
            if self.detector:
                detection_image = self.detector.get_detection_image()
                detection_stats = self.detector.get_detection_stats()

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

            processor_stats = self.data_processor.get_stats() if self.data_processor else {}
            self._update_stats(processor_stats, detection_stats)

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

    def _update_stats(self, stats: dict, detection_stats: dict):
        sent_frames = getattr(self.state, "sent_frames", 0) if self.state else 0
        received_frames = getattr(self.state, "received_frames", 0) if self.state else 0
        processed_frames = stats.get("frame_id", 0)
        detection_frames = detection_stats.get("detection_count", 0) if detection_stats else 0
        detection_fps = detection_stats.get("fps", 0.0) if detection_stats else 0.0
        batch_size = stats.get("batch_size", 0)

        left_col = [
            f"发送帧数: {sent_frames}",
            f"接收帧数: {received_frames}",
            f"数据处理帧数: {processed_frames}",
            f"检测帧数: {detection_frames}",
        ]
        right_col = [
            f"检测FPS: {detection_fps:.2f}",
            f"显示帧数: {self.frame_displayed}",
            f"批处理帧数: {batch_size}",
        ]

        table_rows = []
        max_rows = max(len(left_col), len(right_col))
        for i in range(max_rows):
            lval = left_col[i] if i < len(left_col) else ""
            rval = right_col[i] if i < len(right_col) else ""
            table_rows.append(f"<tr><td>{lval}</td><td>{rval}</td></tr>")

        html_parts = ["<table cellpadding='2' cellspacing='2'>", *table_rows, "</table>"]

        if detection_stats and self.detector:
            try:
                detection_results = self.detector.get_detection_results()
            except Exception:
                detection_results = []

            if detection_results:
                html_parts.append("<p><b>检测结果:</b></p>")
                for i, result in enumerate(detection_results[:3]):
                    html_parts.append(
                        f"<p>&nbsp;&nbsp;{i+1}. {result.get('class_name', '未知')} "
                        f"{result.get('confidence', 0):.2f}</p>"
                    )
                if len(detection_results) > 3:
                    html_parts.append(f"<p>&nbsp;&nbsp;... 还有 {len(detection_results)-3} 个</p>")
            else:
                html_parts.append("<p><b>当前无目标</b></p>")

        self.stats_label.setText("".join(html_parts))

    def update_config(self):
        pass
