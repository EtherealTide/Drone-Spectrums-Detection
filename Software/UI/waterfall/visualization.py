from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import logging

from ..utils.component import Component, BodyLabel
from ..settings.theme_manager import get_theme_manager
from ipc import (
    SHM_DETECTION_SHAPE,
    SHM_DETECTION_DTYPE,
)

logger = logging.getLogger(__name__)


class WaterfallVisualizationCard(QWidget):
    """Shows selected detection slice image from shared memory.

    Reads image data directly from shared-memory numpy arrays.
    Stats are read from state.processor_stats / detection_stats / scan_status.
    """

    def __init__(self, parent=None, shm_detection=None, state=None):
        super().__init__(parent)
        self.setObjectName("WaterfallVisualizationCard")
        self.component = Component()
        self.theme_manager = get_theme_manager()
        self.state = state
        self.frame_displayed = 0

        # Attach to shared-memory numpy views
        self._wf_arr = None
        self._det_arr = None

        if shm_detection is not None:
            # np.frombuffer returns a 1D array without copying data, and we need to reshape it to the expected dimensions
            self._det_arr = np.frombuffer(
                shm_detection.buf, dtype=SHM_DETECTION_DTYPE
            ).reshape(SHM_DETECTION_SHAPE)

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
        self.update_timer.start(25)  # 40 FPS

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
            display_image = None

            # Prefer detection image (written by detector process, 640×640×3)
            if self._det_arr is not None:
                display_image = self._det_arr.copy()

        
            if display_image is not None and display_image.size > 0:
                self._update_label(self.result_label, display_image)
                self.frame_displayed += 1
            else:
                self.result_label.setText("等待检测/瀑布图数据...")

            processor_stats = self.state.processor_stats if self.state else {}
            detection_stats = self.state.detection_stats if self.state else {}
            self._update_stats(processor_stats, detection_stats)

        except Exception as e:
            logger.error(f"Waterfall display error: {e}", exc_info=True)
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
        processed_fps = stats.get("fps", 0.0)
        detection_frames = detection_stats.get("detection_count", 0)
        yolo_fps = detection_stats.get("yolo_fps", 0.0)
        yolo_infer_time_ms = detection_stats.get("yolo_infer_time_ms", 0.0)
        inference_device = detection_stats.get("inference device", "N/A")
        compute_capability = detection_stats.get("compute capability", "N/A")
        selected_window = int(detection_stats.get("selected_window", 0))
        window_count = int(detection_stats.get("window_count", 0))

        stats_table = f"""
        <table cellpadding='4' cellspacing='0' width='100%'>
            <tr>
                <td width='33%'>发送帧数: {sent_frames}</td>
                <td width='33%'>接收帧数: {received_frames}</td>
                <td width='34%'>数据处理FPS: {processed_fps:.2f}</td>
            </tr>
            <tr>
                <td>检测帧数: {detection_frames}</td>
                <td>推理设备: {inference_device}</td>
                <td>计算能力: {compute_capability}(CUDA)</td>
                
            </tr>
            <tr>
                <td>YOLO FPS: {yolo_fps:.2f}</td>
                <td>YOLO推理耗时: {yolo_infer_time_ms:.2f} ms</td>
                <td>显示帧数: {self.frame_displayed}</td>
            </tr>
            <tr>
                <td>窗口: {selected_window + 1}/{max(1, window_count)}</td>
            </tr>
        </table>
        """
        self.stats_label.setText(stats_table)

    def update_config(self):
        pass
