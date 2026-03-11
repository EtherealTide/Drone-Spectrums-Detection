from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
import logging

from ..utils.component import Component, BodyLabel
from ..settings.theme_manager import get_theme_manager
from ipc import (
    SHM_WATERFALL_SHAPE,
    SHM_WATERFALL_DTYPE,
    SHM_DETECTION_SHAPE,
    SHM_DETECTION_DTYPE,
)

logger = logging.getLogger(__name__)


class WaterfallVisualizationCard(QWidget):
    """Shows detector result image (or raw waterfall as fallback).

    Reads image data directly from shared-memory numpy arrays.
    Stats are read from state.processor_stats / detection_stats / scan_status.
    """

    def __init__(self, parent=None, shm_waterfall=None, shm_detection=None, state=None):
        super().__init__(parent)
        self.setObjectName("WaterfallVisualizationCard")
        self.component = Component()
        self.theme_manager = get_theme_manager()
        self.state = state
        self.frame_displayed = 0

        # Attach to shared-memory numpy views
        self._wf_arr = None
        self._det_arr = None
        if shm_waterfall is not None:
            self._wf_arr = np.frombuffer(
                shm_waterfall.buf, dtype=SHM_WATERFALL_DTYPE
            ).reshape(SHM_WATERFALL_SHAPE)
        if shm_detection is not None:
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

            # Fallback to full waterfall slice
            if (
                display_image is None or display_image.size == 0
            ) and self._wf_arr is not None:
                total_fft = (
                    self.state.total_fft_length
                    if self.state
                    else SHM_WATERFALL_SHAPE[0]
                )
                wf_h = (
                    self.state.waterfall_height
                    if self.state
                    else SHM_WATERFALL_SHAPE[1]
                )
                total_fft = max(1, min(total_fft, SHM_WATERFALL_SHAPE[0]))
                wf_h = max(1, min(wf_h, SHM_WATERFALL_SHAPE[1]))
                display_image = self._wf_arr[:total_fft, :wf_h, :].copy()

            if display_image is not None and display_image.size > 0:
                # Transpose: axis-0=freq → rows; after transpose rows=time, cols=freq
                if display_image.ndim == 3:
                    display_image = np.transpose(display_image, (1, 0, 2))
                self._update_label(self.result_label, display_image)
                self.frame_displayed += 1
            else:
                self.result_label.setText("等待检测/瀑布图数据...")

            processor_stats = self.state.processor_stats if self.state else {}
            detection_stats = self.state.detection_stats if self.state else {}
            scan_status = self.state.scan_status if self.state else {}
            self._update_stats(processor_stats, detection_stats, scan_status)

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

    def _update_stats(self, stats: dict, detection_stats: dict, scan_status: dict):
        sent_frames = getattr(self.state, "sent_frames", 0) if self.state else 0
        received_frames = getattr(self.state, "received_frames", 0) if self.state else 0
        processed_fps = stats.get("fps", 0.0)
        detection_frames = detection_stats.get("detection_count", 0)
        detection_fps = detection_stats.get("fps", 0.0)
        freq_range_str = scan_status.get("frequency_range_str", "N/A")
        scan_mode = scan_status.get("scan_mode", "N/A")

        stats_table = f"""
        <table cellpadding='4' cellspacing='0' width='100%'>
            <tr>
                <td width='33%'>发送帧数: {sent_frames}</td>
                <td width='33%'>接收帧数: {received_frames}</td>
                <td width='34%'>数据处理FPS: {processed_fps:.2f}</td>
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
        self.stats_label.setText(stats_table)

    def update_config(self):
        pass
