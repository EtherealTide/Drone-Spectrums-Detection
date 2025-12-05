"""
Scanning and Locking Controller
Manages frequency scanning and target tracking logic
"""

import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ScanMode(Enum):
    """Scanning state machine modes"""

    SCANNING = "scanning"
    LOCKED = "locked"
    DISABLED = "disabled"


class ScanningController:
    """
    Frequency scanning and target tracking controller

    Responsibilities:
    - Fixed-step scanning in SCANNING mode
    - Dynamic frequency tracking in LOCKED mode
    - State machine transitions
    - Window range calculation
    - Status information provision for UI
    """

    def __init__(self, state, data_processor):
        """
        Initialize scanning controller

        Args:
            state: System state holder
            data_processor: Data processor instance
        """
        self.state = state
        self.data_processor = data_processor

        # Scanning parameters
        self.enable_scanning = state.get_parameter("Scanning", "enable_scanning", False)
        self.scan_bandwidth_mhz = state.get_parameter(
            "Scanning", "scan_bandwidth_mhz", 100
        )
        self.overlap_ratio = state.get_parameter("Scanning", "overlap_ratio", 0.5)
        self.control_lost_threshold = state.get_parameter(
            "Scanning", "control_lost_threshold", 50
        )

        # Control signal class identification
        self.control_signal_class_id = None  # 用于找到控制信号的框的中心位置

        # Calculate window parameters
        self._calculate_window_parameters()

        # State machine
        self.scan_mode = (
            ScanMode.SCANNING if self.enable_scanning else ScanMode.DISABLED
        )
        self.current_window_index = 0  # Scanning mode: current window index
        self.locked_center_point = 0  # Locked mode: tracking center FFT point
        self.no_control_frame_count = 0  # Counter for frames without control signal

        # Tracking smoothing (avoid jitter)
        self.tracking_smoothing = 0.3  # Smoothing factor (0-1), lower = smoother

    def _calculate_window_parameters(self):
        """Calculate sliding window parameters based on scan bandwidth"""
        # Single window FFT points，比如200M，就是两个通道的FFT点数
        self.window_size = int(
            self.state.scan_bandwidth_mhz / 100 * self.state.fft_length
        )

        # Window step size (considering overlap)
        self.window_step = int(self.window_size * (1 - self.overlap_ratio))

        # Total number of windows
        total_points = self.data_processor.total_fft_length
        self.total_windows = (total_points - self.window_size) // self.window_step + 1

    def set_control_signal_class_id(self, class_id):
        """Set control signal class ID from detector"""
        self.control_signal_class_id = class_id
        logger.info(f"Control signal class ID set to: {class_id}")

    def get_current_window_image(self):
        """
        Get current window image based on state machine

        Returns:
            tuple: (image, window_start_point, window_end_point)
        """
        if not self.enable_scanning:
            # Non-scanning mode: return full image
            image = self.data_processor.get_waterfall_image()
            return image, 0, self.data_processor.total_fft_length

        if self.scan_mode == ScanMode.SCANNING:
            # Scanning mode: fixed-step scanning
            start_pt, end_pt = self._get_window_range_scanning(
                self.current_window_index
            )
        else:  # LOCKED
            # Locked mode: dynamic tracking
            start_pt, end_pt = self._get_window_range_locked(self.locked_center_point)

        # Slice image
        image = self.data_processor.get_window_image(start_pt, end_pt)
        return image, start_pt, end_pt

    def _get_window_range_scanning(self, window_index):
        """
        Get FFT point range for scanning mode
        按照固定步长滑动窗口扫描
        Args:
            window_index: Window index

        Returns:
            tuple: (start_point, end_point)
        """
        start_point = window_index * self.window_step
        end_point = start_point + self.window_size
        end_point = min(end_point, self.data_processor.total_fft_length)
        return start_point, end_point

    def _get_window_range_locked(self, center_point):
        """
        Get FFT point range for locked mode (centered on target)
        以锁定点为中心，计算窗口范围
        Args:
            center_point: Target center FFT point

        Returns:
            tuple: (start_point, end_point)
        """
        half_window = self.window_size // 2
        start_point = center_point - half_window
        end_point = center_point + half_window

        # Boundary handling
        if start_point < 0:
            start_point = 0
            end_point = self.window_size
        elif end_point > self.data_processor.total_fft_length:
            end_point = self.data_processor.total_fft_length
            start_point = end_point - self.window_size

        return start_point, end_point

    def update_state_machine(
        self, detections, window_start_point, window_end_point, image_width
    ):
        """
        Update state machine based on detection results

        Args:
            detections: List of detection results
            window_start_point: Current window start FFT point
            window_end_point: Current window end FFT point
            image_width: Width of detection image (pixels)
        """
        if not self.enable_scanning:
            self.scan_mode = ScanMode.DISABLED
            return

        if self.control_signal_class_id is None:
            logger.warning(
                "Control signal class ID not set, skipping state machine update"
            )
            return

        # Filter control signals
        control_signals = [
            det for det in detections if det["class_id"] == self.control_signal_class_id
        ]

        has_control_signal = len(control_signals) > 0

        if self.scan_mode == ScanMode.SCANNING:
            if has_control_signal:
                # 转入 LOCKED 模式
                # Select control signal with highest confidence
                target_signal = max(control_signals, key=lambda x: x["confidence"])

                # Calculate target position in full FFT
                bbox = target_signal["bbox"]  # [x1, y1, x2, y2]
                target_center_x_in_window = (bbox[1] + bbox[3]) / 2

                # Convert to full FFT point index
                window_width = window_end_point - window_start_point
                target_center_point = window_start_point + int(
                    target_center_x_in_window / image_width * window_width
                )

                # Enter LOCKED mode
                self.scan_mode = ScanMode.LOCKED
                self.locked_center_point = target_center_point
                self.no_control_frame_count = 0

            else:
                # Continue scanning next window
                self.current_window_index = (
                    self.current_window_index + 1
                ) % self.total_windows

        elif self.scan_mode == ScanMode.LOCKED:
            if has_control_signal:
                # ⭐ Dynamic tracking: update center point
                target_signal = max(control_signals, key=lambda x: x["confidence"])
                bbox = target_signal["bbox"]
                target_center_x_in_window = (bbox[1] + bbox[3]) / 2

                # Calculate new center FFT point
                window_width = window_end_point - window_start_point
                new_center_point = window_start_point + int(
                    target_center_x_in_window / image_width * window_width
                )

                # 平滑更新锁定点位置，将新位置和旧位置加权平均
                self.locked_center_point = int(
                    self.locked_center_point * (1 - self.tracking_smoothing)
                    + new_center_point * self.tracking_smoothing
                )

                self.no_control_frame_count = 0
            else:
                # No signal detected, increment counter
                self.no_control_frame_count += 1

                if self.no_control_frame_count >= self.control_lost_threshold:
                    # ⭐ Lost lock → Return to SCANNING mode
                    # Continue scanning from current position
                    self.current_window_index = (
                        self.locked_center_point // self.window_step
                    )
                    self.current_window_index = min(
                        self.current_window_index, self.total_windows - 1
                    )

                    self.scan_mode = ScanMode.SCANNING
                    self.locked_center_point = 0

    def get_status(self):
        """
        Get current scanning status information for UI display

        Returns:
            dict: Comprehensive status information
        """
        status = {
            "enabled": self.enable_scanning,
            "mode": self.scan_mode.value,
            "scan_bandwidth_mhz": self.scan_bandwidth_mhz,
            "overlap_ratio": self.overlap_ratio,
            "control_lost_threshold": self.control_lost_threshold,
        }

        if not self.enable_scanning:
            status["frequency_range_mhz"] = (
                0.0,
                self.data_processor.total_bandwidth_mhz,
            )
            status["frequency_range_str"] = (
                f"0-{self.data_processor.total_bandwidth_mhz} MHz"
            )
            return status

        if self.scan_mode == ScanMode.SCANNING:
            start_pt, end_pt = self._get_window_range_scanning(
                self.current_window_index
            )
            start_freq = self.data_processor.get_point_to_frequency(start_pt)
            end_freq = self.data_processor.get_point_to_frequency(end_pt)

            status.update(
                {
                    "window_index": self.current_window_index + 1,
                    "total_windows": self.total_windows,
                    "frequency_range_mhz": (start_freq, end_freq),
                    "frequency_range_str": f"{start_freq:.1f}-{end_freq:.1f} MHz",
                    "scan_mode": "Scanning",
                }
            )

        elif self.scan_mode == ScanMode.LOCKED:
            start_pt, end_pt = self._get_window_range_locked(self.locked_center_point)
            start_freq = self.data_processor.get_point_to_frequency(start_pt)
            end_freq = self.data_processor.get_point_to_frequency(end_pt)
            center_freq = self.data_processor.get_point_to_frequency(
                self.locked_center_point
            )

            status.update(
                {
                    "center_frequency_mhz": center_freq,
                    "center_frequency_str": f"{center_freq:.2f} MHz",
                    "frequency_range_mhz": (start_freq, end_freq),
                    "frequency_range_str": f"{start_freq:.1f}-{end_freq:.1f} MHz",
                    "no_signal_count": self.no_control_frame_count,
                    "lost_threshold": self.control_lost_threshold,
                    "scan_mode": "Locked",
                }
            )

        return status

    def get_current_frequency_range(self):
        """
        Get current frequency range

        Returns:
            tuple: (start_freq_mhz, end_freq_mhz)
        """
        if not self.enable_scanning:
            return (0.0, self.data_processor.total_bandwidth_mhz)

        if self.scan_mode == ScanMode.SCANNING:
            start_pt, end_pt = self._get_window_range_scanning(
                self.current_window_index
            )
        else:  # LOCKED
            start_pt, end_pt = self._get_window_range_locked(self.locked_center_point)

        start_freq = self.data_processor.get_point_to_frequency(start_pt)
        end_freq = self.data_processor.get_point_to_frequency(end_pt)
        return (start_freq, end_freq)

    def update_parameters(self):
        """Update scanning parameters from state"""
        new_enable = self.state.get_parameter("Scanning", "enable_scanning", False)
        new_bandwidth = self.state.get_parameter("Scanning", "scan_bandwidth_mhz", 100)
        new_threshold = self.state.get_parameter(
            "Scanning", "control_lost_threshold", 5
        )

        params_changed = False

        if new_bandwidth != self.scan_bandwidth_mhz:
            self.scan_bandwidth_mhz = new_bandwidth
            self._calculate_window_parameters()
            params_changed = True
            logger.info(f"Scan bandwidth updated to {self.scan_bandwidth_mhz} MHz")

        if new_enable != self.enable_scanning:
            self.enable_scanning = new_enable
            self.scan_mode = ScanMode.SCANNING if new_enable else ScanMode.DISABLED
            logger.info(f"Scanning mode {'enabled' if new_enable else 'disabled'}")

        if new_threshold != self.control_lost_threshold:
            self.control_lost_threshold = new_threshold
            logger.info(f"Control lost threshold updated to {new_threshold}")

        return params_changed
