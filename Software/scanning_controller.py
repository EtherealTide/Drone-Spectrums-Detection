"""scanning_controller.py — Frequency scanning and target-tracking controller.

Runs inside the detector sub-process alongside DroneDetector.
Reads the current window image directly from a shared-memory numpy array
that is owned and attached by the parent DroneDetector.

No dependency on State or DataProcessor.
"""

import logging
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class ScanMode(Enum):
    SCANNING = "scanning"
    LOCKED = "locked"
    DISABLED = "disabled"


class ScanningController:
    """
    Frequency scanning and target tracking controller.

    Parameters come from an init_params dict.
    Waterfall image slices are read from a numpy view into shared memory.
    """

    def __init__(self, init_params: dict, wf_arr, waterfall_lock):
        """
        Args:
            init_params:    Configuration dict from main process.
            wf_arr:         numpy view into shared waterfall memory,
                            shape (MAX_TOTAL_FFT, MAX_WATERFALL_H, 3), dtype uint8.
                            axis-0 = FFT/frequency, axis-1 = time.
            waterfall_lock: mp.Lock protecting waterfall shared-memory writes.
        """
        self.wf_arr = wf_arr
        self.waterfall_lock = waterfall_lock

        # ── Parameters ────────────────────────────────────────────────────────
        self.fft_length = init_params.get("fft_length", 512)
        self.channel_count = init_params.get("channel_count", 20)
        self.total_fft_length = init_params.get(
            "total_fft_length", self.fft_length * self.channel_count
        )
        self.total_bandwidth_mhz = init_params.get("total_bandwidth_mhz", 2000.0)
        self.waterfall_height = init_params.get("waterfall_height", 512)
        self.enable_scanning = init_params.get("enable_scanning", False)
        self.scan_bandwidth_mhz = init_params.get("scan_bandwidth_mhz", 100)
        self.overlap_ratio = init_params.get("overlap_ratio", 0.5)
        self.control_lost_threshold = init_params.get("control_lost_threshold", 50)
        self.start_frequency_mhz = init_params.get("start_frequency_mhz", 0.0)

        # ── Tracking state ────────────────────────────────────────────────────
        self.start_pt = 0
        self.end_pt = self.fft_length
        self.control_signal_class_id = None
        self.tracking_smoothing = 0.3

        self._calculate_window_parameters()

        self.scan_mode = (
            ScanMode.SCANNING if self.enable_scanning else ScanMode.DISABLED
        )
        self.current_window_index = 0
        self.locked_center_point = 0
        self.no_control_frame_count = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _point_to_frequency(self, point: int) -> float:
        denom = max(1, self.total_fft_length)
        return point * self.total_bandwidth_mhz / denom

    def _calculate_window_parameters(self):
        self.window_size = max(1, int(self.scan_bandwidth_mhz / 100 * self.fft_length))
        self.window_step = max(1, int(self.window_size * (1 - self.overlap_ratio)))
        total_pts = max(1, self.total_fft_length)
        self.total_windows = max(
            1, (total_pts - self.window_size) // self.window_step + 1
        )
        logger.info(
            f"Window params: size={self.window_size}, step={self.window_step}, "
            f"total_windows={self.total_windows}"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def set_control_signal_class_id(self, class_id: int):
        self.control_signal_class_id = class_id
        logger.info(f"Control signal class ID: {class_id}")

    def get_current_window_image(self) -> np.ndarray:
        """Return the image slice for the current scanning window.

        Returns:
            numpy.ndarray shape (window_width, waterfall_height, 3), BGR uint8.
            axis-0 = FFT/frequency, axis-1 = time.
        """
        wf_h = self.waterfall_height

        if not self.enable_scanning:
            self.start_pt = int(
                self.start_frequency_mhz
                * self.total_fft_length
                / max(1.0, self.total_bandwidth_mhz)
            )
            self.end_pt = self.start_pt + self.window_size
        elif self.scan_mode == ScanMode.SCANNING:
            self.start_pt, self.end_pt = self._get_window_range_scanning(
                self.current_window_index
            )
        elif self.scan_mode == ScanMode.LOCKED:
            self.start_pt, self.end_pt = self._get_window_range_locked(
                self.locked_center_point
            )

        self.start_pt = max(0, min(self.start_pt, self.total_fft_length - 1))
        self.end_pt = max(self.start_pt + 1, min(self.end_pt, self.total_fft_length))

        with self.waterfall_lock:
            return self.wf_arr[self.start_pt : self.end_pt, :wf_h, :].copy()

    def _get_window_range_scanning(self, idx: int):
        start = idx * self.window_step
        end = min(start + self.window_size, self.total_fft_length)
        return start, end

    def _get_window_range_locked(self, center: int):
        half = self.window_size // 2
        start = center - half
        end = center + half
        if start < 0:
            start, end = 0, self.window_size
        elif end > self.total_fft_length:
            end = self.total_fft_length
            start = max(0, end - self.window_size)
        return start, end

    def update_state_machine(self, detections: list, image_freq_size: int):
        """Update scanning state machine based on detection results.

        Args:
            detections:      YOLO detection result list.
            image_freq_size: input_image.shape[0] — frequency axis dimension (rows).
        """
        if not self.enable_scanning:
            self.scan_mode = ScanMode.DISABLED
            return
        if self.scan_mode == ScanMode.DISABLED:
            self.scan_mode = ScanMode.SCANNING
            return
        if self.control_signal_class_id is None:
            return

        ctrl_signals = [
            d for d in detections if d["class_id"] == self.control_signal_class_id
        ]
        has_ctrl = len(ctrl_signals) > 0
        window_width = self.end_pt - self.start_pt

        if self.scan_mode == ScanMode.SCANNING:
            if has_ctrl:
                target = max(ctrl_signals, key=lambda d: d["confidence"])
                bbox = target["bbox"]
                # Frequency is along axis-0 (rows) → y-coordinates
                y_center = (bbox[1] + bbox[3]) / 2.0
                target_pt = self.start_pt + int(
                    y_center / max(1, image_freq_size) * window_width
                )
                self.scan_mode = ScanMode.LOCKED
                self.locked_center_point = target_pt
                self.no_control_frame_count = 0
            else:
                self.current_window_index = (
                    self.current_window_index + 1
                ) % self.total_windows

        elif self.scan_mode == ScanMode.LOCKED:
            if has_ctrl:
                target = max(ctrl_signals, key=lambda d: d["confidence"])
                bbox = target["bbox"]
                y_center = (bbox[1] + bbox[3]) / 2.0
                new_pt = self.start_pt + int(
                    y_center / max(1, image_freq_size) * window_width
                )
                self.locked_center_point = int(
                    self.locked_center_point * (1 - self.tracking_smoothing)
                    + new_pt * self.tracking_smoothing
                )
                self.no_control_frame_count = 0
            else:
                self.no_control_frame_count += 1
                if self.no_control_frame_count >= self.control_lost_threshold:
                    self.current_window_index = min(
                        self.locked_center_point // max(1, self.window_step),
                        self.total_windows - 1,
                    )
                    self.scan_mode = ScanMode.SCANNING
                    self.locked_center_point = 0

    def get_status(self) -> dict:
        start_freq = self._point_to_frequency(self.start_pt)
        end_freq = self._point_to_frequency(self.end_pt)
        status = {
            "enabled": self.enable_scanning,
            "mode": self.scan_mode.value,
            "scan_mode": "Disabled",
            "scan_bandwidth_mhz": self.scan_bandwidth_mhz,
            "overlap_ratio": self.overlap_ratio,
            "control_lost_threshold": self.control_lost_threshold,
            "frequency_range_mhz": (start_freq, end_freq),
            "frequency_range_str": f"{start_freq:.1f}-{end_freq:.1f} MHz",
        }
        if self.scan_mode == ScanMode.SCANNING:
            status.update(
                {
                    "scan_mode": "Scanning",
                    "window_index": self.current_window_index + 1,
                    "total_windows": self.total_windows,
                }
            )
        elif self.scan_mode == ScanMode.LOCKED:
            center_freq = self._point_to_frequency(self.locked_center_point)
            status.update(
                {
                    "scan_mode": "Locked",
                    "center_frequency_mhz": center_freq,
                    "center_frequency_str": f"{center_freq:.2f} MHz",
                    "no_signal_count": self.no_control_frame_count,
                    "lost_threshold": self.control_lost_threshold,
                }
            )
        return status

    def apply_params(self, new_params: dict):
        """Apply a dict of updated parameters and recalculate window geometry."""
        for key in (
            "enable_scanning",
            "scan_bandwidth_mhz",
            "overlap_ratio",
            "control_lost_threshold",
            "total_fft_length",
            "total_bandwidth_mhz",
            "fft_length",
            "channel_count",
            "waterfall_height",
            "start_frequency_mhz",
        ):
            if key in new_params:
                setattr(self, key, new_params[key])
        self._calculate_window_parameters()
