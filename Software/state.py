from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class State(QObject):
    """System state holder with signal support."""

    connection_changed = pyqtSignal(bool)
    parameters_changed = pyqtSignal(dict)

    # ── Runtime statistics signals (emitted by main-process QTimer from stats queues) ──
    receiver_stats_updated = pyqtSignal(dict)  # Receiver stats
    detection_updated = pyqtSignal(dict)  # Detector stats

    def __init__(self):
        super().__init__()
        self._communication_thread = False
        self.data_processing_thread = False
        self.detection_thread = False
        self.data_queue_status = "idle"
        self._parameters = self._load_parameters()
        self.device_ip = "127.0.0.1"
        # self.device_ip = "192.168.1.100"
        self.device_port = 5000

        # ── Runtime statistics (populated by main-process QTimer) ──────────────
        self.receiver_stats: dict = {
            "fps": 0.0,
            "sent_frames": 0,
            "received_frames": 0,
        }
        self.detection_stats: dict = {
            "fps": 0.0,
            "total_detections": 0,
            "total_objects": 0,
            "detection_count": 0,
        }


    def _load_parameters(self) -> dict:
        config_path = Path(__file__).parent / "parameters.json"
        try:
            if config_path.exists():
                params = json.loads(config_path.read_text(encoding="utf-8-sig"))
                logger.info(f"parameters loaded from {config_path}")
                return params
            logger.warning(f"parameter file missing {config_path}, using defaults")
            return self._get_default_parameters()
        except Exception as e:
            logger.error(f"load parameters failed: {e}, using defaults")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> dict:
        return {
            "Slave Computer": {
                "FFT_Length": 10240,
                "Centre_frequency(MHz)": 2400.0,
                "SPAN(MHz)": 100.0,
            },
            "Detection": {
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "image_size": 512,
            },
            "UI_Spectrum": {
                "spectrum_left_freq(MHz)": 0.0,
                "spectrum_right_freq(MHz)": 200.0,
            },
            "UI_Waterfall": {
                "spectrum_left_freq(MHz)": 0.0,
                "spectrum_right_freq(MHz)": 200.0,
            },
            "Data_Process": {
                "waterfall_height": 512,
                "enable_noise_filter": False,
                "noise_filter_mode": "subtraction",
                "noise_alpha": 0.05,
            },
        }

    def save_parameters(self):
        config_path = Path(__file__).parent / "parameters.json"
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                json.dumps(self._parameters, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(f"parameters saved to {config_path}")
            return True
        except Exception as e:
            logger.error(f"save parameters failed: {e}")
            return False

    @property
    def parameters(self):
        return self._parameters

    def get_parameter(self, group: str, name: str, default=None):
        return self._parameters.get(group, {}).get(name, default)

    def set_parameter(self, group: str, name: str, value):
        if group not in self._parameters:
            self._parameters[group] = {}

        old_value = self._parameters[group].get(name)
        self._parameters[group][name] = value

        logger.info(f"parameter updated: {group}.{name} = {value} (old {old_value})")

        self.save_parameters()

        self.parameters_changed.emit(
            {"group": group, "name": name, "value": value, "old_value": old_value}
        )

    # ==================== common parameters ====================
    # 使用@property装饰器为常用参数提供便捷访问

    @property
    def center_frequency(self):
        return self.get_parameter("Slave Computer", "Centre_frequency(MHz)", 2400.0)

    @property
    def span(self):
        return self.get_parameter("Slave Computer", "SPAN(MHz)", 100.0)

    @property
    def spectrum_left_freq(self):
        return self.get_parameter("UI_Spectrum", "spectrum_left_freq(MHz)", 0.0)

    @property
    def spectrum_right_freq(self):
        return self.get_parameter("UI_Spectrum", "spectrum_right_freq(MHz)", 200.0)

    @property
    def sample_rate(self):
        # return 5e9 / self.decimation_factor
        return 2e9

    @property
    def conf_threshold(self):
        return self.get_parameter("Detection", "conf_threshold", 0.25)

    @property
    def iou_threshold(self):
        return self.get_parameter("Detection", "iou_threshold", 0.45)

    @property
    def image_size(self):
        return self.get_parameter("Detection", "image_size", 512)
    @property
    def waterfall_height(self):
        return self.get_parameter("Data_Process", "waterfall_height", 512)

    @property
    def waterfall_left_freq(self):
        return self.get_parameter("UI_Waterfall", "spectrum_left_freq(MHz)", 0.0)

    @property
    def waterfall_right_freq(self):
        return self.get_parameter("UI_Waterfall", "spectrum_right_freq(MHz)", 200.0)

    @property
    def enable_noise_filter(self):
        return self.get_parameter("Data_Process", "enable_noise_filter", False)

    @property
    def noise_filter_mode(self):
        return self.get_parameter("Data_Process", "noise_filter_mode", "subtraction")

    @property
    def noise_alpha(self):
        return self.get_parameter("Data_Process", "noise_alpha", 0.05)

    @property
    def total_fft_length(self) -> int:
        return self.get_parameter("Slave Computer", "FFT_Length", 10240)

    @property
    def total_bandwidth_mhz(self) -> float:
        return self.get_parameter("Slave Computer", "Total_Bandwidth_MHz", 2000.0)

    # ==================== connection status ====================

    @property
    def communication_thread(self):
        return self._communication_thread

    @communication_thread.setter
    def communication_thread(self, value: bool):
        if self._communication_thread != value:
            self._communication_thread = value
            self.connection_changed.emit(value)
            logger.info(f"connection status changed: {value}")
