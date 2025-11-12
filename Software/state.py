from PyQt6.QtCore import QObject, pyqtSignal
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class State(QObject):
    """系统状态类 - 支持信号"""

    # 定义信号
    connection_changed = pyqtSignal(bool)  # 连接状态变化信号
    parameters_changed = pyqtSignal(dict)  # 参数变化信号（通知UI刷新显示）

    def __init__(self):
        super().__init__()
        self._communication_thread = False
        self.data_processing_thread = False
        self.detection_thread = False
        self.data_queue_status = "idle"
        self.packet_size = 128
        # 系统配置参数（从文件加载）
        self._parameters = self._load_parameters()
        # 兼容旧代码的属性
        self.device_ip = "127.0.0.1"
        self.device_port = 5000

    def _load_parameters(self) -> dict:
        """从JSON文件加载参数"""
        config_path = Path(__file__).parent / "UI" / "config" / "parameters.json"

        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    params = json.load(f)
                    logger.info(f"✓ 参数已从 {config_path} 加载")
                    return params
            else:
                logger.warning(f"参数文件不存在: {config_path}，使用默认参数")
                return self._get_default_parameters()
        except Exception as e:
            logger.error(f"加载参数失败: {e}，使用默认参数")
            return self._get_default_parameters()

    def _get_default_parameters(self) -> dict:
        """返回默认参数"""
        return {
            "FFT": {
                "Length": 512,
                "Decimation_factor": 100,
                "Centre_frequency(MHz)": 2400.0,
                "bandwidth(MHz)": 100.0,
            },
            "UI": {"spectum_left_freq(MHz)": 2350.0, "spectum_right_freq(MHz)": 2450.0},
        }

    def save_parameters(self):
        """保存参数到JSON文件"""
        config_path = Path(__file__).parent / "UI" / "config" / "parameters.json"

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self._parameters, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ 参数已保存到 {config_path}")
            return True
        except Exception as e:
            logger.error(f"保存参数失败: {e}")
            return False

    # ==================== 参数访问接口 ====================

    @property
    def parameters(self):
        """获取所有参数（只读）"""
        return self._parameters

    def get_parameter(self, group: str, name: str, default=None):
        """获取单个参数值

        Args:
            group: 参数组名（如 "FFT", "UI"）
            name: 参数名（如 "Length", "spectum_left_freq(MHz)"）
            default: 默认值

        Returns:
            参数值
        """
        return self._parameters.get(group, {}).get(name, default)

    def set_parameter(self, group: str, name: str, value):
        """设置单个参数值

        Args:
            group: 参数组名
            name: 参数名
            value: 新值
        """
        if group not in self._parameters:
            self._parameters[group] = {}

        old_value = self._parameters[group].get(name)
        self._parameters[group][name] = value

        logger.info(f"参数更新: {group}.{name} = {value} (旧值: {old_value})")

        # 自动保存
        self.save_parameters()

        # 发射信号（但UI不需要响应，只是记录日志）
        self.parameters_changed.emit(
            {"group": group, "name": name, "value": value, "old_value": old_value}
        )

    # ==================== 常用参数的便捷访问 ====================

    @property
    def fft_length(self):  # 访问方法示例：直接 state.fft_length
        """FFT长度"""
        return self.get_parameter("FFT", "Length", 512)

    @property
    def decimation_factor(self):
        """抽取因子"""
        return self.get_parameter("FFT", "Decimation_factor", 100)

    @property
    def center_frequency(self):
        """中心频率 (MHz)"""
        return self.get_parameter("FFT", "Centre_frequency(MHz)", 2400.0)

    @property
    def bandwidth(self):
        """带宽 (MHz)"""
        return self.get_parameter("FFT", "bandwidth(MHz)", 100.0)

    @property
    def spectrum_left_freq(self):
        """频谱左边界 (MHz)"""
        return self.get_parameter("UI", "spectum_left_freq(MHz)", 2350.0)

    @property
    def spectrum_right_freq(self):
        """频谱右边界 (MHz)"""
        return self.get_parameter("UI", "spectum_right_freq(MHz)", 2450.0)

    @property
    def sample_rate(self):
        """实际采样率 (Hz)"""
        return 5e9 / self.decimation_factor  # ADC 5GHz / 抽取因子

    @property
    def conf_threshold(self):
        """YOLO置信度阈值"""
        return self.get_parameter("Detection", "conf_threshold", 0.25)

    @property
    def iou_threshold(self):
        """YOLO IOU阈值"""
        return self.get_parameter("Detection", "iou_threshold", 0.45)

    # ==================== 连接状态管理 ====================

    @property
    def communication_thread(self):
        return self._communication_thread

    @communication_thread.setter
    def communication_thread(self, value: bool):
        if self._communication_thread != value:
            self._communication_thread = value
            self.connection_changed.emit(value)
            logger.info(f"连接状态变化: {value}")
