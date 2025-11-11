import threading
import time
import numpy as np
from collections import deque
import logging


class DataProcessor:
    def __init__(self, state):
        self.state = state
        self.fft_data_queue = None
        self.data_lock = threading.Lock()
        self.process_thread = None

        # 数据处理参数
        self.enable_averaging = False
        self.averaging_count = 10
        self.history_buffer = deque(maxlen=self.averaging_count)

        # 最新频谱数据（完整分辨率）
        self.latest_spectrum = None

        # 瀑布图参数和数据
        self.waterfall_height = 128
        self.waterfall_width = 1024

        # 初始化时用灰色填充buffer（-50.0 dB表示灰色）
        gray_line = np.full(self.waterfall_width, -50.0, dtype=np.float32)
        self.waterfall_buffer = deque(
            [gray_line.copy() for _ in range(self.waterfall_height)],
            maxlen=self.waterfall_height,
        )

        # 统计信息
        self.processed_frame_count = 0
        self.max_value = -100.0
        self.min_value = 0.0

    def start_processing(self):
        """启动数据处理线程"""
        if not self.process_thread or not self.process_thread.is_alive():
            self.state.data_processing_thread = True
            self.process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self.process_thread.start()
            logging.info("数据处理线程已启动")

    def stop_processing(self):
        """停止数据处理线程"""
        self.state.data_processing_thread = False
        if self.process_thread:
            self.process_thread.join(timeout=2)
        logging.info("数据处理线程已停止")

    def _process_loop(self):
        """数据处理主循环"""
        while self.state.data_processing_thread:
            try:
                # 从队列获取FFT数据（超时1秒）
                fft_frame = self.fft_data_queue.get(timeout=1)

                # 提取数据
                fft_data = fft_frame["data"]
                timestamp = fft_frame["timestamp"]

                # 数据处理
                processed_spectrum = self._process_fft_data(fft_data)

                # 降采样频谱
                downsampled_spectrum = self._downsample_spectrum(processed_spectrum)

                # 使用线程锁更新数据
                with self.data_lock:
                    # 直接append到buffer（自动替换最老的数据）
                    self.waterfall_buffer.append(downsampled_spectrum)

                    # 更新最新频谱和统计信息
                    self.latest_spectrum = processed_spectrum
                    self.max_value = np.max(processed_spectrum)
                    self.min_value = np.min(processed_spectrum)
                    self.processed_frame_count += 1

                self.state.data_queue_status = "processing"

            except Exception as e:
                if "Empty" not in str(type(e).__name__):
                    logging.error(f"数据处理异常: {e}", exc_info=True)
                self.state.data_queue_status = "idle"
                time.sleep(0.1)

    def get_latest_spectrum(self):
        """获取最新的完整频谱数据（线程安全）"""
        with self.data_lock:
            return (
                self.latest_spectrum.copy()
                if self.latest_spectrum is not None
                else None
            )

    def get_waterfall_buffer(self):
        """获取瀑布图buffer的副本（线程安全）- 返回list"""
        with self.data_lock:
            return list(self.waterfall_buffer)  # 浅拷贝deque为list

    def get_stats(self):
        """获取统计信息（线程安全）"""
        with self.data_lock:
            return {
                "frame_id": self.processed_frame_count,
                "max_value": self.max_value,
                "min_value": self.min_value,
            }

    def _downsample_spectrum(self, spectrum_data):
        """将频谱降采样到固定宽度"""
        fft_length = len(spectrum_data)
        downsample_factor = max(1, fft_length // self.waterfall_width)
        downsampled = spectrum_data[::downsample_factor][: self.waterfall_width]
        return downsampled

    def _process_fft_data(self, fft_data):
        """处理FFT数据"""
        # 转换为dB scale
        magnitude = np.abs(fft_data)
        magnitude = np.where(magnitude > 0, magnitude, 1e-10)  # 避免log(0)
        db_data = 20 * np.log10(magnitude)

        # 可选：移动平均滤波
        if self.enable_averaging:
            self.history_buffer.append(db_data)
            db_data = np.mean(list(self.history_buffer), axis=0)

        return db_data

    def set_averaging(self, enable, count=10):
        """设置移动平均"""
        self.enable_averaging = enable
        self.averaging_count = count
        self.history_buffer = deque(maxlen=count)
        logging.info(f"移动平均: {'启用' if enable else '禁用'}, 窗口: {count}")

    def set_waterfall_params(self, height=None, width=None):
        """设置瀑布图参数"""
        with self.data_lock:
            if height is not None:
                self.waterfall_height = height
                # 重新初始化buffer
                gray_line = np.full(self.waterfall_width, -50.0, dtype=np.float32)
                self.waterfall_buffer = deque(
                    [gray_line.copy() for _ in range(height)], maxlen=height
                )
                logging.info(f"瀑布图高度已设置为: {height}")

            if width is not None:
                self.waterfall_width = width
                # 重新初始化buffer
                gray_line = np.full(width, -50.0, dtype=np.float32)
                self.waterfall_buffer = deque(
                    [gray_line.copy() for _ in range(self.waterfall_height)],
                    maxlen=self.waterfall_height,
                )
                logging.info(f"瀑布图宽度已设置为: {width}")
