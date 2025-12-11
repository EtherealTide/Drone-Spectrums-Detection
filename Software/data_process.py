import threading
import time
import numpy as np
from collections import deque
import logging
import queue
import matplotlib.pyplot as plt
import cv2
import traceback

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, state):
        self._init_complete = False
        self.state = state
        self.fft_data_queue = None
        self.data_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.process_thread = None
        self.image_thread = None

        self.enable_averaging = False
        self.averaging_count = 10
        self.history_buffer = deque(maxlen=self.averaging_count)

        self.latest_spectrum = None

        # 扫描参数
        self.channel_count = 20  # 固定20通道
        self.fft_length = state.fft_length  # 单通道FFT点数(如512)
        self.total_fft_length = self.fft_length * self.channel_count  # 10240
        self.total_bandwidth_mhz = 2000  # 总带宽2000MHz
        self.waterfall_width = self.total_fft_length
        self.waterfall_height = max(1, int(state.waterfall_height))

        zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
        self.waterfall_buffer = deque(
            [zero_line.copy() for _ in range(self.waterfall_height)],
            maxlen=self.waterfall_height,
        )

        self.transfer_time_interval = 0.01
        self.waterfall_image = np.zeros(
            (self.waterfall_width, self.waterfall_height, 3), dtype=np.uint8
        )

        self.image_needs_update = False

        cmap = plt.get_cmap("jet")
        self.colormap = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        self.processed_frame_count = 0
        self.max_value = 0.0
        self.min_value = 0.0
        self.batch_size = 0

        self.use_opencv_colormap = True
        self._init_complete = True

    def start_processing(self):
        if not self.process_thread or not self.process_thread.is_alive():
            self.state.data_processing_thread = True
            self.process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self.process_thread.start()
            logger.info("Data processing thread started")

        if not self.image_thread or not self.image_thread.is_alive():
            self.image_thread = threading.Thread(
                target=self._image_conversion_loop, daemon=True
            )
            self.image_thread.start()
            logger.info("Image conversion thread started")

    def stop_processing(self):
        self.state.data_processing_thread = False

        if self.process_thread:
            self.process_thread.join(timeout=2)

        if self.image_thread:
            self.image_thread.join(timeout=2)

        logger.info("Data processing and image conversion threads stopped")

    def _process_loop(self):
        while self.state.data_processing_thread:
            try:
                batch_frames = []
                try:
                    first_frame = self.fft_data_queue.get(timeout=1)
                    batch_frames.append(first_frame)
                except queue.Empty:
                    continue

                while True:
                    try:
                        frame = self.fft_data_queue.get_nowait()
                        batch_frames.append(frame)
                    except queue.Empty:
                        break

                self.batch_size = len(batch_frames)

                processed_batch = []
                for fft_frame in batch_frames:
                    fft_data = fft_frame["data"]

                    # 确保长度为总FFT长度
                    if len(fft_data) != self.total_fft_length:
                        if len(fft_data) > self.total_fft_length:
                            fft_data = fft_data[: self.total_fft_length]
                        else:
                            padded = np.zeros(
                                self.total_fft_length, dtype=fft_data.dtype
                            )
                            padded[: len(fft_data)] = fft_data
                            fft_data = padded

                    # 转换为dB
                    fft_data = 20 * np.log10(np.abs(fft_data) + 1e-12)
                    processed_batch.append(fft_data)

                # 保存原始dB值到buffer
                with self.data_lock:
                    for spectrum_db in processed_batch:
                        self.waterfall_buffer.append(spectrum_db)

                    # latest_spectrum 也保存dB值
                    self.latest_spectrum = processed_batch[-1].copy()
                    self.processed_frame_count += len(batch_frames)
                    self.image_needs_update = True

                self.state.data_queue_status = "processing"

            except Exception as e:
                logger.error(f"Data processing error: {e}", exc_info=True)
                self.state.data_queue_status = "error"
                time.sleep(0.1)

    def _image_conversion_loop(self):
        while self.state.data_processing_thread:
            try:
                if not self.image_needs_update:
                    time.sleep(self.transfer_time_interval)
                    continue

                # 获取整个waterfall buffer的dB数据
                with self.data_lock:
                    waterfall_list = list(self.waterfall_buffer)
                    self.image_needs_update = False

                # ⭐ 在这里对整张图进行归一化
                waterfall_array = np.array(
                    waterfall_list, dtype=np.float32
                )  # [height, width]
                min_db = np.min(waterfall_array)
                max_db = np.max(waterfall_array)

                # 归一化到 [0, 1]
                waterfall_normalized = (waterfall_array - min_db) / (
                    max_db - min_db + 1e-12
                )

                # 更新统计信息
                with self.data_lock:
                    self.max_value = float(max_db)
                    self.min_value = float(min_db)

                # 转置并翻转
                waterfall_normalized = np.flipud(
                    waterfall_normalized
                ).T  # [width, height]

                # 应用colormap
                if self.use_opencv_colormap:
                    gray_image = (waterfall_normalized * 255.0).astype(np.uint8)
                    bgr_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                else:
                    color_indices = (waterfall_normalized * 255.0).astype(np.uint8)
                    rgb_image = self.colormap[color_indices]

                with self.image_lock:
                    self.waterfall_image = rgb_image

            except Exception as e:
                logger.error(f"Image conversion error: {e}", exc_info=True)
                time.sleep(0.1)

    # ==================== 新增扫描接口 ====================

    def get_window_image(self, start_point, end_point):
        """
        Get waterfall image slice for specified FFT point range

        Args:
            start_point: Start FFT point index (0-10239)
            end_point: End FFT point index (0-10239)

        Returns:
            Sliced RGB image [height, width, 3]
        """
        # Boundary check
        start_point = max(0, min(start_point, self.total_fft_length - 1))
        end_point = max(start_point + 1, min(end_point, self.total_fft_length))

        with self.image_lock:
            sliced_image = self.waterfall_image[start_point:end_point, :, :].copy()
        return sliced_image

    def get_point_to_frequency(self, point_index):
        """
        Convert FFT point index to frequency (MHz)

        Args:
            point_index: FFT point index (0-10239)

        Returns:
            Frequency in MHz
        """
        return point_index * self.total_bandwidth_mhz / self.total_fft_length

    def get_latest_spectrum(self):
        with self.data_lock:
            return (
                self.latest_spectrum.copy()
                if self.latest_spectrum is not None
                else None
            )

    def get_waterfall_buffer(self):
        with self.data_lock:
            return list(self.waterfall_buffer)

    def get_waterfall_image(self):
        with self.image_lock:
            return self.waterfall_image.copy()

    # 获取统计信息
    def get_stats(self):
        with self.data_lock:
            return {
                "frame_id": self.processed_frame_count,
                "max_value": self.max_value,  # 现在是dB值
                "min_value": self.min_value,  # 现在是dB值
                "batch_size": self.batch_size,
                "waterfall_height": self.waterfall_height,
                "waterfall_width": self.waterfall_width,
            }

    def set_fft_length(self, length):
        """Update single channel FFT length"""
        with self.data_lock:
            self.fft_length = length
            self.total_fft_length = self.fft_length * self.channel_count
            self.waterfall_width = self.total_fft_length

            zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
            self.waterfall_buffer = deque(
                [zero_line.copy() for _ in range(self.waterfall_height)],
                maxlen=self.waterfall_height,
            )

            logger.info(
                f"FFT length updated: single_channel={length}, total={self.total_fft_length}"
            )

        with self.image_lock:
            self.waterfall_image = np.zeros(
                (self.waterfall_width, self.waterfall_height, 3), dtype=np.uint8
            )

    def set_waterfall_parameters(self, height=None):
        height_changed = False
        if height is not None:
            try:
                new_height = max(1, int(height))
            except (TypeError, ValueError):
                new_height = None
            if new_height and new_height != self.waterfall_height:
                with self.data_lock:
                    self.waterfall_height = new_height
                    zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
                    self.waterfall_buffer = deque(
                        [zero_line.copy() for _ in range(self.waterfall_height)],
                        maxlen=self.waterfall_height,
                    )
                height_changed = True

        if height_changed:
            with self.image_lock:
                self.waterfall_image = np.zeros(
                    (self.waterfall_height, self.waterfall_width, 3), dtype=np.uint8
                )
            logger.info(f"Waterfall parameters updated: height={self.waterfall_height}")
