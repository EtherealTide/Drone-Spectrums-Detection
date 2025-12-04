import threading
import time
import numpy as np
from collections import deque
import logging
import queue
import matplotlib.pyplot as plt
import cv2

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, state):
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

        self.fft_length = state.fft_length
        self.waterfall_width = self.fft_length
        self.waterfall_height = max(1, int(state.waterfall_height))

        zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
        self.waterfall_buffer = deque(
            [zero_line.copy() for _ in range(self.waterfall_height)],
            maxlen=self.waterfall_height,
        )

        self.transfer_time_interval = 0.01
        self.waterfall_image = np.zeros(
            (self.waterfall_height, self.waterfall_width, 3), dtype=np.uint8
        )

        self.image_needs_update = False

        cmap = plt.get_cmap("jet")
        self.colormap = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        self.processed_frame_count = 0
        self.max_value = 0.0
        self.min_value = 0.0
        self.batch_size = 0

        # 【改进】预计算LUT表，使用OpenCV的applyColorMap
        self.use_opencv_colormap = True

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
                batch_frames = []  # 一轮批次处理的帧列表
                try:
                    first_frame = self.fft_data_queue.get(timeout=1)
                    batch_frames.append(first_frame)
                except queue.Empty:
                    continue

                while True:  # 尽可能多地获取队列中的数据，直到队列为空
                    try:
                        frame = self.fft_data_queue.get_nowait()
                        batch_frames.append(frame)
                    except queue.Empty:
                        break

                self.batch_size = len(
                    batch_frames
                )  # 只要长度为1，就说明数据处理速度可以跟上通信层速度

                processed_batch = []
                for fft_frame in batch_frames:
                    fft_data = fft_frame["data"]
                    if len(fft_data) != self.fft_length:
                        if len(fft_data) > self.fft_length:
                            fft_data = fft_data[: self.fft_length]
                        else:
                            padded = np.zeros(self.fft_length, dtype=fft_data.dtype)
                            padded[: len(fft_data)] = fft_data
                            fft_data = padded
                    processed_batch.append(fft_data)

                batch_array = np.array(processed_batch, dtype=np.float32)
                global_min = np.min(batch_array)
                global_max = np.max(batch_array)

                if global_max > global_min + 1e-10:
                    normalized_batch = (batch_array - global_min) / (
                        global_max - global_min
                    )
                else:
                    normalized_batch = np.zeros_like(batch_array)

                with self.data_lock:
                    for normalized_spectrum in normalized_batch:
                        self.waterfall_buffer.append(normalized_spectrum)

                    self.latest_spectrum = normalized_batch[-1].copy()
                    self.max_value = float(np.max(normalized_batch))
                    self.min_value = float(np.min(normalized_batch))
                    self.processed_frame_count += len(batch_frames)
                    self.image_needs_update = True

                self.state.data_queue_status = "processing"

            except Exception as e:
                logger.error(f"数据处理异常: {e}", exc_info=True)
                self.state.data_queue_status = "error"
                time.sleep(0.1)

    def _image_conversion_loop(self):
        while self.state.data_processing_thread:
            try:
                if not self.image_needs_update:
                    time.sleep(self.transfer_time_interval)
                    continue

                with self.data_lock:
                    waterfall_list = list(self.waterfall_buffer)
                    self.image_needs_update = False

                # 【改进1】直接转为uint8，减少中间转换
                waterfall_array = np.array(waterfall_list, dtype=np.float32)
                waterfall_array = np.flipud(waterfall_array).T

                # 【改进2】使用OpenCV的硬件加速颜色映射
                if self.use_opencv_colormap:
                    gray_image = (waterfall_array * 255.0).astype(np.uint8)
                    bgr_image = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                else:
                    # 原始方法（保留作为fallback）
                    color_indices = (waterfall_array * 255.0).astype(np.uint8)
                    rgb_image = self.colormap[color_indices]

                with self.image_lock:
                    self.waterfall_image = rgb_image

            except Exception as e:
                logger.error(f"图像转换异常: {e}", exc_info=True)
                time.sleep(0.1)

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

    def get_stats(self):
        with self.data_lock:
            return {
                "frame_id": self.processed_frame_count,
                "max_value": self.max_value,
                "min_value": self.min_value,
                "batch_size": self.batch_size,
                "waterfall_height": self.waterfall_height,
                "waterfall_width": self.waterfall_width,
            }

    def set_fft_length(self, length):
        with self.data_lock:
            self.fft_length = length
            self.waterfall_width = length
            if self.waterfall_height <= 0:
                self.waterfall_height = length

            zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
            self.waterfall_buffer = deque(
                [zero_line.copy() for _ in range(self.waterfall_height)],
                maxlen=self.waterfall_height,
            )

            logger.info(
                f"FFT长度已设置为: {length}, 瀑布图尺寸 {self.waterfall_width}x{self.waterfall_height}"
            )

        with self.image_lock:
            self.waterfall_image = np.zeros(
                (self.waterfall_height, self.waterfall_width, 3), dtype=np.uint8
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
            logger.info(f"瀑布图参数更新 height={self.waterfall_height}")
