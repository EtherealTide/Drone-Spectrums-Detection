import threading
import time
import numpy as np
from collections import deque
import logging
import queue
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, state):
        self.state = state
        self.fft_data_queue = None
        self.data_lock = threading.Lock()
        self.image_lock = threading.Lock()  # 保护图像数据
        self.process_thread = None
        self.image_thread = None

        # 数据处理参数
        self.enable_averaging = False
        self.averaging_count = 10
        self.history_buffer = deque(maxlen=self.averaging_count)

        # 最新频谱数据（完整分辨率，归一化后）
        self.latest_spectrum = None

        # 瀑布图参数（正方形：height = width = FFT点数）
        self.fft_length = state.fft_length
        self.waterfall_height = self.fft_length  # 正方形
        self.waterfall_width = self.fft_length

        # 瀑布图buffer（归一化数据 [0, 1]）
        zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
        self.waterfall_buffer = deque(
            [zero_line.copy() for _ in range(self.waterfall_height)],
            maxlen=self.waterfall_height,
        )

        # 瀑布图RGB图像（用于算法层，shape: (height, width, 3)）
        self.transfer_time = 0.01  # 10ms更新一次图像
        self.waterfall_image = np.zeros(
            (self.waterfall_height, self.waterfall_width, 3), dtype=np.uint8
        )

        # 图像更新标志
        self.image_needs_update = False

        # 预计算jet色图（256级）
        cmap = plt.get_cmap("jet")
        self.colormap = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        # 统计信息
        self.processed_frame_count = 0
        self.max_value = 0.0
        self.min_value = 0.0
        self.batch_size = 0  # 上次批量处理的帧数

    def start_processing(self):
        """启动数据处理线程和图像转换线程"""
        # 启动数据处理线程
        if not self.process_thread or not self.process_thread.is_alive():
            self.state.data_processing_thread = True
            self.process_thread = threading.Thread(
                target=self._process_loop, daemon=True
            )
            self.process_thread.start()
            logging.info("数据处理线程已启动")

        # 启动图像转换线程
        if not self.image_thread or not self.image_thread.is_alive():
            self.image_thread = threading.Thread(
                target=self._image_conversion_loop, daemon=True
            )
            self.image_thread.start()
            logging.info("图像转换线程已启动")

    def stop_processing(self):
        """停止数据处理和图像转换线程"""
        self.state.data_processing_thread = False

        if self.process_thread:
            self.process_thread.join(timeout=2)

        if self.image_thread:
            self.image_thread.join(timeout=2)

        logging.info("数据处理和图像转换线程已停止")

    def _process_loop(self):
        """数据处理主循环 - 批量获取并全局归一化"""
        while self.state.data_processing_thread:
            try:
                # 步骤1: 批量获取队列中的所有数据
                batch_frames = []

                # 至少获取一帧（阻塞等待）
                try:
                    first_frame = self.fft_data_queue.get(timeout=1)
                    batch_frames.append(first_frame)
                except queue.Empty:
                    continue  # 无数据，继续循环

                # 获取队列中的剩余所有数据（非阻塞）
                while True:
                    try:
                        frame = self.fft_data_queue.get_nowait()
                        batch_frames.append(frame)
                    except queue.Empty:
                        break  # 队列已空

                self.batch_size = len(batch_frames)

                # 步骤2: 处理批量数据（计算幅度，不转dB）
                processed_batch = []
                for fft_frame in batch_frames:
                    fft_data = fft_frame["data"]  # 已经是幅度形式
                    # 转dB
                    # fft_data = 20 * np.log10(fft_data + 1e-10)  # 加1e-10避免log(0)
                    # 确保数据长度匹配
                    if len(fft_data) != self.fft_length:
                        logging.warning(
                            f"FFT数据长度不匹配: 期望{self.fft_length}, 实际{len(fft_data)}"
                        )
                        # 裁剪或填充
                        if len(fft_data) > self.fft_length:
                            fft_data = fft_data[: self.fft_length]
                        else:
                            padded = np.zeros(self.fft_length, dtype=fft_data.dtype)
                            padded[: len(fft_data)] = fft_data
                            fft_data = padded

                    processed_batch.append(fft_data)

                # 步骤3: 转换为numpy数组并进行全局归一化
                # shape: (num_frames, fft_length)
                batch_array = np.array(processed_batch, dtype=np.float32)

                # 全局归一化到 [0, 1]
                global_min = np.min(batch_array)
                global_max = np.max(batch_array)

                if global_max > global_min + 1e-10:  # 避免除零
                    normalized_batch = (batch_array - global_min) / (
                        global_max - global_min
                    )
                else:
                    normalized_batch = np.zeros_like(batch_array)

                logging.debug(
                    f"归一化范围: [{global_min:.6e}, {global_max:.6e}] -> [0, 1]"
                )

                # 步骤4: 使用线程锁更新数据
                with self.data_lock:
                    # 将批量归一化数据逐行插入瀑布图buffer
                    for normalized_spectrum in normalized_batch:
                        self.waterfall_buffer.append(normalized_spectrum)

                    # 最新一帧作为当前频谱
                    self.latest_spectrum = normalized_batch[-1].copy()

                    # 更新统计信息
                    self.max_value = float(np.max(normalized_batch))
                    self.min_value = float(np.min(normalized_batch))
                    self.processed_frame_count += len(batch_frames)

                    # 设置图像更新标志
                    self.image_needs_update = True

                self.state.data_queue_status = "processing"

            except Exception as e:
                logging.error(f"数据处理异常: {e}", exc_info=True)
                self.state.data_queue_status = "error"
                time.sleep(0.1)

    def _image_conversion_loop(self):
        """图像转换线程 - 将瀑布图转换为RGB图像"""
        while self.state.data_processing_thread:
            try:
                # 检查是否需要更新图像
                if not self.image_needs_update:
                    time.sleep(self.transfer_time)  # 定期检查一次
                    continue

                # 获取瀑布图数据（浅拷贝list）
                with self.data_lock:
                    waterfall_list = list(self.waterfall_buffer)
                    self.image_needs_update = False

                # 转换为numpy数组并翻转（新数据在顶部）
                # shape: (height, width)
                waterfall_array = np.array(waterfall_list, dtype=np.float32)
                waterfall_array = np.flipud(waterfall_array)
                # 数据转置
                waterfall_array = waterfall_array.T  # shape: (width, height)

                # 转换为颜色索引 [0, 255]
                color_indices = (waterfall_array * 255.0).astype(np.uint8)

                # 应用jet色图
                rgb_image = self.colormap[color_indices]  # (height, width, 3)

                # 更新图像（使用图像锁）
                with self.image_lock:
                    self.waterfall_image = rgb_image

                logging.debug("瀑布图RGB图像已更新")

            except Exception as e:
                logging.error(f"图像转换异常: {e}", exc_info=True)
                time.sleep(0.1)

    # ==================== 对外接口 ====================

    def get_latest_spectrum(self):
        """获取最新频谱数据（归一化到[0,1]）- 线程安全"""
        with self.data_lock:
            return (
                self.latest_spectrum.copy()
                if self.latest_spectrum is not None
                else None
            )

    def get_waterfall_buffer(self):
        """获取瀑布图buffer（归一化数据[0,1]）- 线程安全

        Returns:
            list: 长度为height的列表，每个元素是长度为width的numpy数组
        """
        with self.data_lock:
            return list(self.waterfall_buffer)

    def get_waterfall_image(self):
        """获取瀑布图RGB图像（用于算法层）- 线程安全

        Returns:
            np.ndarray: shape=(height, width, 3), dtype=uint8, 范围[0, 255]
        """
        with self.image_lock:
            return self.waterfall_image.copy()

    def get_stats(self):
        """获取统计信息 - 线程安全"""
        with self.data_lock:
            return {
                "frame_id": self.processed_frame_count,
                "max_value": self.max_value,
                "min_value": self.min_value,
                "batch_size": self.batch_size,
                "waterfall_height": self.waterfall_height,
                "waterfall_width": self.waterfall_width,
            }

    # ==================== 配置接口 ====================

    def set_fft_length(self, length):
        """设置FFT长度（同时更新瀑布图尺寸为正方形）"""
        with self.data_lock:
            self.fft_length = length
            self.waterfall_height = length
            self.waterfall_width = length

            # 重新初始化buffer
            zero_line = np.zeros(self.waterfall_width, dtype=np.float32)
            self.waterfall_buffer = deque(
                [zero_line.copy() for _ in range(self.waterfall_height)],
                maxlen=self.waterfall_height,
            )

            logging.info(f"FFT长度已设置为: {length}, 瀑布图尺寸: {length}x{length}")

        # 重新初始化图像数组
        with self.image_lock:
            self.waterfall_image = np.zeros(
                (self.waterfall_height, self.waterfall_width, 3), dtype=np.uint8
            )

    def set_averaging(self, enable, count=10):
        """设置移动平均（可选功能）"""
        self.enable_averaging = enable
        self.averaging_count = count
        self.history_buffer = deque(maxlen=count)
        logging.info(f"移动平均: {'启用' if enable else '禁用'}, 窗口: {count}")
