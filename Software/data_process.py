import threading
import queue
import time
import numpy as np
from collections import deque
import logging


class DataProcessor:
    def __init__(self, state):
        self.state = state
        self.fft_data_queue = None  # 从Communication接收的原始FFT数据
        self.processed_data_queue = None  # 处理后供UI使用的数据
        self.process_thread = None

        # 数据处理参数
        self.enable_averaging = False
        self.averaging_count = 10
        self.history_buffer = deque(maxlen=self.averaging_count)

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
                processed_data = self._process_fft_data(fft_data)

                # 放入UI队列（非阻塞）
                try:
                    self.processed_data_queue.put_nowait(
                        {
                            "timestamp": timestamp,
                            "spectrum": processed_data,
                            "max_value": np.max(processed_data),
                            "min_value": np.min(processed_data),
                        }
                    )
                except queue.Full:
                    # UI队列满，丢弃最旧数据
                    try:
                        self.processed_data_queue.get_nowait()
                        self.processed_data_queue.put_nowait(
                            {
                                "timestamp": timestamp,
                                "spectrum": processed_data,
                                "max_value": np.max(processed_data),
                                "min_value": np.min(processed_data),
                            }
                        )
                    except:
                        pass

                self.state.data_queue_status = "processing"

            except queue.Empty:
                self.state.data_queue_status = "idle"
                continue
            except Exception as e:
                logging.error(f"数据处理异常: {e}")
                time.sleep(0.1)

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
