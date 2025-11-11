import socket
import struct
import numpy as np
import time
import threading
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class MockDevice:
    """模拟下位机设备，发送FFT数据"""

    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.send_thread = None

        # FFT参数
        self.fft_length = 512
        self.packet_size = 128  # 每次发送128个点
        self.send_interval = 0.001  # 100ms发送一帧完整FFT

        # 数据流相关
        self.data_dir = Path(__file__).parent / "2"
        self.npy_files = sorted(self.data_dir.glob("*.npy"))
        self._current_file_idx = 0
        self._buffer = np.array([], dtype=np.float32)

    def start(self):
        """启动模拟设备"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            logging.info(f"模拟设备启动，监听 {self.host}:{self.port}")

            # 等待连接
            logging.info("等待上位机连接...")
            self.client_socket, addr = self.server_socket.accept()
            logging.info(f"上位机已连接: {addr}")

            # 启动数据发送线程
            self.running = True
            self.send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self.send_thread.start()

        except Exception as e:
            logging.error(f"启动失败: {e}")

    def stop(self):
        """停止模拟设备"""
        self.running = False
        if self.send_thread:
            self.send_thread.join(timeout=2)
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        logging.info("模拟设备已停止")

    def _generate_fft_data(self):
        """从指定目录中读取npy文件并转换为数据流"""
        if not self.npy_files:
            raise RuntimeError(f"未在目录 {self.data_dir} 中找到任何.npy文件")

        while self._buffer.size < self.fft_length:
            next_chunk = self._load_next_file_chunk()
            if next_chunk.size == 0:
                continue
            if self._buffer.size == 0:
                self._buffer = next_chunk
            else:
                self._buffer = np.concatenate((self._buffer, next_chunk))

        fft_data = self._buffer[: self.fft_length]
        self._buffer = self._buffer[self.fft_length :]
        return fft_data

    def _load_next_file_chunk(self):
        """加载下一个有效的npy文件数据"""
        attempts = 0
        total_files = len(self.npy_files)
        while attempts < total_files:
            file_path = self.npy_files[self._current_file_idx]
            self._current_file_idx = (self._current_file_idx + 1) % total_files
            attempts += 1

            try:
                data = np.load(file_path)
            except Exception as exc:
                logging.error(f"加载文件 {file_path} 失败: {exc}", exc_info=True)
                continue

            flat_data = np.asarray(data, dtype=np.float32).ravel()
            if flat_data.size == 0:
                logging.warning(f"文件 {file_path} 为空，跳过")
                continue

            return flat_data

        logging.error("无法从任何npy文件中获取有效数据")
        return np.array([], dtype=np.float32)

    def _send_loop(self):
        """数据发送循环 - 每个包前加魔数"""
        while self.running:
            try:
                # 生成一帧完整的FFT数据
                fft_data = self._generate_fft_data()

                # 分包发送
                num_packets = self.fft_length // self.packet_size

                for i in range(num_packets):
                    # 提取当前包的数据
                    start_idx = i * self.packet_size
                    end_idx = start_idx + self.packet_size
                    packet_data = fft_data[start_idx:end_idx].tobytes()

                    # 构造数据包: [magic(4)] + [packet_id(4)] + [data_length(4)] + [data]
                    header = struct.pack(
                        ">III",
                        0xAABBCCDD,  # 魔数
                        i,  # 包ID（帧内序号，从0开始）
                        len(packet_data),  # 数据长度
                    )

                    # 发送
                    self.client_socket.sendall(header + packet_data)
                    # time.sleep(0.001)

                logging.info(f"已发送一帧 ({num_packets} 个包)")

                # 等待下一帧
                # time.sleep(self.send_interval)

            except Exception as e:
                if self.running:
                    logging.error(f"发送数据异常: {e}", exc_info=True)
                break

        logging.info("发送线程已退出")

    def set_fft_length(self, length):
        """设置FFT长度"""
        self.fft_length = length
        logging.info(f"FFT长度已设置为: {length}")


if __name__ == "__main__":
    # 创建并启动模拟设备
    device = MockDevice(host="127.0.0.1", port=5000)

    try:
        device.start()

        # 保持运行
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("收到停止信号")
        device.stop()
