import socket
import struct
import numpy as np
import time
import threading
import logging
import json
from pathlib import Path
from scipy.io import loadmat

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
        self.command_thread = None

        # ⭐ 扫描模式参数
        self.single_channel_fft = 512  # 单通道FFT点数
        self.channel_count = 20  # 固定20通道
        self.total_fft_length = self.single_channel_fft * self.channel_count  # 10240
        self.send_interval = 0.001  # 发送间隔

        # 数据流相关
        self.data_dir = Path(__file__).parent.parent.parent / "2"
        self.npy_files = sorted(self.data_dir.glob("*.npy"))
        self._current_file_idx = 0
        self._buffer = np.array([], dtype=np.float32)

        # 发送帧数
        self.frame_id = 0

        logging.info(
            f"MockDevice initialized: single_channel_fft={self.single_channel_fft}, "
            f"total_fft_length={self.total_fft_length}"
        )

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

            # 启动指令接收线程
            self.command_thread = threading.Thread(
                target=self._command_loop, daemon=True
            )
            self.command_thread.start()

        except Exception as e:
            logging.error(f"启动失败: {e}")

    def stop(self):
        """停止模拟设备"""
        self.running = False
        if self.send_thread:
            self.send_thread.join(timeout=2)
        if self.command_thread:
            self.command_thread.join(timeout=2)
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        logging.info("模拟设备已停止")

    def _command_loop(self):
        """指令接收循环"""
        logging.info("📡 指令接收线程已启动")

        while self.running:
            try:
                # 接收指令码（1字节）
                code_data = self._recv_exact(1)
                if not code_data:
                    break

                code = struct.unpack(">B", code_data)[0]

                # 根据code解析参数
                if code == 0x01:  # SET_FFT_LENGTH
                    value_data = self._recv_exact(4)
                    if not value_data:
                        break
                    new_length = struct.unpack(">I", value_data)[0]

                    # ⭐ 更新单通道FFT长度
                    self.single_channel_fft = new_length
                    self.total_fft_length = self.single_channel_fft * self.channel_count

                    logging.info(
                        f"✓ 接收到指令: SET_FFT_LENGTH = {new_length}, "
                        f"total_fft_length = {self.total_fft_length}"
                    )

                else:
                    logging.warning(f"⚠ 未知指令码: 0x{code:02X}")

            except Exception as e:
                if self.running:
                    logging.error(f"指令接收异常: {e}", exc_info=True)
                break

        logging.info("指令接收线程已退出")

    def _recv_exact(self, num_bytes):
        """精确接收指定字节数"""
        data = bytearray()
        while len(data) < num_bytes:
            try:
                packet = self.client_socket.recv(num_bytes - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except Exception as e:
                if self.running:
                    logging.error(f"接收数据错误: {e}")
                return None
        return bytes(data)

    def _generate_raw_fft_data(self):
        """生成单通道512点原始FFT数据（从npy文件读取）"""
        if not self.npy_files:
            raise RuntimeError(f"未在目录 {self.data_dir} 中找到任何.npy文件")

        # 确保缓冲区有足够的数据
        while self._buffer.size < self.single_channel_fft:
            next_chunk = self._load_next_file_chunk()
            if next_chunk.size == 0:
                continue
            if self._buffer.size == 0:
                self._buffer = next_chunk
            else:
                self._buffer = np.concatenate((self._buffer, next_chunk))

        # 提取单通道FFT数据
        raw_fft_data = self._buffer[: self.single_channel_fft]
        self._buffer = self._buffer[self.single_channel_fft :]
        return raw_fft_data

    def _prepare_full_frame(self, raw_fft_data):
        """
        ⭐ 模拟下位机数据准备逻辑

        Args:
            raw_fft_data: 原始单通道FFT数据（512点）

        Returns:
            完整帧数据（10240点）
        """
        # 1. 找到512点中的最小值
        min_value = np.min(raw_fft_data)

        # 2. 创建完整缓冲区（10240点）
        full_frame = np.zeros(self.total_fft_length, dtype=np.float32)
        full_frame[:5120] = min_value  # 前半部分填充最小值
        # 3. 拷贝原始512点
        full_frame[5120 : 5120 + self.single_channel_fft] = raw_fft_data

        # 4. 填充剩余点为最小值
        full_frame[5120 + self.single_channel_fft :] = min_value

        return full_frame

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
                print(f"加载文件 {file_path}，数据形状: {data.shape}")
            except Exception as exc:
                logging.error(f"加载文件 {file_path} 失败: {exc}", exc_info=True)
                continue
            # 将数据压缩到512*512
            # 如果数据维度是512*514，则进行转置
            if data.shape[0] == 512 and data.shape[1] == 514:
                data = data.T
            data = data[:512, :512]
            flat_data = np.asarray(data, dtype=np.float32).ravel()
            if flat_data.size == 0:
                logging.warning(f"文件 {file_path} 为空，跳过")
                continue

            return flat_data

        logging.error("无法从任何npy文件中获取有效数据")
        return np.array([], dtype=np.float32)

    def _send_loop(self):
        """数据发送循环 - 发送10240点完整帧"""
        while self.running:
            try:
                # ⭐ 1. 生成单通道512点原始FFT数据
                raw_fft_data = self._generate_raw_fft_data()

                # ⭐ 2. 准备完整10240点帧（模拟下位机逻辑）
                full_frame = self._prepare_full_frame(raw_fft_data)

                self.frame_id += 1
                header = struct.pack(
                    ">III", 0xAABBCCDD, self.frame_id, len(full_frame) * 4
                )
                self.client_socket.sendall(header + full_frame.tobytes())
                # 日志输出（每1000帧输出一次）
                if self.frame_id % 1000 == 0:
                    logging.info(
                        f"已发送 {self.frame_id} 帧数据 "
                        f"(每帧{self.total_fft_length}点, "
                        f"原始{self.single_channel_fft}点, "
                        f"最小值={np.min(raw_fft_data):.4f})"
                    )

            except Exception as e:
                if self.running:
                    logging.error(f"发送数据异常: {e}", exc_info=True)
                break

        # 重新初始化连接
        if self.running:
            logging.warning("发送线程异常退出，尝试重新连接...")
            self.start()
        else:
            logging.info("发送线程已退出")

    def set_fft_length(self, length):
        """设置单通道FFT长度"""
        self.single_channel_fft = length
        self.total_fft_length = self.single_channel_fft * self.channel_count
        logging.info(
            f"单通道FFT长度已设置为: {length}, 总长度: {self.total_fft_length}"
        )


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
