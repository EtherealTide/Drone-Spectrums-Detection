import socket
import struct
import numpy as np
import time
import threading
import logging

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
        self.fft_length = 4096
        self.packet_size = 128  # 每次发送128个点
        self.send_interval = 0.2  # 100ms发送一帧完整FFT

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
        """生成模拟FFT数据"""
        # 生成基础噪声
        noise = np.random.normal(0, 0.1, self.fft_length)

        # 添加几个峰值信号（模拟无人机信号）
        signal = np.zeros(self.fft_length)

        # 峰值1: 中心频率附近
        peak1_idx = self.fft_length // 2 + 100
        signal[peak1_idx - 5 : peak1_idx + 5] = np.hamming(10) * 2.0

        # 峰值2: 偏移频率
        peak2_idx = self.fft_length // 2 - 200
        signal[peak2_idx - 3 : peak2_idx + 3] = np.hamming(6) * 1.5

        # 峰值3: 随机位置（模拟干扰）
        peak3_idx = np.random.randint(100, self.fft_length - 100)
        signal[peak3_idx - 2 : peak3_idx + 2] = np.hamming(4) * 0.8

        # 合成最终信号
        fft_data = signal + noise

        return fft_data.astype(np.float32)

    def _send_loop(self):
        """数据发送循环"""
        packet_id = 0
        frame_count = 0

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

                    # 构造数据包: [packet_id(4字节)] + [data_length(4字节)] + [data]
                    header = struct.pack(">II", packet_id, len(packet_data))

                    # 发送
                    self.client_socket.sendall(header + packet_data)

                    packet_id += 1

                    # 小延迟模拟网络传输
                    time.sleep(0.001)

                frame_count += 1
                logging.info(
                    f"已发送第 {frame_count} 帧完整FFT (packets: {packet_id-num_packets} - {packet_id-1})"
                )

                # 等待下一帧
                time.sleep(self.send_interval)

            except Exception as e:
                if self.running:
                    logging.error(f"发送数据异常: {e}")
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
